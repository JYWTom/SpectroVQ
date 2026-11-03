import argparse
import math
import os
import shutil
import sys
from datetime import timedelta
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.fabric.accelerators import CUDAAccelerator
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities.exceptions import SIGTERMException
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import get_cosine_schedule_with_warmup

from utils.model import modelStream
from utils.SpectrumProcessing.SpectraLoading import massSpectrumToVector


def find0arr(arr):
    return len(arr["intensity"])


def PreprocessData(data_path):
    print("Loading Training Data ....")
    df_train = pd.read_feather(data_path)
    df_train = df_train.rename(
        columns={"Name": "CompoundName", "Annotation": "Peak Comments"}
    )
    df_train["Charge"] = df_train["CompoundName"].str.extract(r"(\d+)$").astype(int)

    df_train.rename(
        columns={
            "IT": "intensity",
            "MZ": "m/z",
        },
        inplace=True,
    )

    df_train["Peptide"] = df_train["CompoundName"].apply(lambda x: x.split("/")[0])
    df_train["Peptide"] = df_train["Peptide"].str.replace(
        r"\[[^\]]*\]", "X", regex=True
    )
    df_train["Peptide"] = df_train["Peptide"].str.replace(r"[^A-Za-z]", "", regex=True)

    df_train["length"] = df_train.apply(find0arr, axis=1)
    df_train = df_train[df_train["length"] > 10]
    df_train.drop("length", inplace=True, axis=1)
    df_train["isHLA"] = df_train["Comment"].str.contains("_HLA_").astype(int)

    df_train["PeptideLen"] = df_train["Peptide"].apply(lambda x: len(x) > 10)
    from sklearn.model_selection import train_test_split

    df_train["class"] = df_train.apply(
        lambda x: str(x["PeptideLen"]) + str(x["Charge"]), axis=1
    )
    train_data, test_data = train_test_split(df_train, test_size=0.05, random_state=42)

    # Print the shapes of the training and testing sets
    print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)

    train_data.reset_index(drop=True, inplace=True)

    return train_data, test_data


def transformToTensor(x):
    return torch.from_numpy(np.array(x.to_list(), dtype="float32"))


Bin_size = 0.1
# Fixed, also on resume: every data order and augmentation below derives from it.
BASE_SEED = 42
# Exit code of a graceful pause (EX_TEMPFAIL): train.py can be continued with --resume.
EXIT_PAUSED = 75


class SpectrumAugmentation:
    """Spectrum noise. Every method draws from the generator it is given (see
    SpectrumBatch.__getitem__), so the noise of a sample does not depend on earlier samples."""

    def __init__(self, returntype="torch"):
        if returntype == "torch":
            self.returntype = torch.from_numpy
        elif returntype == "numpy":
            self.returntype = None
        else:
            raise NotImplementedError("returntype must be either torch or numpy")

    def MovingAverage(self, x, w):
        """
        x: input signal
        w: kernel size
        """
        return np.convolve(x, np.ones(w), "valid") / w

    def AddGausianNoise(self, x, rng):
        """
        x: input signal
        rng: numpy random generator
        """
        return x + rng.normal(0, 0.001, size=x.shape)

    def ModifyMZ(self, mz, rng):
        MZModified = rng.binomial(len(mz), 0.25)
        Changeidx = rng.integers(low=0, high=len(mz), size=MZModified)
        for i in Changeidx:
            mz[i] = mz[i] + rng.normal(0, 0.01)
        return mz

    def ApplyMovingAverage(self, x, rng):
        MovingAverage = rng.binomial(len(x), 0.5)
        Changeidx = rng.integers(low=0, high=len(x) - 11, size=MovingAverage)
        for u in Changeidx:
            if (x[u] == 0) & (x[u + 1] == 0):
                continue
            else:
                x[u : u + 11] = self.MovingAverage(x[u : u + 11], w=11)
        return x

    def AddPeriodicPeaks(self, x, rng):
        NoisePeakWavelength = rng.random(x.shape[0])
        magnitude = rng.gamma(1, 0.001, x.shape[0])
        temp = np.linspace(0, 10, x.shape[0])
        periodicPeaks = (np.sin(2 * np.pi * NoisePeakWavelength * temp) + 1) * magnitude
        return x + periodicPeaks

    def AddRandomPeaks(self, x, rng, lambdaPoisson=1):
        # Generate simulated electronic noises
        NumberNoisePeaks = rng.integers(1, 150)
        Position = rng.integers(low=0, high=x.shape[-1], size=NumberNoisePeaks)
        # Generating Poisson-distributed random variables
        intensity = rng.poisson(lam=lambdaPoisson, size=NumberNoisePeaks)
        intensity = intensity / 100
        for i in range(NumberNoisePeaks):
            x[Position[i]] += intensity[i]
        return x


class SpectrumBatch(Dataset):
    def __init__(self, df):
        df.sort_values(by="PrecursorMZ", inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.df = df
        self.Xmz = df["m/z"]
        self.Xintensity = df["intensity"]
        self.peptide = df["Peptide"]
        self.PrecursorMz = df["PrecursorMZ"]
        self.PrecursorMz = self.PrecursorMz.apply(lambda x: float(x))
        self.length = df.shape[0]
        self.base_seed = BASE_SEED
        self.epoch = 0  # set before each epoch's workers start; part of the sample seed
        self.Augmentation = SpectrumAugmentation(returntype="torch")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Per sample, so a sample's augmentation doesn't depend on which worker loads it or
        # on what was loaded before it, and a resumed epoch can start anywhere.
        rng = np.random.default_rng([self.base_seed, self.epoch, index])
        mz, it = self.Xmz.iloc[index], self.Xintensity.iloc[index]
        precursormz = self.PrecursorMz.iloc[index]
        # Peak Masking
        delete_mask = rng.binomial(1, 0.1, size=len(mz))
        mz = mz[delete_mask == 0]
        it = it[delete_mask == 0]
        spectra, _ = massSpectrumToVector(
            self.Xmz.iloc[index],
            self.Xintensity.iloc[index],
            bin_size=Bin_size,
            SPECTRA_DIMENSION=13500,
            rawIT=False,
            Mode=None,
            mean0=False,
            CenterIntegerBins=False,
            AlterMZ=False,
            GenerateMZList=False,
            MZDiff=False,
            mzRange=[150, 1500],
        )
        augmentedSpectra, _ = massSpectrumToVector(
            mz,
            it,
            bin_size=Bin_size,
            SPECTRA_DIMENSION=13500,
            rawIT=False,
            Mode=None,
            mean0=False,
            CenterIntegerBins=False,
            AlterMZ=False,
            GenerateMZList=False,
            MZDiff=False,
            mzRange=[150, 1500],
            returntorch=False,
        )
        # Electronic Noises
        if rng.binomial(1, 0.5):
            augmentedSpectra = self.Augmentation.AddRandomPeaks(
                augmentedSpectra, rng, lambdaPoisson=5
            )
        # Peptide Noises
        if rng.binomial(1, 0.5):
            currentPrecursorMz = precursormz
            mixupMz = rng.uniform(
                currentPrecursorMz - 5, currentPrecursorMz + 5
            )  # 5 Th Tolerance
            mixupidx = np.argmin(np.abs(self.PrecursorMz - mixupMz))
            N = 0
            while mixupidx == index:
                mixupMz = rng.uniform(
                    currentPrecursorMz - 5, currentPrecursorMz + 5
                )  # 5 Th Tolerance
                mixupidx = np.argmin(np.abs(self.PrecursorMz - mixupMz))
                N += 1
                if N > 10:
                    break
            mixupidx = np.clip(mixupidx, a_min=0, a_max=self.length - 1)
            mixupSpectra, _ = massSpectrumToVector(
                self.Xmz.iloc[mixupidx],
                self.Xintensity.iloc[mixupidx],
                bin_size=Bin_size,
                SPECTRA_DIMENSION=13500,
                rawIT=False,
                Mode=None,
                mean0=False,
                CenterIntegerBins=False,
                AlterMZ=False,
                GenerateMZList=False,
                MZDiff=False,
                mzRange=[150, 1500],
                returntorch=False,
            )
            mix = rng.uniform(0.1, 0.5, size=1)
            augmentedSpectra = np.clip(
                (augmentedSpectra * (1 - mix) + mixupSpectra * mix), a_min=0, a_max=1
            )
        return torch.unsqueeze(torch.sqrt(spectra), dim=0), torch.unsqueeze(
            torch.sqrt(torch.from_numpy(augmentedSpectra)), 0
        ).float()


class ResumableSampler(Sampler):
    """Shuffles from (seed, epoch) alone, and can start an epoch after its finished batches
    by index, so no data is loaded to skip them."""

    def __init__(self, dataset, batch_size, seed=BASE_SEED):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.epoch, self.skip_batches = 0, 0

    def set_epoch(self, epoch):  # called by Lightning before each epoch
        # The dataset is copied to the workers when they start: persistent_workers must stay off.
        self.epoch = self.dataset.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset), generator=g).tolist()
        skip, self.skip_batches = self.skip_batches, 0
        return iter(order[skip * self.batch_size :])

    def __len__(self):
        # The full epoch, so Lightning's batch count stays stable.
        return len(self.dataset)


eps = 1e-6
cos2 = nn.CosineSimilarity(dim=2, eps=1e-6)


def SpectralConstrastLoss(output, target):
    Similarity = cos2(output, target)
    dotProduct = (2 * torch.acos(Similarity)) / (math.pi)
    return dotProduct.mean()


def rescaleSepectra(Spectra, dim):
    SpectraMax = torch.max(Spectra, dim=dim, keepdim=True)[0]
    return Spectra / SpectraMax


def spectral_entropy(spectrum, dim=2, reweigh=False):
    if reweigh:
        spectrum = torch.where(
            spectrum >= 3,
            torch.tensor(1.0),
            0.25 + spectrum * torch.log(spectrum + eps),
        )
    return -torch.sum(spectrum * torch.log(spectrum + eps), dim=dim)


# https://www.nature.com/articles/s41592-021-01331-z#Sec9
def spectral_entropy_loss(input, target, dim=2):
    inputspectrum = input / (torch.sum(input, dim=dim, keepdim=True) + eps)
    targetspectrum = target / (torch.sum(target, dim=dim, keepdim=True) + eps)
    SpectrumAB = inputspectrum + targetspectrum
    SpectrumAB = (
        SpectrumAB / 2
    )  # (torch.sum(SpectrumAB,dim = dim,keepdim = True) + eps)
    spectral_entropyAB = spectral_entropy(SpectrumAB, dim=dim)
    spectral_entropyA = spectral_entropy(inputspectrum, dim=dim)
    spectral_entropyB = spectral_entropy(targetspectrum, dim=dim)
    return 1 - (
        ((2 * spectral_entropyAB) - spectral_entropyA - spectral_entropyB) / np.log(4)
    )


Threshold = 1e-2


def peak_portion(input, target, dim=2):
    inputspectrum = torch.clamp(input / Threshold, min=0, max=1)
    numinputpeaks = torch.sum(inputspectrum, dim=dim, keepdim=True)
    targetspectrum = torch.clamp(target / Threshold, min=0, max=1)
    numtargetpeaks = torch.sum(targetspectrum, dim=dim, keepdim=True)
    mismatch = torch.abs(inputspectrum - targetspectrum)
    noiseRatio = torch.sum(mismatch, dim=dim) / (numinputpeaks + numtargetpeaks + eps)
    return noiseRatio


weightCS, weightSE = 1.0, 1.0


class LightningModelSpectraStream(L.LightningModule):
    def __init__(
        self, model, learning_rate, len_train_loader, len_test_loader, max_epochs
    ):
        super().__init__()
        self.model = model
        self.max_epochs = max_epochs
        self.len_train_loader = len_train_loader
        self.len_test_loader = len_test_loader
        self.learning_rate = learning_rate
        self.AverageTestLossList = []
        self.AverageTrainingLoss = []
        self.rng = np.random.default_rng(42)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        spectra, aug_spectra = batch
        vq = bool(self.rng.integers(0, 2))
        output, quantize_loss, _ = self.model(aug_spectra, use_vq=vq)
        OrigCosineLoss = SpectralConstrastLoss(output, spectra)

        QuantizeLoss = quantize_loss.mean()
        SE_Loss = spectral_entropy_loss(output, spectra, dim=2).mean()

        loss = weightSE * SE_Loss + weightCS * OrigCosineLoss + QuantizeLoss

        if batch_idx % 5000 == 0:
            print(".", end="")
            print(f"Training Loss: {loss.item()}")
            print(f"SpectralConstrastiveLoss: {OrigCosineLoss.item()}")
            print(f"QuantizeLoss: {QuantizeLoss.item()}")
            print(f"SELoss: {SE_Loss.item()}")

        if torch.isnan(loss):
            print(f"NAN Loss Detected at Training batch {batch_idx}")
        else:
            self.AverageTrainingLoss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):

        spectra, _ = batch

        output, _, _ = self.model(spectra)

        OrigCosineLoss = SpectralConstrastLoss(output, spectra)

        SE_Loss = spectral_entropy_loss(output, spectra, dim=2).mean()

        loss = OrigCosineLoss + SE_Loss

        if batch_idx % 100 == 0:
            print(f"Testing Loss: {loss.item()}")
            print(f"SpectralConstrastiveLoss: {OrigCosineLoss.item()}")
            print(f"SELoss: {SE_Loss.item()}")

        if torch.isnan(loss):
            print(f"NAN Loss Detected at Testing batch {batch_idx}")
        else:
            self.AverageTestLossList.append(loss.item())
        self.log("val_loss", loss, sync_dist=True, on_epoch=True)

    def on_train_epoch_end(self):
        self.AverageTrainingLoss.clear()
        self.rng = np.random.default_rng(42 + self.rng.integers(0, 1000))

    def on_validation_epoch_end(self):
        AverageTLoss = np.mean(self.AverageTestLossList)
        self.log("val_mean_loss", AverageTLoss, sync_dist=True)
        self.AverageTestLossList.clear()
        val_loader = self.trainer.val_dataloaders
        assert val_loader is not None  # this hook only runs when validating
        val_loader.dataset.epoch += 1

    def configure_optimizers(self):
        warmup_epochs = 7
        total_epochs = self.max_epochs
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        num_train_optimization_steps = total_epochs * self.len_train_loader
        num_warmup_steps = warmup_epochs * self.len_train_loader
        lr_scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_optimization_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class ResumeState(Callback):
    """Sampler position and RNG states, stored in the checkpoint (what Lightning leaves out).

    Holds only ints, lists, dicts of those and tensors: ApplyWeights reads checkpoints with
    the default torch.load.
    """

    # Set in setup(), which Lightning calls before any checkpoint is saved.
    trainer: L.Trainer
    module: LightningModelSpectraStream

    def __init__(self, sampler, val_dataset):
        self.sampler, self.val_dataset = sampler, val_dataset
        self.batches_done = 0  # finished this epoch
        self._rng_to_restore = None
        self._should_stop = False

    def setup(self, trainer, pl_module, stage):
        self.trainer, self.module = trainer, pl_module

    def _generators(self):
        """The model's numpy generators: VQ on/off and quantizer dropout."""
        gens = {"module_rng": self.module.rng}
        if hasattr(self.module.model, "rng"):
            gens["model_rng"] = self.module.model.rng
        return gens

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches_done += 1

    def on_train_epoch_end(self, trainer, pl_module):
        self.batches_done = 0

    def state_dict(self):
        rng = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            rng["mps"] = torch.mps.get_rng_state()
        for name, gen in self._generators().items():
            rng[name] = gen.bit_generator.state
        return {
            # What Lightning passes to set_epoch, not current_epoch (which reads `completed`).
            # A _ProcessedTracker at runtime; Lightning declares the narrower base type.
            "epoch": self.trainer.fit_loop.epoch_progress.current.processed,  # ty: ignore[unresolved-attribute]
            "batches_done": self.batches_done,
            "val_epoch": self.val_dataset.epoch,
            "should_stop": bool(self.trainer.should_stop),
            "rng": rng,
        }

    def load_state_dict(self, state_dict):
        # Before Lightning creates the data iterator (fit_loop.setup_data), which is built
        # before its _set_sampler_epoch runs.
        self.sampler.set_epoch(state_dict["epoch"])
        self.sampler.skip_batches = self.batches_done = state_dict["batches_done"]
        self.val_dataset.epoch = state_dict["val_epoch"]
        self._should_stop = state_dict["should_stop"]
        self._rng_to_restore = state_dict["rng"]

    def on_train_start(self, trainer, pl_module):
        # Not on load: creating the data iterator draws a seed from the torch RNG. Restoring
        # after it puts the RNG back where the uninterrupted run had it.
        rng, self._rng_to_restore = self._rng_to_restore, None
        if rng is None:
            return
        # Lightning resets this on every fit, so an early stop just before a pause would be lost.
        trainer.should_stop = self._should_stop
        torch.set_rng_state(rng["torch"].cpu())
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if "mps" in rng and torch.backends.mps.is_available():
            torch.mps.set_rng_state(rng["mps"].cpu())
        for name, gen in self._generators().items():
            gen.bit_generator.state = rng[name]


class ResumeCheckpoint(ModelCheckpoint):
    """Periodic resume checkpoint, also written on a graceful stop (SIGTERM)."""

    def on_train_batch_end(self, trainer, *args, **kwargs):
        if not trainer.is_last_batch:  # a save here may skip validation on resume
            super().on_train_batch_end(trainer, *args, **kwargs)

    def on_exception(self, trainer, pl_module, exception):
        # During validation the loss average is in memory only: leave the last periodic save.
        if isinstance(exception, SIGTERMException) and not trainer.validating:
            super().on_exception(trainer, pl_module, exception)


model_kwargs2: dict[str, Any] = {
    "inputChannel": 1,
    "n_q": 12,
    "n_filters": 32,
    "n_residual_layers": 1,
    "lstm": 0,
    "ratios": [9, 7, 4, 2],
    "final_activation": "Sigmoid",
    "residual_kernel_size": 6,
    "codebook_size": 1024,
    "FiLMEncoder": False,
    "FiLMDecoder": False,
    "flattenQuantization": False,
    "quantizationdim": 256,
    "biLSTM": False,
    "causal": False,
    "useQINCo": False,
    "inputSize": 13500,
    "dimension": 256,
    "kernel_size": 8,
    "encoders": False,
    "bucketTransform": False,
    "applyquantizerdroput": True,
    "transformer": 6,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Training script for SpectroVQ")
    parser.add_argument(
        "--data_path",
        type=str,
        default="library.feather",
        help="data path to the consensus library file (.feather)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="output directory for the trained model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loading"
    )

    # Model hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="maximum number of training epochs"
    )

    # Pause and resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from <output_dir>/resume/resume.ckpt; use the same --batch_size, "
        "--max_epochs, --deterministic and library as the first run",
    )
    parser.add_argument(
        "--checkpoint_minutes",
        type=float,
        default=30,
        help="minutes between resume checkpoints",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use deterministic algorithms, for exact GPU resumes (slower)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Lightning accelerator: auto, cpu, cuda, mps",
    )
    return parser.parse_args(argv)


def main(argv=None, extra_callbacks=()):
    args = parse_args(argv)
    output_dir = args.output_dir
    resume_dir = os.path.join(os.path.abspath(output_dir), "resume")
    resume_ckpt = os.path.join(resume_dir, "resume.ckpt")
    if args.resume and not os.path.exists(resume_ckpt):
        sys.exit(f"Nothing to resume: {resume_ckpt} does not exist.")
    if not args.resume and os.path.exists(resume_ckpt):
        sys.exit(
            f"{resume_ckpt} exists, so a paused run would be overwritten. Continue it with "
            "--resume, or use another output directory (or delete the file) to start over."
        )
    # Pause with: kill -TERM <pid>, to this process only, not its data loader workers.
    print(f"Training process PID: {os.getpid()}", flush=True)

    seed_everything(BASE_SEED, workers=True)
    torch.set_float32_matmul_precision("high")

    if CUDAAccelerator.is_available():
        torch.backends.cudnn.benchmark = True

    print("Starting Training Program")

    num_epochs = args.max_epochs
    SpectraStream = modelStream.SpectroVQ(**model_kwargs2)
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    print("Preprocessing Training Data")
    traindata, testdata = PreprocessData(args.data_path)
    print("Creating Training Data Loaders")
    spectraTrainData = SpectrumBatch(traindata)
    train_sampler = ResumableSampler(spectraTrainData, batch_size)
    train_loader = DataLoader(
        spectraTrainData,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=24 if args.num_workers > 0 else None,
    )
    spectraTestData = SpectrumBatch(testdata)
    test_loader = DataLoader(
        spectraTestData,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=24 if args.num_workers > 0 else None,
    )
    LightningTrainer = LightningModelSpectraStream(
        SpectraStream,
        learning_rate=learning_rate,
        len_train_loader=len(train_loader),
        len_test_loader=len(test_loader),
        max_epochs=num_epochs,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        monitor="val_mean_loss",
        filename="SpectroVQ-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_mean_loss",
        min_delta=0.005,
        patience=4,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=False,
    )
    resume_callback = ResumeCheckpoint(
        dirpath=resume_dir,  # a subfolder: the sessions sidebar scans only the top level
        filename="resume",
        monitor=None,
        save_top_k=1,
        enable_version_counter=False,  # always overwrite one file
        train_time_interval=timedelta(minutes=args.checkpoint_minutes),
        save_on_exception=True,
    )
    trainer = L.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
        callbacks=[
            ResumeState(train_sampler, spectraTestData),
            early_stop_callback,
            checkpoint_callback,
            resume_callback,
            *extra_callbacks,
        ],
        accelerator=args.accelerator,
        devices=1,
        num_nodes=1,
        strategy="auto",
        gradient_clip_val=1,
        deterministic=True if args.deterministic else None,
    )
    try:
        trainer.fit(
            LightningTrainer,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
            ckpt_path=resume_ckpt if args.resume else None,
        )
    except SIGTERMException:
        # Otherwise it exits 0 and looks finished.
        print(
            "Paused. Continue with --resume, using the same settings."
            if os.path.exists(resume_ckpt)
            else "Paused before any resume checkpoint was written: nothing to continue from.",
            flush=True,
        )
        sys.exit(EXIT_PAUSED)
    shutil.rmtree(resume_dir, ignore_errors=True)  # finished: free about 619 MB


if __name__ == "__main__":
    main()
