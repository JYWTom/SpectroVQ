"""The Model panel's text, and loading the model it names (utils.model.loader)."""

from gui import config
from gui.errors import shows_errors
from utils.errors import UserError
from utils.model import loader
from utils.settings import read_settings, remember_model


def initial_ckpt():
    """Checkpoint shown when a page opens: --ckpt wins, else the last one that loaded."""
    return config.CKPT or read_settings(config.CONFIG_PATH).get("ckpt", "")


def initial_yaml():
    return read_settings(config.CONFIG_PATH).get("yaml", "")


def initial_model_status():
    ckpt = initial_ckpt()
    if not ckpt:
        return "_No model loaded yet. It is loaded automatically on first use._"
    source = "from --ckpt" if config.CKPT else "last used"
    return (
        f"_Checkpoint ({source}): `{ckpt}`. It is loaded automatically on first use._"
    )


@shows_errors
def load_model(ckpt_path, yaml_path, device, compile_mode="none"):
    """The model for one run, with the Model panel's settings; remembered for the next start."""
    model = loader.load_model(ckpt_path, yaml_path, device, compile_mode)
    remember_model(config.CONFIG_PATH, ckpt_path.strip(), (yaml_path or "").strip())
    return model


def model_status(ckpt_path, yaml_path, device, compile_mode="none"):
    try:
        model = loader.load_model(ckpt_path, yaml_path, device, compile_mode)
    except UserError as e:
        return f"⚠️ {e}"
    except Exception as e:  # noqa: BLE001
        return f"⚠️ Failed to load model: {e}"
    remember_model(config.CONFIG_PATH, ckpt_path.strip(), (yaml_path or "").strip())
    d = loader.describe(model)
    return (
        f"✅ Loaded on **{d['device']}** · {d['parameters']:.1f}M parameters · "
        f"{d['quantizers']} quantizers · {d['codes']} codes per quantizer per spectrum · "
        f"torch.compile {'on' if d['compiled'] else 'off'}"
    )
