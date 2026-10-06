"""Generate a synthetic MS/MS MGF file for profiling SpectroVQ.

Spectra are built from theoretical b/y fragment ions of random tryptic peptides
(charge 1 and 2, with neutral losses and isotopes) plus random noise peaks, so
peak counts and the m/z distribution resemble real Orbitrap DDA MS2 data.
The model's compute cost does not depend on content, so timings are representative.
"""
import argparse

import numpy as np
from pyteomics import mass, mgf

AA = 'ACDEFGHIKLMNPQRSTVWY'
PROTON = 1.007276


def random_peptide(rng):
    length = rng.integers(7, 25)
    seq = ''.join(rng.choice(list(AA), size=length - 1))
    return seq + rng.choice(['K', 'R'])


def fragment_peaks(seq, rng):
    mz, it = [], []
    for i in range(1, len(seq)):
        for ion, frag in (('b', seq[:i]), ('y', seq[i:])):
            for z in (1, 2):
                m = mass.fast_mass(frag, ion_type=ion, charge=z)
                base = rng.lognormal(8, 1.2)
                mz += [m, m + 1.00336 / z]
                it += [base, base * rng.uniform(0.1, 0.6)]
                if rng.random() < 0.3:  # water / ammonia loss
                    mz.append(m - rng.choice([18.0106, 17.0265]) / z)
                    it.append(base * rng.uniform(0.05, 0.3))
    keep = rng.random(len(mz)) < 0.6  # not every ion is observed
    return np.array(mz)[keep], np.array(it)[keep]


def make_spectrum(idx, rng):
    seq = random_peptide(rng)
    charge = int(rng.choice([2, 3], p=[0.6, 0.4]))
    precursor = mass.fast_mass(seq, charge=charge)
    mz, it = fragment_peaks(seq, rng)
    n_noise = rng.integers(50, 300)
    noise_mz = rng.uniform(100, min(2000, precursor * charge), n_noise)
    noise_it = rng.lognormal(6, 1.0, n_noise)
    mz = np.concatenate([mz, noise_mz])
    it = np.concatenate([it, noise_it])
    sel = (mz > 100) & (mz < 2000)
    order = np.argsort(mz[sel])
    return {
        'm/z array': np.round(mz[sel][order], 4),
        'intensity array': np.round(it[sel][order], 1),
        'params': {
            'title': f'Synthetic.{idx}.{idx}.{charge}',
            'rtinseconds': float(rng.uniform(300, 7200)),
            'pepmass': (precursor, float(rng.lognormal(13, 1))),
            'charge': [charge],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--out', default='synthetic.mgf')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    mgf.write((make_spectrum(i, rng) for i in range(a.n)), a.out)


if __name__ == '__main__':
    main()
