"""Convert (a random sample of) a SpectraST binary .splib library into MGF.

Used to get realistic peptide MS/MS spectra for profiling from the libraries
published with SpectroVQ on Zenodo (record 20043624, splibLibraries.zip).
"""
import argparse
import struct

import numpy as np
from pyteomics import mgf


def read_line(f):
    out = bytearray()
    while True:
        c = f.read(1)
        if not c or c == b'\n':
            return out.decode('latin-1')
        out += c


def iter_splib(path):
    with open(path, 'rb') as f:
        f.read(8)                      # file version, sub-version
        read_line(f)                   # library file name
        n_lines, = struct.unpack('<i', f.read(4))
        for _ in range(n_lines):
            read_line(f)               # preamble lines
        while True:
            head = f.read(4)           # library id
            if len(head) < 4:
                return
            name = read_line(f)
            precursor_mz, = struct.unpack('<d', f.read(8))
            read_line(f)               # status
            n_peaks, = struct.unpack('<I', f.read(4))
            mz = np.empty(n_peaks)
            it = np.empty(n_peaks)
            for k in range(n_peaks):
                mz[k], it[k] = struct.unpack('<dd', f.read(16))
                read_line(f)           # annotation
                read_line(f)           # info
            comment = read_line(f)
            yield name, precursor_mz, mz, it, comment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splib', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--keep_prob', type=float, default=0.05,
                   help='Probability of keeping each entry (spreads the sample over the library)')
    p.add_argument('--include_decoys', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    spectra = []
    for i, (name, pmz, mz, it, comment) in enumerate(iter_splib(a.splib)):
        if not a.include_decoys and 'DECOY' in (name + comment).upper():
            continue
        if rng.random() > a.keep_prob:
            continue
        charge = int(name.split('/')[1].split()[0]) if '/' in name else 2
        order = np.argsort(mz)
        spectra.append({'m/z array': mz[order], 'intensity array': it[order],
                        'params': {'title': f'splib.{i}.{i}.{charge} {name}', 'pepmass': (pmz, None),
                                   'charge': [charge], 'rtinseconds': 0.0}})
        if len(spectra) >= a.n:
            break
    mgf.write(spectra, a.out)
    peaks = [len(s['m/z array']) for s in spectra]
    print(f'wrote {len(spectra)} spectra, peaks/spectrum mean {np.mean(peaks):.1f} median {np.median(peaks):.0f}')


if __name__ == '__main__':
    main()
