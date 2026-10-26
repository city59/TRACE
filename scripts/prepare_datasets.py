"""Unpack the bundled datasets into ./dataset/ and verify their checksums.

Taobao (tmall_beh) and IJCAI (ijcai_beh) are stored as xz-compressed tar
archives in dataset/archives/ because their training files exceed GitHub's
per-file size limit; the IJCAI archive is additionally split into parts.
Retailrocket (retail_beh) is small and is stored uncompressed.

Usage (from the repository root):
    python scripts/prepare_datasets.py
"""

import argparse
import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / 'dataset'
ARCHIVE_DIR = DATASET_DIR / 'archives'
ARCHIVES = ('tmall_beh', 'ijcai_beh')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def expected_checksums():
    checksums = {}
    for line in (ARCHIVE_DIR / 'SHA256SUMS').read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            checksums[name.lstrip('*')] = digest
    return checksums


def archive_stream(name):
    single = ARCHIVE_DIR / f'{name}.tar.xz'
    if single.is_file():
        return open(single, 'rb')
    parts = sorted(ARCHIVE_DIR.glob(f'{name}.tar.xz.part*'))
    if not parts:
        raise FileNotFoundError(f'No archive found for {name} in {ARCHIVE_DIR}')
    return io.BytesIO(b''.join(part.read_bytes() for part in parts))


def files_ok(name, checksums):
    files = [f for f in checksums if f.startswith(name + '/')]
    return bool(files) and all(
        (DATASET_DIR / f).is_file() and sha256(DATASET_DIR / f) == checksums[f]
        for f in files
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--force', action='store_true',
                        help='Re-extract even if verified files already exist.')
    args = parser.parse_args()

    checksums = expected_checksums()
    for name in ARCHIVES:
        if not args.force and files_ok(name, checksums):
            print(f'{name}: already present')
            continue
        print(f'{name}: extracting ...')
        with archive_stream(name) as stream, tarfile.open(fileobj=stream, mode='r:xz') as tar:
            for member in tar.getmembers():
                target = (DATASET_DIR / member.name).resolve()
                if DATASET_DIR.resolve() not in target.parents:
                    raise RuntimeError(f'Unsafe path in archive: {member.name}')
            tar.extractall(DATASET_DIR)

    failed = [f for f, digest in checksums.items()
              if not (DATASET_DIR / f).is_file() or sha256(DATASET_DIR / f) != digest]
    if failed:
        raise SystemExit('Checksum mismatch: ' + ', '.join(failed))
    print(f'All {len(checksums)} dataset files verified in {DATASET_DIR}')


if __name__ == '__main__':
    main()
