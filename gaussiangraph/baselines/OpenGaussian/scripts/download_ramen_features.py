"""Download and SHA256-verify third-party LangSplat-style features for a baseline.

These are NOT the OpenGaussian authors' original preprocessed features.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
repo = 'JUNHAKBAE/THGS-lerf-ovs-language-features'
meta = requests.get(f'https://huggingface.co/api/datasets/{repo}', timeout=30)
meta.raise_for_status()
revision = meta.json()['sha']
url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/ramen/language_features?limit=1000'
response = requests.get(url, timeout=30)
response.raise_for_status()
entries = [f for f in response.json() if f['type'] == 'file']
args.root.mkdir(parents=True, exist_ok=True)
manifest = {'repository': repo, 'revision': revision, 'files': entries,
            'note': 'Third-party preprocessing; not official OpenGaussian features'}
(args.root / 'feature_source.json').write_text(json.dumps(manifest, indent=2))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def download(entry):
    path = args.root / entry['path']
    checksum = entry['lfs']['oid']
    if path.is_file() and path.stat().st_size == entry['size'] and digest(path) == checksum:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            with requests.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{entry["path"]}',
                              stream=True, timeout=(20, 120)) as r:
                r.raise_for_status()
                with partial.open('wb') as handle:
                    for chunk in r.iter_content(1024 * 1024):
                        handle.write(chunk)
            if digest(partial) != checksum:
                raise ValueError(f'Checksum mismatch: {entry["path"]}')
            partial.replace(path)
            return
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for count, _ in enumerate(pool.map(download, entries), 1):
        if count % 20 == 0 or count == len(entries):
            print(f'{count}/{len(entries)} verified', flush=True)
