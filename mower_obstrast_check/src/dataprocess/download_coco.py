"""Download original COCO 2017 images containing any of six requested categories."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/coco2017'
META = ROOT / 'data/manifests/coco2017'
BASE = 'https://s3.amazonaws.com/images.cocodataset.org'
NAMES = ('person', 'cat', 'dog', 'sports ball', 'frisbee', 'teddy bear')
LOCAL = threading.local()


def session():
    """Reuse one HTTP session per worker, retrying transient HTTP failures."""
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
        LOCAL.session.mount('https://', HTTPAdapter(max_retries=retry))
    return LOCAL.session


def save_json(path, data):
    """Publish a complete JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.json.part')
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    partial.replace(path)


def digest(path, algorithm):
    """Hash a file without loading it into memory."""
    result = hashlib.new(algorithm)
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def annotations(workers):
    """Keep the official ZIP and extract byte-identical instance annotation JSONs."""
    url = BASE + '/annotations/annotations_trainval2017.zip'
    archive = RAW / 'archives/annotations_trainval2017.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    with session().head(url, timeout=(20, 60)) as response:
        response.raise_for_status()
        size = int(response.headers['Content-Length'])
        etag = response.headers['ETag'].strip('"')
    if not archive.exists():
        chunks = archive.parent / 'annotation_parts'
        chunks.mkdir(exist_ok=True)
        starts = range(0, size, 4 * 1024 * 1024)

        def fetch(start):
            end = min(start + 4 * 1024 * 1024, size) - 1
            part = chunks / str(start)
            if part.exists() and part.stat().st_size == end - start + 1:
                return part
            headers = {'Range': f'bytes={start}-{end}', 'If-Match': '"' + etag + '"'}
            with session().get(url, headers=headers, timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                    raise ValueError('Unexpected annotation range response')
                payload = response.content
            if len(payload) != end - start + 1:
                raise ValueError('Incomplete annotation chunk')
            partial = part.with_suffix('.part')
            partial.write_bytes(payload)
            partial.replace(part)
            return part

        print(f'Downloading official annotations: {size / 1048576:.1f} MiB', flush=True)
        parts, received = [], 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(fetch, starts):
                parts.append(part)
                received += part.stat().st_size
                save_json(META / 'download_status.json', {'state': 'downloading_annotations', 'pid': os.getpid(),
                                                         'received_bytes': received, 'total_bytes': size})
                print(f'Annotations: {received}/{size} bytes', flush=True)
        partial = archive.with_suffix('.zip.part')
        with partial.open('wb') as target:
            for part in parts:
                with part.open('rb') as source:
                    shutil.copyfileobj(source, target)
        if partial.stat().st_size != size or digest(partial, 'md5') != etag:
            raise ValueError('Official archive size/ETag integrity check failed')
        partial.replace(archive)
        for part in parts:
            part.unlink()  # Remove only the verified, assembled download chunks.
        chunks.rmdir()
    if archive.stat().st_size != size or digest(archive, 'md5') != etag:
        raise ValueError('Existing official archive failed integrity check')
    with zipfile.ZipFile(archive) as package:
        for split in ('train2017', 'val2017'):
            name = f'annotations/instances_{split}.json'
            target = RAW / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                partial = target.with_suffix('.json.part')
                with package.open(name) as source, partial.open('wb') as output:
                    shutil.copyfileobj(source, output)
                partial.replace(target)
    save_json(META / 'source.json', {'dataset': 'COCO 2017', 'source': 'https://cocodataset.org/#download',
                                    'archive_url': url, 'archive_etag': etag,
                                    'archive_sha256': digest(archive, 'sha256'), 'requested_classes': NAMES,
                                    'note': 'Original annotations retain all 80 categories and all original records.'})


def select_images():
    """Select the union of target-category image IDs; do not clean or remap labels."""
    jobs, summary = [], {}
    for split in ('val2017', 'train2017'):
        path = RAW / f'annotations/instances_{split}.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        categories = {c['id']: c['name'] for c in data['categories'] if c['name'] in NAMES}
        if set(categories.values()) != set(NAMES):
            raise ValueError('Requested COCO categories were not found')
        groups = {key: set() for key in categories}
        instances = {key: 0 for key in categories}
        for ann in data['annotations']:
            if ann['category_id'] in groups:
                groups[ann['category_id']].add(ann['image_id'])
                instances[ann['category_id']] += 1
        selected = set().union(*groups.values())
        images = sorted((im for im in data['images'] if im['id'] in selected), key=lambda im: im['id'])
        summary[split] = {'unique_images': len(images), 'categories': {
            name: {'coco_category_id': key, 'images': len(groups[key]), 'instances': instances[key]}
            for key, name in categories.items()}}
        save_json(META / f'selection_{split}.json', {'split': split, 'images': images, 'statistics': summary[split]})
        (RAW / 'images' / split).mkdir(parents=True, exist_ok=True)
        jobs.extend((split, im['file_name']) for im in images)
    save_json(META / 'selection_summary.json', summary)
    return jobs, summary


def fetch_image(job):
    """Download an original JPEG, checking HTTP length and JPEG boundary markers."""
    split, name = job
    target = RAW / 'images' / split / name
    if target.name != name:
        raise ValueError('Unexpected COCO filename')
    if not target.exists():
        with session().get(f'{BASE}/{split}/{name}', timeout=(20, 60)) as response:
            response.raise_for_status()
            payload = response.content
            expected = response.headers.get('Content-Length')
        if expected and len(payload) != int(expected):
            raise ValueError('Incomplete image HTTP response')
        if not payload.startswith(b'\xff\xd8') or not payload.rstrip().endswith(b'\xff\xd9'):
            raise ValueError('Invalid or truncated JPEG')
        partial = target.with_suffix('.jpg.part')
        partial.write_bytes(payload)
        partial.replace(target)
    with target.open('rb') as source:
        if source.read(2) != b'\xff\xd8':
            raise ValueError('Existing file is not a JPEG')
    return {'split': split, 'file_name': name, 'bytes': target.stat().st_size, 'sha256': digest(target, 'sha256')}


def main():
    """Prepare official annotations and download all selected images with a progress ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be between 1 and 32')
    save_json(META / 'download_status.json', {'state': 'preparing', 'pid': os.getpid()})
    annotations(args.workers)
    jobs, summary = select_images()
    print(json.dumps(summary), flush=True)
    if args.prepare_only:
        save_json(META / 'download_status.json', {'state': 'prepared', 'expected_images': len(jobs)})
        return
    status = {'state': 'downloading', 'pid': os.getpid(), 'expected_images': len(jobs),
              'verified_images': 0, 'failed_images': 0}
    failures, last = [], time.monotonic()
    save_json(META / 'download_status.json', status)
    with (META / 'downloaded_images.jsonl').open('w', encoding='utf-8') as ledger:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_image, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    record = future.result()
                    ledger.write(json.dumps(record) + '\n')
                    status['verified_images'] += 1
                except Exception as exc:
                    failures.append({'image': futures[future], 'error': str(exc)})
                    status['failed_images'] += 1
                if time.monotonic() - last > 20:
                    ledger.flush()
                    save_json(META / 'download_status.json', status)
                    print(json.dumps(status), flush=True)
                    last = time.monotonic()
    status['state'] = 'complete' if not failures else 'incomplete'
    save_json(META / 'failed_images.json', failures)
    save_json(META / 'download_status.json', status)
    print(json.dumps(status), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save_json(META / 'download_status.json', {'state': 'failed', 'error': str(error)})
        raise
