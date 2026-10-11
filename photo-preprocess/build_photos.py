"""Build photos/ from the original iPhone photos (Google Drive download).

    pip install pillow pillow-heif
    python photo-preprocess/build_photos.py "<unzipped drive folder>"

Reads photo_map.csv (melon_file, drive_name) next to this script, finds each
IMG_xxxx photo in the source folder (any extension, subfolders searched),
fixes EXIF rotation, resizes to 2576 px wide and saves NN_vK.jpg and NN_id.jpg
into photos/ at the repo root. Same conversion as step 2 of watermelon_features.ipynb.
"""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

TARGET_WIDTH = 2576      # every threshold in the notebook was tuned at this width
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / 'photos'   # repo-root photos/, which the notebook reads


def prepare(src, dst):
    """Fix EXIF rotation, convert to RGB, resize to TARGET_WIDTH, save as jpg."""
    with Image.open(src) as raw:
        im = ImageOps.exif_transpose(raw).convert('RGB')
    if im.width != TARGET_WIDTH:
        im = im.resize((TARGET_WIDTH, round(im.height * TARGET_WIDTH / im.width)), Image.LANCZOS)
    im.save(dst, quality=92)


def main(src_dir):
    by_stem = {p.stem.upper(): p for p in Path(src_dir).rglob('*')
               if p.suffix.lower() in {'.heic', '.jpg', '.jpeg', '.png'}}
    with open(HERE / 'photo_map.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    OUT_DIR.mkdir(exist_ok=True)
    done = 0
    for r in rows:
        src = by_stem.get(Path(r['drive_name']).stem.upper())
        if src is None:
            print('MISSING', r['drive_name'], '->', r['melon_file'])
            continue
        prepare(src, OUT_DIR / r['melon_file'])
        done += 1
    print(f'done: {done}/{len(rows)}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
