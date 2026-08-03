#!/usr/bin/env python3
"""
Build a 3-class leash dataset (leashed / unleashed / not_a_dog) for the crop
classifier, using crops the DETECTOR actually produces rather than the
annotators' boxes.

Why not just add the "other animal" label as a third class
---------------------------------------------------------
The annotators' ``other animal`` boxes have a median short side of 36 px
against 139 px for dogs. A classifier trained on those learns "small means
other animal" -- a size shortcut that scores well in validation and then fails
on exactly the hard case, the distant street dog. Measured on real imagery the
detector emits crops with a median short side of 88 px, so the third class has
to live in the same size range as the dog classes.

So the negative class is harvested from the detector's own false positives:
run the detector over every annotated image, keep each predicted box that does
NOT overlap a dog ground-truth box, and label it ``not_a_dog``. Those crops are
by construction the ones that will reach the classifier in production -- non-dog
animals, but also poles, bags and shadows the detector trips on. Annotator
``other animal`` boxes are folded in only above --min-size, so they cannot
reintroduce the size shortcut.

Crops are cut from the FULL-RESOLUTION originals under the grid_runs roots, not
from the 1280px-wide copies in archived_datasets/, because the production
pipeline reads the full-res files: a crop taken from a downscaled copy is ~3x
smaller than the one the classifier will actually be asked to judge.

The Label Studio SQLite database is always copied before reading, never opened
in place -- it belongs to a running server.

    python build_leash3_dataset.py \\
        --ls-db ~/label-studio/data/label_studio.sqlite3 \\
        --image-roots /path/to/grid_runs ... \\
        --out leash_3class

Read-only with respect to every source: the only writes are the dataset under
--out and the database copy under --work-dir.
"""

import argparse
import glob
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

DOG_LABELS = {'leashed dog': 'leashed', 'unleashed dog': 'unleashed'}
# Everything an annotator can draw that is definitively not a dog.
NON_DOG_LABELS = {'other animal', 'monkey', 'cow', 'sheep', 'goat', 'item'}


def copy_db(src, work_dir):
    """Copy the Label Studio DB (plus any WAL/SHM) and return the copy's path."""
    os.makedirs(work_dir, exist_ok=True)
    dst = os.path.join(work_dir, 'label_studio_copy.sqlite3')
    shutil.copy(src, dst)
    for ext in ('-wal', '-shm'):
        if os.path.exists(src + ext):
            shutil.copy(src + ext, dst + ext)
    return dst


def read_annotations(db, project_id):
    """{image_id: {'dogs': [(cls, xyxy_norm)], 'others': [xyxy_norm]}}.

    Boxes come back in normalised xyxy so they are independent of whichever
    resolution the annotator happened to see.
    """
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    rows = con.execute(
        "SELECT t.data, tc.result FROM task_completion tc "
        "JOIN task t ON tc.task_id = t.id "
        "WHERE t.project_id = ? AND tc.was_cancelled = 0", [project_id])
    out = {}
    for data, result in rows:
        try:
            image = json.loads(data).get('image', '')
            items = json.loads(result) if result else []
        except (ValueError, TypeError):
            continue
        iid = os.path.basename(image).rsplit('.', 1)[0]
        if not iid:
            continue
        rec = out.setdefault(iid, {'dogs': [], 'others': []})
        for it in items:
            v = it.get('value', {})
            labels = v.get('rectanglelabels') or []
            if not labels or 'width' not in v:
                continue
            # Label Studio stores x/y/width/height as percentages.
            box = (v['x'] / 100, v['y'] / 100, (v['x'] + v['width']) / 100,
                   (v['y'] + v['height']) / 100)
            if labels[0] in DOG_LABELS:
                rec['dogs'].append((DOG_LABELS[labels[0]], box))
            elif labels[0] in NON_DOG_LABELS:
                rec['others'].append(box)
    con.close()
    return out


def _scan_jpgs(d, idx):
    """Add every <id>.jpg directly inside ``d`` to ``idx``. First root wins."""
    n = 0
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name.endswith('.jpg'):
                    idx.setdefault(e.name[:-4], e.path)
                    n += 1
    except OSError:
        pass
    return n


def index_images(roots):
    """{image_id: path} for the jpgs under each root.

    Accepts both layouts without being told which: the cell tree that the
    harvest writes (``<root>/<cell>/ground_animal_images/<id>.jpg``) and a flat
    directory of ``<id>.jpg`` (e.g. the wasabi upload staging dir). Passing a
    flat dir to a cell-tree-only scan silently yields zero matches, and the run
    then looks like "those images just aren't on disk" -- so try both and report
    what each root actually contributed.
    """
    idx = {}
    for root in roots:
        before = len(idx)
        cells = 0
        for d in glob.glob(os.path.join(root, '*', 'ground_animal_images')):
            cells += _scan_jpgs(d, idx)
        flat = _scan_jpgs(root, idx)  # flat layout, or loose files at root
        print(f'    {root}: +{len(idx) - before:,} new '
              f'({cells:,} in cell tree, {flat:,} flat)')
    return idx


def iou(a, b):
    """IoU of two xyxy boxes in the same coordinate space."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def crop(img, box, pad_frac):
    """Cut an xyxy pixel box with padding, clamped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    pad = pad_frac * max(x2 - x1, y2 - y1)
    x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    x2, y2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img[y1:y2, x1:x2]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ls-db',
                   required=True,
                   help='Label Studio sqlite3 file (copied before reading).')
    p.add_argument('--project-id', type=int, default=12)
    p.add_argument('--image-roots',
                   nargs='+',
                   required=True,
                   help='Roots holding the FULL-RES originals. Either layout '
                   'works and they can be mixed: a grid_runs cell tree '
                   '(<root>/<cell>/ground_animal_images/) or a flat '
                   'directory of <image_id>.jpg. Earlier roots win.')
    p.add_argument(
        '--fallback-dirs',
        nargs='+',
        default=[],
        help='Extra dirs searched when an id is not in --image-roots '
        '(e.g. archived_datasets/*/images/*; note these are '
        'downscaled, so crops from them are smaller).')
    p.add_argument('--weights', required=True, help='Detector .pt weights.')
    p.add_argument('--out', default='leash_3class')
    p.add_argument('--work-dir', default='.leash3_work')
    p.add_argument(
        '--conf',
        type=float,
        default=0.05,
        help='Detector confidence floor (default 0.05: the negative '
        'class must cover the low-confidence junk too).')
    p.add_argument('--iou-nms', type=float, default=0.9)
    p.add_argument('--iou-match',
                   type=float,
                   default=0.5,
                   help='IoU at which a predicted box counts as the GT dog.')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument(
        '--pad',
        type=float,
        default=0.15,
        help='Fraction of box size added as context (default 0.15).')
    p.add_argument('--min-size',
                   type=int,
                   default=64,
                   help='Drop crops whose short side is under this. Also the '
                   'floor for folding in annotator non-dog boxes, which '
                   'is what keeps the negative class from becoming '
                   '"anything small".')
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--limit',
                   type=int,
                   help='Only process N images (smoke test).')
    args = p.parse_args()

    from ultralytics import YOLO

    print('copying Label Studio DB (never read in place)...')
    db = copy_db(args.ls_db, args.work_dir)
    ann = read_annotations(db, args.project_id)
    print(f'  annotated images: {len(ann):,}')

    print('indexing full-res originals...')
    idx = index_images(args.image_roots)
    print(f'  indexed: {len(idx):,}')
    for d in args.fallback_dirs:
        for pth in glob.glob(d + '/*'):
            idx.setdefault(os.path.basename(pth).rsplit('.', 1)[0], pth)

    todo = [(i, idx[i]) for i in sorted(ann) if i in idx]
    missing = len(ann) - len(todo)
    print(f'  resolved {len(todo):,} images ({missing:,} unresolved)')
    if args.limit:
        todo = todo[:args.limit]

    model = YOLO(args.weights)
    rng = random.Random(args.seed)
    # Split by IMAGE, not by crop: two crops from one photo share background and
    # lighting, so splitting by crop leaks the val set into training.
    val_ids = {i for i, _ in todo if rng.random() < args.val_frac}

    counts = Counter()
    sizes = defaultdict(list)
    for cls in ('leashed', 'unleashed', 'not_a_dog'):
        for split in ('train', 'val'):
            os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)

    def emit(iid, n, cls, img, box):
        cc = crop(img, box, args.pad)
        if cc is None or min(cc.shape[:2]) < args.min_size:
            counts['skipped_too_small'] += 1
            return
        split = 'val' if iid in val_ids else 'train'
        cv2.imwrite(os.path.join(args.out, split, cls, f'{iid}_{n}.jpg'), cc)
        counts[f'{split}/{cls}'] += 1
        sizes[cls].append(min(cc.shape[:2]))

    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        results = model.predict([p for _, p in chunk],
                                imgsz=args.imgsz,
                                half=True,
                                conf=args.conf,
                                iou=args.iou_nms,
                                verbose=False,
                                save=False)
        for (iid, path), r in zip(chunk, results):
            img = cv2.imread(path)
            if img is None:
                counts['unreadable'] += 1
                continue
            h, w = img.shape[:2]
            gt_dogs = [(c, (b[0] * w, b[1] * h, b[2] * w, b[3] * h))
                       for c, b in ann[iid]['dogs']]
            gt_other = [(b[0] * w, b[1] * h, b[2] * w, b[3] * h)
                        for b in ann[iid]['others']]
            pred = (r.boxes.xyxy.float().cpu().numpy()
                    if len(r.boxes) else np.zeros((0, 4)))
            n = 0
            for pb in pred:
                best_c, best_i = None, 0.0
                for c, gb in gt_dogs:
                    v = iou(pb, gb)
                    if v > best_i:
                        best_c, best_i = c, v
                if best_i >= args.iou_match:
                    emit(iid, n, best_c, img, pb)  # detector found the dog
                else:
                    # No dog here: exactly the crop that misleads a 2-class model.
                    emit(iid, n, 'not_a_dog', img, pb)
                n += 1
            # Annotator non-dog boxes, but only in the dog size range.
            for gb in gt_other:
                if min(gb[2] - gb[0], gb[3] - gb[1]) >= args.min_size:
                    emit(iid, n, 'not_a_dog', img, gb)
                    n += 1
        if (start // args.batch) % 25 == 0:
            print(f'  [{start + len(chunk)}/{len(todo)}] {dict(counts)}',
                  flush=True)

    print('\n=== dataset ===')
    for k in sorted(counts):
        print(f'  {k:<24}{counts[k]:>7,}')
    print('\ncrop short side by class (they must overlap, or the classifier '
          'can separate on size alone):')
    for cls, v in sizes.items():
        a = np.array(v)
        print(
            f'  {cls:<12} n={len(a):>6}  p10={np.percentile(a, 10):>5.0f} '
            f'p50={np.percentile(a, 50):>5.0f} p90={np.percentile(a, 90):>5.0f}'
        )
    with open(os.path.join(args.out, 'build_manifest.json'), 'w') as f:
        json.dump(
            {
                'counts': dict(counts),
                'args': vars(args),
                'n_images': len(todo),
                'unresolved': missing
            },
            f,
            indent=1)
    print(
        f'\n-> {args.out}    train with: '
        f'yolo classify train data={args.out} model=yolo26x-cls.pt imgsz=640')
    return 0


if __name__ == '__main__':
    sys.exit(main())
