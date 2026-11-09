"""Export real full-gallery set results and unbiased visual review samples."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from curio.retrieve.engine import load_items  # noqa: E402
from curio.retrieve.probe_sets import ProbeSetService  # noqa: E402


def main():
    out = ROOT / 'artifacts/catalogue_set_visual_review_v5'
    out.mkdir(parents=True, exist_ok=True)
    index = ROOT / 'artifacts/siglip'
    ids = json.loads((index / 'item_ids.json').read_text(encoding='utf-8'))
    catalogue = {row['item_id']: row for row in load_items(ROOT / 'data/snapshots/collection_v2.jsonl')}
    vectors = np.load(index / 'image_embeddings.npy', allow_pickle=False)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    heads = ROOT / 'artifacts/catalogue_set_runtime_v5'
    service = ProbeSetService([catalogue[item] for item in ids], vectors,
                              probes_path=heads, set_report_path=heads / 'set_policy.json')
    queries = [('porcelain', '瓷器'), ('bronze', '青铜'), ('porcelain_bowl', '瓷碗'),
               ('bronze_mirror', '青铜镜'), ('jade_pendant', '玉佩'),
               ('embroidered', '刺绣'), ('dragon_porcelain', '龙纹瓷器'),
               ('ming_porcelain', '明代瓷器')]
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    summaries = []
    for qid, query in queries:
        try:
            result = service.search(query, top_k=100)
        except ValueError as exc:
            summaries.append({'qid': qid, 'query': query, 'unsupported': exc.args[0]})
            continue
        (out / f'{qid}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        atoms = service._atoms(query)[0]
        margins = np.min(np.stack([
            service._probabilities[task][:, service._classes[task].index(value)] - service._thresholds[(task, value)]
            for task, _, value in atoms], axis=1), axis=1)
        selected = set(result['matched_item_ids'])
        accepted = np.asarray([i for i, item in enumerate(ids) if item in selected], dtype=int)
        rng = np.random.default_rng(5523)
        random_accepted = rng.choice(accepted, size=min(8, len(accepted)), replace=False).tolist()
        strong = sorted(accepted.tolist(), key=lambda i: (-margins[i], ids[i]))[:4]
        rejected = [i for i, item in enumerate(ids) if item not in selected]
        near = sorted(rejected, key=lambda i: (-margins[i], ids[i]))[:4]
        samples = [(i, 'strong yes') for i in strong] + [(i, 'random yes') for i in random_accepted] + [(i, 'near no') for i in near]
        sheet = Image.new('RGB', (1200, 60 + 280 * ((len(samples) + 3) // 4)), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'{qid}: {len(selected)} yes / {len(ids)} gallery. Fixed p>=.55; not calibrated accuracy.', fill='black', font=font)
        review = []
        for k, (i, role) in enumerate(samples):
            item = catalogue[ids[i]]
            x, y = (k % 4) * 300, 60 + (k // 4) * 280
            with Image.open(ROOT / item['image_path']) as opened:
                image = opened.convert('RGB')
                image.thumbnail((280, 212))
                sheet.paste(image, (x + (300 - image.width) // 2, y))
            draw.text((x + 8, y + 218), f'{ids[i]} | {role}', fill='black', font=font)
            draw.text((x + 8, y + 240), f'min margin {margins[i]:+.3f}', fill='black', font=font)
            review.append({'item_id': ids[i], 'role': role, 'margin': float(margins[i])})
        sheet.save(out / f'{qid}.jpg', quality=92)
        summaries.append({'qid': qid, 'query': query, 'matched_count': len(selected), 'review_samples': review})
        print(qid, len(selected), flush=True)
    (out / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
