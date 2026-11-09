"""Inspect free-text, full-gallery image sets without catalogue-label scoring."""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HUB_CACHE', str(ROOT / 'models/huggingface/hub'))
from curio.retrieve.engine import SearchService, load_items  # noqa: E402
from curio.retrieve.visual_sets import VisualSetService  # noqa: E402

QUERIES = (
    ('porcelain', 'porcelain'), ('bronze', 'bronze'),
    ('green_vase', 'green narrow-necked porcelain vase'),
    ('dragon_porcelain', 'blue and white porcelain with dragon decoration'),
    ('lotus_bowl', 'blue and white porcelain bowl with lotus flowers'),
    ('embroidered', 'embroidered textile with flowers'),
    ('ming_porcelain', 'Ming dynasty blue and white porcelain'),
    ('no_car', 'a red sports car'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/visual_set_review_v2_20261108')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(ROOT / 'data/snapshots/collection_v2.jsonl')
    search = SearchService(items, index_dir=ROOT / 'artifacts/siglip_adapted', encoder_name='siglip', root=ROOT, probe_weight=0)
    service = VisualSetService(search)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    summaries = []
    for qid, query in QUERIES:
        for method in ('adaptive_margin', 'adaptive_graph', 'adaptive_evidence'):
            result = service.search(query, method=method)
            (out / f'{qid}_{method}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            scores = search._dense(result['encoded_query'])
            margin = scores - service._generic
            selected = set(result['matched_item_ids'])
            accepted = np.asarray([i for i, item in enumerate(items) if item['item_id'] in selected], dtype=int)
            rng = np.random.default_rng(5523)
            random_yes = rng.choice(accepted, size=min(8, len(accepted)), replace=False).tolist()
            strong = sorted(accepted.tolist(), key=lambda i: (-margin[i], items[i]['item_id']))[:4]
            rejected = [i for i, item in enumerate(items) if item['item_id'] not in selected]
            near = sorted(rejected, key=lambda i: (-margin[i], items[i]['item_id']))[:4]
            samples = [(i, 'strong yes') for i in strong] + [(i, 'random yes') for i in random_yes] + [(i, 'near no') for i in near]
            sheet = Image.new('RGB', (1200, 70 + 270 * ((len(samples) + 3) // 4)), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f'{qid} / {method}: {len(selected)} yes of {len(items)} images', fill='black', font=font)
            draw.text((8, 30), query, fill='black', font=font)
            review = []
            for k, (i, role) in enumerate(samples):
                item = items[i]
                x, y = (k % 4) * 300, 70 + (k // 4) * 270
                with Image.open(ROOT / item['image_path']) as opened:
                    image = opened.convert('RGB')
                    image.thumbnail((280, 204))
                    sheet.paste(image, (x + (300 - image.width) // 2, y))
                draw.text((x + 8, y + 210), f'{item["item_id"]} | {role}', fill='black', font=font)
                draw.text((x + 8, y + 232), f'cos={scores[i]:.3f} margin={margin[i]:+.3f}', fill='black', font=font)
                review.append({'item_id': item['item_id'], 'role': role})
            sheet.save(out / f'{qid}_{method}.jpg', quality=92)
            summaries.append({'qid': qid, 'query': query, 'method': method, 'matched_count': len(selected),
                              'boundary': result['boundary'], 'review_samples': review})
            print(qid, method, len(selected), flush=True)
    (out / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
