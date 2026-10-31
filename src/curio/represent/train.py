"""Image-tower LoRA with frozen texts, multi-positive loss and object holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageOps

from .captions import visual_caption
from .encoders import Encoder
from .validation import validate_object_splits


def read_items(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def multi_positive_loss(image_features, text_features, positive_mask, temperature=0.07):
    logits = (
        F.normalize(image_features.float(), dim=-1)
        @ F.normalize(text_features.float(), dim=-1).T
        / temperature
    )
    mask = positive_mask.float()
    forward = -(F.log_softmax(logits, dim=1) * mask).sum(1) / mask.sum(1)
    backward = -(F.log_softmax(logits.T, dim=1) * mask.T).sum(1) / mask.T.sum(1)
    return (forward.mean() + backward.mean()) / 2


def make_batches(order, batch_size, neighbors=None):
    """Cover every object once; optionally pack nearby frozen-teacher images."""
    if neighbors is None:
        return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    available = np.ones(len(order), dtype=bool)
    batches = []
    for seed in order:
        if not available[seed]:
            continue
        candidates = neighbors[seed]
        candidates = candidates[(candidates != seed) & available[candidates]]
        indices = np.concatenate(([seed], candidates[: batch_size - 1]))
        available[indices] = False
        batches.append(indices)
    return batches


def paired_retrieval(image_features, text_features, captions):
    """All exact-equal captions are positives; report text->image only."""
    scores = text_features @ image_features.T
    order = np.argsort(-scores, axis=1, kind="stable")
    captions = np.asarray(captions)
    first = np.array(
        [np.flatnonzero(captions[row] == captions[i])[0] + 1 for i, row in enumerate(order)]
    )
    return {
        "r@1": float((first <= 1).mean()),
        "r@5": float((first <= 5).mean()),
        "r@10": float((first <= 10).mean()),
        "mrr": float((1 / first).mean()),
        "queries": len(captions),
        "gallery": len(captions),
        "unique_captions": len(set(captions)),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    items = read_items(args.data)
    validate_object_splits(items)
    train_items = [x for x in items if x["split"] == "train"]
    dev_items = [x for x in items if x["split"] == "dev"]
    annotations_path = getattr(args, "visual_annotations", None)
    annotations = {}
    if annotations_path:
        from .detail_probes import load_annotations

        annotations = load_annotations(annotations_path, items)
        train_items = [x for x in train_items if x["item_id"] in annotations]
        dev_items = [x for x in dev_items if x["item_id"] in annotations]
    if not train_items or len(dev_items) < 10:
        raise ValueError("Need object-disjoint train and dev items (at least 10 dev)")
    if args.max_train:
        train_items = train_items[: args.max_train]
    encoder = Encoder(args.encoder, args.device)
    caption = (lambda x: annotations[x["item_id"]]["caption_en"]) if annotations else visual_caption
    train_captions = [caption(x) for x in train_items]
    dev_captions = [caption(x) for x in dev_items]
    dev_text = encoder.encode_texts(dev_captions)
    train_text = torch.from_numpy(encoder.encode_texts(train_captions))
    dev_paths = [x["image_path"] for x in dev_items]
    base_dev_features = encoder.encode_images(dev_paths, args.batch_size)
    baseline = paired_retrieval(base_dev_features, dev_text, dev_captions)
    # Teacher image embeddings anchor visual semantics against small-domain collapse.
    teacher_path = output / "teacher_train.npy"
    teacher = encoder.encode_images([x["image_path"] for x in train_items], args.batch_size)
    np.save(teacher_path, teacher)
    neighbors = None
    if args.sampling == "hard":
        # Neighbors use train images only. Exact-equal captions remain positives.
        neighbors = np.argsort(-(teacher @ teacher.T), axis=1, kind="stable")
    teacher = torch.from_numpy(teacher)
    target_modules = [
        name
        for name, _ in encoder.model.named_modules()
        if "vision_model" in name and name.endswith(("q_proj", "v_proj"))
    ]
    if not target_modules:
        raise RuntimeError("No vision attention projections found; refusing a fake training run")
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(encoder.model, config)
    encoder.model = model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=encoder.device.type == "cuda")
    steps_per_epoch = (len(train_items) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min((step + 1) / max(1, total_steps * 0.05), 1)
            * 0.5
            * (1 + np.cos(np.pi * step / max(1, total_steps)))
        ),
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = {
        "config": vars(args),
        "encoder": encoder.metadata,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "train_items": len(train_items),
        "dev_items": len(dev_items),
        "trainable_parameters": params,
        "baseline_dev": baseline,
        "caption_policy": (
            "AI image-only blind-card captions; no museum metadata or retriever rankings supplied to annotators"
            if annotations
            else "visibility-oriented weak metadata v1; removes dynasty/date/culture/source; title and medium can retain inferred use/identity. Exact repeated captions are multiple positives"
        ),
        "visual_annotations_sha256": hashlib.sha256(Path(annotations_path).read_bytes()).hexdigest()
        if annotations_path
        else None,
        "selection_metric": "dev text-to-image r@5 then mrr",
        "label_source": "ai_image_caption"
        if annotations
        else "museum metadata-derived paired captions, not human relevance labels",
    }
    (output / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_file = (output / "history.jsonl").open("w", encoding="utf-8")
    start = time.perf_counter()
    best = (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep frozen text tower deterministic; its cached vectors are never updated.
        model.base_model.model.text_model.eval()
        order = np.random.permutation(len(train_items))
        losses = []
        batches = make_batches(order, args.batch_size, neighbors)
        for step, indices in enumerate(batches):
            if len(indices) < 2:
                continue
            images = []
            for index in indices:
                with Image.open(train_items[index]["image_path"]) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    # Avoid random crops/flips that can remove motifs or alter handedness.
                    images.append(image)
            inputs = encoder.image_inputs(images)
            text = train_text[indices].to(encoder.device)
            target = teacher[indices].to(encoder.device)
            batch_captions = [train_captions[i] for i in indices]
            positives = torch.tensor(
                [[a == b for b in batch_captions] for a in batch_captions], device=encoder.device
            )
            optimizer.zero_grad(set_to_none=True)
            with encoder.autocast():
                image_features = model.base_model.model.get_image_features(**inputs)
                contrastive = multi_positive_loss(image_features, text, positives)
                anchor = (1 - F.cosine_similarity(image_features.float(), target)).mean()
                loss = contrastive + args.anchor_weight * anchor
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % 20 == 0:
                progress = {
                    "event": "step",
                    "epoch": epoch,
                    "step": step,
                    "steps_per_epoch": steps_per_epoch,
                    "loss": float(loss.detach()),
                    "elapsed_seconds": round(time.perf_counter() - start, 2),
                }
                print(json.dumps(progress), flush=True)
        model.eval()
        dev_features = encoder.encode_images(dev_paths, args.batch_size)
        metrics = paired_retrieval(dev_features, dev_text, dev_captions)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "dev": metrics,
            "elapsed_seconds": time.perf_counter() - start,
            "peak_cuda_mb": torch.cuda.max_memory_allocated() / 2**20
            if encoder.device.type == "cuda"
            else 0,
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(json.dumps(record), flush=True)
        key = (metrics["r@5"], metrics["mrr"])
        if key > best:
            best = key
            model.save_pretrained(output / "best_adapter")
            metadata["best_epoch"] = epoch
            metadata["best_dev"] = metrics
            metadata["improved_over_baseline"] = key > (baseline["r@5"], baseline["mrr"])
            (output / "run.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    log_file.close()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--output", default="artifacts/training/siglip_lora")
    parser.add_argument("--encoder", choices=["clip", "siglip"], default="siglip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--sampling", choices=["random", "hard"], default="random")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--anchor-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=5523)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument(
        "--visual-annotations",
        default=None,
        help="Use only annotated train/dev image captions instead of museum captions",
    )
    print(json.dumps(train(parser.parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
