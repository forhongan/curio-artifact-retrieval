"""Ranking metrics: unjudged=0, relevance threshold=2, explicit denominators."""

import math


def dcg(grades, k=10):
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))


def ndcg(ranked_ids, qrels, k=10):
    ideal = dcg(sorted(qrels.values(), reverse=True), k)
    return dcg([qrels.get(i, 0) for i in ranked_ids], k) / ideal if ideal else 0.0


def precision(ranked_ids, qrels, k=5, threshold=2):
    return sum(qrels.get(i, 0) >= threshold for i in ranked_ids[:k]) / k


def recall(ranked_ids, qrels, k=20, threshold=2):
    total = sum(g >= threshold for g in qrels.values())
    return sum(qrels.get(i, 0) >= threshold for i in ranked_ids[:k]) / total if total else 0.0


def average_precision(ranked_ids, qrels, k=10, threshold=2):
    """Truncated AP with full relevant-set denominator (not min(total, k))."""
    total = sum(g >= threshold for g in qrels.values())
    if not total:
        return 0.0
    found, score = 0, 0.0
    for rank, item_id in enumerate(ranked_ids[:k], 1):
        if qrels.get(item_id, 0) >= threshold:
            found += 1
            score += found / rank
    return score / total


def reciprocal_rank(ranked_ids, qrels, k=10, threshold=2):
    return next(
        (1 / rank for rank, i in enumerate(ranked_ids[:k], 1) if qrels.get(i, 0) >= threshold), 0.0
    )


def evaluate_ranking(ranked_ids, qrels):
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Ranking contains duplicate item IDs")
    return {
        "ndcg@10": ndcg(ranked_ids, qrels),
        "ap@10": average_precision(ranked_ids, qrels),
        "precision@5": precision(ranked_ids, qrels),
        "recall@20": recall(ranked_ids, qrels),
        "mrr@10": reciprocal_rank(ranked_ids, qrels),
    }
