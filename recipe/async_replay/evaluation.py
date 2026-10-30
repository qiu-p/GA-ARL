"""Sample-average correctness metrics for repeated evaluation prompts."""
from collections import defaultdict


def average_accuracy(sample_uids, sample_scores, sample_counts, data_sources):
    if not (len(sample_uids) == len(sample_scores) == len(data_sources)):
        raise ValueError("Evaluation arrays must have equal lengths")
    groups = defaultdict(list)
    for uid, score, source in zip(sample_uids, sample_scores, data_sources):
        groups[(source, uid)].append(float(score) > 0)
    metrics = {}
    for source in {source for source, _ in groups}:
        responses = [scores for (ds, _), scores in groups.items() if ds == source]
        for count in sample_counts:
            if count < 1 or any(len(scores) < count for scores in responses):
                raise ValueError(f"Insufficient responses for avg@{count}")
            accuracy = sum(sum(scores[:count]) / count for scores in responses) / len(responses)
            metrics[f"val-core/{source}/acc/avg@{count}"] = accuracy
    return metrics
