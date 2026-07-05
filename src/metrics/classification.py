import torch

__all__ = ["binary_classification_metrics"]

_EPS = 1e-12


def _avg_tied_ranks(scores: torch.Tensor) -> torch.Tensor:
    """1-indexed ranks of ``scores`` (ascending), ties averaged."""
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    ends = torch.cumsum(counts, 0)            # 1-indexed group end ranks
    starts = ends - counts                    # 0-indexed group starts
    group_avg = (starts + 1 + ends).to(scores.dtype) / 2.0
    ranks = torch.empty_like(scores)
    ranks[order] = group_avg[inverse]
    return ranks


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC via the Mann-Whitney U statistic (tie-aware)."""
    n_pos = labels.sum()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_tied_ranks(scores)
    sum_pos = ranks[labels == 1].sum()
    return ((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)).item()


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the precision-recall curve, sum_n (R_n - R_{n-1}) * P_n over
    distinct score thresholds (tied scores collapse to one point, as sklearn)."""
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    s, lab = scores[order], labels[order]
    tp_cum = torch.cumsum(lab, 0)
    fp_cum = torch.cumsum(1.0 - lab, 0)

    # Keep the last index of each tied-score group (a threshold boundary).
    distinct = torch.ones_like(s, dtype=torch.bool)
    distinct[:-1] = s[1:] != s[:-1]
    tp_t, fp_t = tp_cum[distinct], fp_cum[distinct]

    precision = tp_t / (tp_t + fp_t)
    recall = tp_t / n_pos
    recall_prev = torch.cat([recall.new_zeros(1), recall[:-1]])
    return ((recall - recall_prev) * precision).sum().item()


def _best_f1_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Logit-scale threshold maximizing F1 (predicting ``scores >= t``), swept
    over distinct scores. Informational diagnostic — it does NOT re-threshold
    the reported metrics. Returns 0.0 when only one class is present."""
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == labels.numel():
        return 0.0
    order = torch.argsort(scores, descending=True)
    s, lab = scores[order], labels[order]
    tp_cum = torch.cumsum(lab, 0)
    fp_cum = torch.cumsum(1.0 - lab, 0)
    distinct = torch.ones_like(s, dtype=torch.bool)
    distinct[:-1] = s[1:] != s[:-1]
    tp_t, fp_t = tp_cum[distinct], fp_cum[distinct]
    precision = tp_t / (tp_t + fp_t + _EPS)
    recall = tp_t / n_pos
    f1 = 2.0 * precision * recall / (precision + recall + _EPS)
    return s[distinct][torch.argmax(f1)].item()


def binary_classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.0
) -> dict[str, float]:
    """Binary link-prediction metrics from raw logits + {0,1} labels.

    Runs on the tensors' device (no host transfer). ``threshold`` is on the
    logit scale (0.0 == probability 0.5); ``accuracy``/``precision``/``recall``/
    ``f1``/``mcc`` are computed at it. ``roc_auc`` and ``ap`` return nan when
    only one class is present. ``best_threshold`` is the F1-maximizing logit
    threshold (a diagnostic — it does not re-threshold the other metrics).
    """
    labels = labels.float()
    preds = (logits > threshold).float()

    tp = (preds * labels).sum()
    fp = (preds * (1.0 - labels)).sum()
    fn = ((1.0 - preds) * labels).sum()
    tn = ((1.0 - preds) * (1.0 - labels)).sum()

    precision = tp / (tp + fp + _EPS)
    recall = tp / (tp + fn + _EPS)
    mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / mcc_den).item() if mcc_den.item() > 0 else 0.0
    return {
        "accuracy": ((tp + tn) / (tp + fp + fn + tn + _EPS)).item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": (2.0 * precision * recall / (precision + recall + _EPS)).item(),
        "mcc": mcc,
        "roc_auc": _roc_auc(logits, labels),
        "ap": _average_precision(logits, labels),
        "best_threshold": _best_f1_threshold(logits, labels),
    }
