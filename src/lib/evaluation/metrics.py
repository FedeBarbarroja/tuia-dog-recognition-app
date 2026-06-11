"""Funciones auxiliares para evaluacion (provistas por la catedra).

Cubren las metricas pedidas en el TP:
  - Etapa 1: NDCG@10 (ndcg_at_k).
  - Etapa 2: precision / recall / F1 / specificity.
  - Etapa 4: IoU, matching de detecciones, AP / mAP.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def ndcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """NDCG@k de un ranking.

    `relevances` son las relevancias en el orden devuelto por la busqueda
    (ej: 1.0 si el vecino es de la misma raza que la consulta, 0.0 si no).
    """
    rel = [float(r) for r in list(relevances)[:k]]
    if not rel:
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection over Union entre dos cajas (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Retorna (precision, recall, f1) a partir de conteos TP/FP/FN."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def specificity(tn: int, fp: int) -> float:
    """Especificidad: TN / (TN + FP)."""
    return tn / (tn + fp) if (tn + fp) else 0.0


def match_detections(
    predictions: Sequence[dict],
    ground_truths: Sequence[dict],
    iou_threshold: float = 0.5,
    match_labels: bool = True,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Matching greedy entre predicciones y ground truth de una imagen.

    predictions: dicts {"bbox": [x1,y1,x2,y2], "label": str, "score": float}
    ground_truths: dicts {"bbox": [x1,y1,x2,y2], "label": str}

    Retorna (matches, false_positive_idx, false_negative_idx) donde
    matches = [(pred_idx, gt_idx, iou_value), ...].
    """
    order = sorted(
        range(len(predictions)),
        key=lambda i: predictions[i].get("score", 0.0),
        reverse=True,
    )
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    false_positives: list[int] = []

    for pred_idx in order:
        pred = predictions[pred_idx]
        best_iou, best_gt = 0.0, -1
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in used_gt:
                continue
            if match_labels and pred.get("label") != gt.get("label"):
                continue
            value = iou(pred["bbox"], gt["bbox"])
            if value > best_iou:
                best_iou, best_gt = value, gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt)
            matches.append((pred_idx, best_gt, best_iou))
        else:
            false_positives.append(pred_idx)

    false_negatives = [i for i in range(len(ground_truths)) if i not in used_gt]
    return matches, false_positives, false_negatives


def average_precision(scored_hits: Sequence[tuple[float, bool]], total_gt: int) -> float:
    """AP de una clase: area bajo la curva precision-recall.

    scored_hits: lista de (score, es_tp) por cada prediccion de la clase
    (en cualquier orden; se ordena por score descendente).
    total_gt: cantidad total de objetos ground truth de la clase.
    """
    if total_gt <= 0:
        return 0.0
    ordered = sorted(scored_hits, key=lambda item: item[0], reverse=True)
    tp = fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _, hit in ordered:
        if hit:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_gt)

    # Interpolacion: precision no creciente de derecha a izquierda.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def mean_average_precision(aps_per_class: Iterable[float]) -> float:
    """mAP: promedio de los AP por clase."""
    aps = list(aps_per_class)
    return sum(aps) / len(aps) if aps else 0.0
