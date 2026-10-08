"""Paired backend fidelity metrics. No training and no hardware performance claims."""
import numpy as np


def integer_scores(logits):
    """Exact align_corners=True 4x4->16x16 numerators; denominator is 25.

    Ascending class order + first argmax implements the RTL strict-greater tie rule.
    """
    q = np.asarray(logits)
    if q.ndim != 3 or q.shape[1:] != (4, 4) or not 2 <= q.shape[0] <= 256:
        raise ValueError('Expected [classes,4,4], 2<=classes<=256')
    if not np.issubdtype(q.dtype, np.signedinteger) or np.any((q < -128) | (q > 127)):
        raise ValueError('Expected signed INT8 codes without overflow')
    q = q.astype(np.int64)
    b = np.minimum(np.arange(16) // 5, 2)
    f = np.arange(16) - 5 * b
    horizontal = q[:, :, b] * (5-f) + q[:, :, b+1] * f
    return horizontal[:, b, :] * (5-f)[None, :, None] + horizontal[:, b+1, :] * f[None, :, None]


def numeric(reference, actual):
    r, a = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if r.shape != a.shape or not np.isfinite(r).all() or not np.isfinite(a).all():
        raise ValueError('Invalid paired arrays')
    e = a-r
    squared, ref2 = float(np.square(e).sum()), float(np.square(r).sum())
    return dict(count=int(e.size), mismatches=int(np.count_nonzero(e)),
                MAE=float(np.abs(e).mean()) if e.size else None,
                RMSE=float(np.sqrt(squared/e.size)) if e.size else None,
                max_abs_error=float(np.abs(e).max()) if e.size else None,
                relative_L2=float(np.sqrt(squared/ref2)) if ref2 else None)


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return dict(count=0, mean=None, median=None, p90=None, maximum=None, zero_count=0)
    return dict(count=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), maximum=float(a.max()), zero_count=int((a == 0).sum()))


def paired_labels(reference, actual, truth, indices, classes, margin):
    """Truth is 1-based, 0=unlabeled. Predictions are 0-based.

    Disagreement uses every selected pixel. Correct/wrong exchange uses labeled
    selected pixels only. This distinction is enforced even for scene scopes.
    """
    ix = np.asarray(indices, dtype=np.int64)
    r, a, g = (np.asarray(x).reshape(-1)[ix] for x in (reference, actual, truth))
    if np.any((r < 0) | (r >= classes) | (a < 0) | (a >= classes) | (g < 0) | (g > classes)):
        raise ValueError('Label out of range')
    changed = r != a
    valid = g > 0
    rc, ac = r == g-1, a == g-1
    # Cast BEFORE multiplication: UINT8 class IDs overflow for 16/22 classes.
    transitions = np.bincount(r.astype(np.int64)*classes+a.astype(np.int64),
                             minlength=classes**2).reshape(classes, classes)
    rows = []
    for label in range(classes):
        mask = g == label+1
        rows.append(dict(class_id=label, ground_truth_id=label+1, count=int(mask.sum()),
                         disagreements=int((mask & changed).sum()),
                         reference_correct_actual_wrong=int((mask & rc & ~ac).sum()),
                         reference_wrong_actual_correct=int((mask & ~rc & ac).sum()),
                         both_wrong_changed=int((mask & ~rc & ~ac & changed).sum())))
    n = int(valid.sum())
    return dict(count=int(ix.size), mismatches=int(changed.sum()),
                mismatch_fraction=float(changed.mean()) if ix.size else None,
                labeled_count=n,
                reference_OA=float(rc[valid].mean()) if n else None,
                actual_OA=float(ac[valid].mean()) if n else None,
                reference_correct_actual_wrong=int((valid & rc & ~ac).sum()),
                reference_wrong_actual_correct=int((valid & ~rc & ac).sum()),
                both_wrong_changed=int((valid & ~rc & ~ac & changed).sum()),
                prediction_transition_matrix=transitions.tolist(), per_class=rows,
                reference_margin_all=distribution(np.asarray(margin).reshape(-1)[ix]),
                reference_margin_disagreements=distribution(np.asarray(margin).reshape(-1)[ix][changed]))


def scene_from_tiles(tiles, placements, shape, scale):
    prediction = np.full(shape, -1, dtype=np.int16)
    margins = np.zeros(shape, dtype=np.float64)
    coverage = np.zeros(shape, dtype=np.uint8)
    if len(tiles) != len(placements) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid tile count or common head scale')
    for q, p in zip(tiles, placements):
        y, x, h, w = (int(p[k]) for k in ('top', 'left', 'valid_height', 'valid_width'))
        if not (0 <= y < shape[0] and 0 <= x < shape[1] and 1 <= h <= 16 and 1 <= w <= 16
                and y+h <= shape[0] and x+w <= shape[1]):
            raise ValueError('Invalid tile placement')
        region = np.s_[y:y+h, x:x+w]
        if coverage[region].any():
            raise ValueError('Overlapping tile placement')
        scores = integer_scores(q)
        top = np.partition(scores, -2, axis=0)[-2:]
        prediction[region] = scores.argmax(axis=0)[:h, :w]
        margins[region] = (top[-1]-top[-2])[:h, :w] * scale/25.
        coverage[region] = 1
    if not np.all(coverage == 1):
        raise ValueError('Incomplete scene coverage')
    return prediction.astype(np.uint8), margins
