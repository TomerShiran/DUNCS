from typing import Any, Dict

def _to_float(x: Any) -> float:
    # Works for Python numbers, numpy scalars, and 0-d torch tensors
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.generic):  # numpy scalar
            return float(x.item())
    except Exception:
        pass
    return float(x)

def normalize_result(result: Any) -> Dict[str, float | None]:
    """
    Normalize model outputs into a consistent dict:
      {'loss': float, 'accuracy': float|None}

    Accepts:
      - scalar (loss only)
      - (loss, accuracy) tuple/list
      - dict with keys like 'loss' and optionally 'accuracy'/'acc'/'top1'
    """
    # dict case
    if isinstance(result, dict):
        loss = None
        if 'loss' in result:
            loss = _to_float(result['loss'])
        if loss is None:
            raise ValueError("Result dict must contain 'loss'.")
        # common aliases for accuracy
        acc_key = next((k for k in ('accuracy','acc','top1') if k in result), None)
        acc = 0.0 if acc_key is None else _to_float(result[acc_key])
        return {'loss': loss, 'Accuracy': acc}

    # tuple/list case
    if isinstance(result, (tuple, list)):
        if not result:
            raise ValueError("Empty result.")
        if len(result) == 1 or (len(result) == 2 and result[1] is None):
            return {'loss': _to_float(result[0]), 'Accuracy': 0.0}
        return {'loss': _to_float(result[0]), 'Accuracy': _to_float(result[1])}

    # scalar case
    return {'loss': _to_float(result), 'Accuracy': 0}