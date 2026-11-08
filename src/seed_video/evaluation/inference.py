def eval_items(runtime, items):
    runtime.model.eval()
    with runtime.torch.no_grad():
        return {runtime.key(i): runtime.margin(runtime.student_last(i, grad=False)) for i in items}

def eval_set(runtime):
    """Items to score, each key once (EventHallusion training sets EH = TEST)."""
    items = ((runtime.TRAIN if runtime.CFG['eval_train'] else [])
             + runtime.VAL + runtime.TEST + runtime.EH
             + getattr(runtime, 'EXTRA_EVAL', []))
    seen, out = set(), []
    for item in items:
        k = runtime.key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out
