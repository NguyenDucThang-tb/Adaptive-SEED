def eval_items(runtime, items):
    runtime.model.eval()
    with runtime.torch.no_grad():
        return {runtime.key(i): runtime.margin(runtime.student_last(i, grad=False)) for i in items}

def eval_set(runtime):
    return (runtime.TRAIN if runtime.CFG['eval_train'] else []) + runtime.VAL + runtime.TEST + runtime.EH
