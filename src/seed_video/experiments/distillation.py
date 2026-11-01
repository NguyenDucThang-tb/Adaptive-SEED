"""Compare answer-level KD with a Video-OPD-inspired on-policy adaptation."""
from types import MethodType

from seed_video.evaluation.reporting import result_rows


METHODS = ("kd_basic", "video_opd")
TEACHER = "T_pos_tcd"


def method_registry(runtime, methods, teacher=TEACHER):
    registry = {}
    for method in methods:
        if method == "kd_basic":
            step = lambda T=teacher: runtime.step_kd_basic(T)
            selector = None
        elif method == "video_opd":
            step = lambda T=teacher: runtime.step_onpolicy(T, "rkl")
            selector = lambda T=teacher: runtime.tvdf_select(T)
        else:
            raise ValueError(f"Unsupported distillation method: {method}")
        registry[f"{method}@{teacher}"] = (step, teacher, selector, False)
    return registry


def _save_results(runtime):
    path = runtime.RES / "results.csv"
    new = runtime.pd.DataFrame(runtime.ROWS)
    if path.exists():
        old = runtime.pd.read_csv(path)
        new = runtime.pd.concat(
            [old[~old["run"].isin(set(new["run"]))], new],
            ignore_index=True,
        )
    temp = path.with_name(f"{path.name}.tmp.{runtime.os.getpid()}")
    new.to_csv(temp, index=False)
    runtime.os.replace(temp, path)


def run(runtime, methods):
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unsupported distillation methods: {unknown}")

    runtime.L = {
        branch: {
            key: row[branch]
            for key, row in runtime.SHARED.items()
        }
        for branch in runtime.TEACHER_BRANCHES[TEACHER]
    }

    runtime.METHODS.update(method_registry(runtime, methods))

    tag = f"ap{runtime.CFG['a_pos']:g}_an{runtime.CFG['a_tcd']:g}_tf{runtime.CFG['teacher_frames']}"
    runtime.CFG["epochs"] = 2
    runtime.OUT = runtime.Path(runtime.CACHE) / "distillation_methods"
    runtime.RES = runtime.OUT / "runs" / tag
    runtime.STUDENT_DIR = str(runtime.RES / "students")
    runtime.RES.mkdir(parents=True, exist_ok=True)
    runtime.os.makedirs(runtime.STUDENT_DIR, exist_ok=True)
    runtime.SPLITS = {"VH_VAL": runtime.VAL, "VH_TEST": runtime.TEST, "EH_ALL": runtime.EH}
    for subset in sorted({item["subset"] for item in runtime.EH}):
        runtime.SPLITS[subset] = [item for item in runtime.EH if item["subset"] == subset]
    runtime.result_rows = MethodType(result_rows, runtime)
    runtime.save_distillation_results = MethodType(_save_results, runtime)
    runtime.MARG = {}

    base_meta = {
        "source": runtime.SOURCE_HASH,
        "branch": runtime.branch_cfg("orig"),
        "items": [runtime.item_sig(item) for item in runtime.eval_set()],
    }
    base_path = runtime.RES / f"base_{runtime.cfg_hash(base_meta)}.pt"
    if base_path.exists():
        runtime.MARG["base"] = runtime.torch.load(base_path, weights_only=False)
    else:
        runtime.MARG["base"] = runtime.eval_items(runtime.eval_set())
        runtime.torch.save(runtime.MARG["base"], base_path)

    runtime.ROWS = runtime.result_rows("base", runtime.MARG["base"])
    runtime.save_distillation_results()
    print(
        f"Distillation comparison: teacher={TEACHER}, methods={list(methods)}, "
        f"epochs={runtime.CFG['epochs']}"
    )
    print(f"Run tag: {tag} | results: {runtime.RES}")

    for method in methods:
        name = f"{method}@{TEACHER}"
        for seed in runtime.CFG["seeds"]:
            output = runtime.train_arm(name, seed)
            run_name = f"{name}_s{seed}"
            runtime.MARG[run_name] = output["marg"]
            rows = runtime.result_rows(run_name, output["marg"])
            runtime.ROWS.extend(rows)
            runtime.display(runtime.pd.DataFrame(rows))
            runtime.save_distillation_results()
            runtime.pd.DataFrame([
                {
                    "id": runtime.key(item),
                    "video": item["video"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "base_margin": runtime.MARG["base"][runtime.key(item)],
                    "student_margin": output["marg"][runtime.key(item)],
                }
                for item in runtime.eval_set()
            ]).to_csv(runtime.RES / f"{run_name}_predictions.csv", index=False)
            runtime.gc.collect()
            runtime.torch.cuda.empty_cache()

    runtime.display(runtime.pd.DataFrame(runtime.ROWS))
    print("Saved distillation method results and adapters:", runtime.OUT)
