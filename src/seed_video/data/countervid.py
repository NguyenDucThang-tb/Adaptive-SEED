"""Read CounterVid free-form textual preferences without changing dataset splits."""
from pathlib import Path


def select_freeform(rows, root, split):
    if split not in ("train", "validation"):
        raise ValueError(f"Unsupported CounterVid split: {split}")
    split_root = (Path(root) / split).resolve()
    items = []
    seen = set()
    for row in rows:
        if row.get("task_type") != "A_freeform" or row.get("preference_signal") != "textual":
            continue
        sample_id = row.get("id")
        prompt = row.get("prompt")
        answer = row.get("chosen_answer")
        names = row.get("videos_file_names")
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate sample id: {sample_id!r}")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Missing prompt or answer: {sample_id}")
        if row.get("video_roles") != ["context"] or not names or len(names) != 1:
            raise ValueError(f"Expected one context video: {sample_id}")
        video = (split_root / names[0]).resolve()
        if not video.is_relative_to(split_root) or not video.is_file():
            raise ValueError(f"Invalid or missing video for {sample_id}: {video}")
        seen.add(sample_id)
        items.append(dict(id=sample_id, question=prompt.replace("<video>", "").strip(),
                          answer=answer, video_path=str(video),
                          rejected_answer=row.get("rejected_answer"),
                          base_identifier=row.get("base_identifier"), split=split))
    return items


def load_freeform(root, split):
    import pyarrow.parquet as pq

    path = Path(root) / split / "metadata.parquet"
    return select_freeform(pq.read_table(path).to_pylist(), root, split)
