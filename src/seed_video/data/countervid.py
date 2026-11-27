"""Read CounterVid free-form preferences without changing dataset splits."""
from pathlib import Path


def select_freeform(rows, root, split, preference_signal='textual'):
    if split not in ("train", "validation"):
        raise ValueError(f"Unsupported CounterVid split: {split}")
    if preference_signal not in ('textual', 'visual'):
        raise ValueError(f'Unsupported preference signal: {preference_signal}')
    split_root = (Path(root) / split).resolve()
    items = []
    seen = set()
    for row in rows:
        if row.get("task_type") != "A_freeform" or row.get("preference_signal") != preference_signal:
            continue
        sample_id = row.get("id")
        prompt = row.get("prompt")
        answer = row.get("chosen_answer") if preference_signal == 'textual' else row.get('answer')
        names = row.get("videos_file_names")
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate sample id: {sample_id!r}")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Missing prompt or answer: {sample_id}")
        expected_roles = ['context'] if preference_signal == 'textual' else ['chosen', 'rejected']
        if row.get('video_roles') != expected_roles or not names or len(names) != len(expected_roles):
            raise ValueError(f'Unexpected video roles for {sample_id}: {row.get("video_roles")}')
        video = (split_root / names[0]).resolve()
        if not video.is_relative_to(split_root) or not video.is_file():
            raise ValueError(f"Invalid or missing video for {sample_id}: {video}")
        seen.add(sample_id)
        items.append(dict(id=sample_id, question=prompt.replace("<video>", "").strip(),
                          answer=answer, video_path=str(video),
                          rejected_answer=row.get("rejected_answer"),
                          base_identifier=row.get("base_identifier"), split=split))
    return items


def load_freeform(root, split, preference_signal='textual'):
    import pyarrow.parquet as pq

    path = Path(root) / split / "metadata.parquet"
    return select_freeform(pq.read_table(path).to_pylist(), root, split, preference_signal)
