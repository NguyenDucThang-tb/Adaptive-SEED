"""Offline Video-MME v1 annotations, prompts, and strict choice parsing.

JSON loading uses only the standard library. Parquet support is imported lazily.
Video files are resolved by official YouTube ``videoID``, never numeric video_id.
"""
import json
from pathlib import Path
import re


LETTERS = "ABCD"
DURATIONS = {"short", "medium", "long"}


def _text(record, name):
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {name}: {value!r}")
    return value.strip()


def normalize_videomme_records(records):
    """Flatten official per-video JSON or flat Parquet rows and validate them."""
    if isinstance(records, dict):
        records = records.get("data", records.get("questions", records))
    if not isinstance(records, list) or not records:
        raise ValueError("Video-MME annotations must be a nonempty list")
    flat = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Video-MME records must be objects")
        if "questions" in record:
            questions = record["questions"]
            if not isinstance(questions, list) or not questions:
                raise ValueError("Empty or invalid nested questions")
            parent = {k: v for k, v in record.items() if k != "questions"}
            for question in questions:
                if not isinstance(question, dict):
                    raise ValueError("Nested question must be an object")
                if any(k in parent and parent[k] != v for k, v in question.items()):
                    raise ValueError("Nested question conflicts with video metadata")
                flat.append(dict(parent, **question))
        else:
            flat.append(dict(record))
    seen, by_id, by_video = set(), {}, {}
    for item in flat:
        for name in ("video_id", "videoID", "question_id", "duration", "question",
                     "domain", "sub_category", "task_type"):
            item[name] = _text(item, name)
        qid, vid, youtube = item["question_id"], item["video_id"], item["videoID"]
        if qid in seen:
            raise ValueError(f"Duplicate question_id: {qid}")
        seen.add(qid)
        if item["duration"] not in DURATIONS:
            raise ValueError(f"Invalid duration for {qid}: {item['duration']}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", youtube):
            raise ValueError(f"Invalid videoID: {youtube}")
        metadata = (youtube, item["duration"], item.get("domain"),
                    item.get("sub_category"), item.get("url"))
        if vid in by_id and by_id[vid] != metadata:
            raise ValueError(f"Conflicting video mapping/metadata: {vid}")
        if youtube in by_video and by_video[youtube] != vid:
            raise ValueError(f"Conflicting video mapping: {youtube}")
        by_id[vid], by_video[youtube] = metadata, vid
        options = item.get("options")
        if not isinstance(options, (list, tuple)) or len(options) != 4:
            raise ValueError(f"Expected four options for {qid}")
        bodies = []
        for letter, option in zip(LETTERS, options):
            if not isinstance(option, str):
                raise ValueError(f"Invalid option for {qid}")
            match = re.fullmatch(r"([A-D])\.\s*(\S(?:.*\S)?)", option.strip(), re.S)
            if not match or match[1] != letter:
                raise ValueError(f"Options must be ordered A. through D. for {qid}")
            bodies.append(match[2])
        item["options"] = list(options)
        if not isinstance(item.get("answer"), str) or len(item["answer"]) != 1 or item["answer"] not in LETTERS:
            raise ValueError(f"Invalid answer for {qid}: {item.get('answer')!r}")
    return flat


def read_videomme_annotations(path):
    """Read normalized/official JSON, a Parquet file, or a shard directory."""
    path = Path(path)
    if path.is_dir():
        paths = sorted(path.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No annotation Parquet files in {path}")
        rows = []
        for shard in paths:
            rows.extend(_parquet_rows(shard))
    elif path.suffix == ".parquet":
        rows = _parquet_rows(path)
    else:
        with path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
    return normalize_videomme_records(rows)


def _parquet_rows(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading Video-MME Parquet requires pyarrow; use questions.json offline") from exc
    return pq.read_table(path).to_pylist()


def load_videomme(root, require_videos=True):
    """Return validated item dicts; missing/ambiguous videos are fatal by default.

    Each item retains official fields and adds video (absolute path), subset,
    pair_id, role, and prompt. ``require_videos=False`` is annotation-only mode;
    it returns the expected path when no video exists, but still rejects ambiguity.
    """
    root = Path(root).resolve()
    annotations = root / "questions.json"
    if not annotations.exists():
        annotations = root / "videomme"
    rows = read_videomme_annotations(annotations)
    index = {}
    for path in sorted((root / "videos").rglob("*")):
        if path.is_file() and path.suffix.lower() == ".mp4":
            index.setdefault(path.stem, []).append(path)
    items = []
    for row in rows:
        candidates = index.get(row["videoID"], [])
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous videoID {row['videoID']}: {candidates}")
        if not candidates and require_videos:
            raise FileNotFoundError(f"Missing Video-MME video: {row['videoID']}.mp4 under {root / 'videos'}")
        path = candidates[0] if candidates else root / "videos" / f"{row['videoID']}.mp4"
        item = dict(row, video=str(path), subset="videomme", role="videomme",
                    pair_id=f"videomme/{row['question_id']}")
        item["prompt"] = build_prompt(item)
        items.append(item)
    return items


def build_prompt(item):
    """Build the no-subtitle A/B/C/D video question."""
    return ("Select the best answer to the following multiple-choice question based on the video.\n"
            + item["question"] + "\n" + "\n".join(item["options"])
            + "\nRespond with only the letter (A, B, C, or D) of the correct option.\nThe best answer is:")


def parse_answer(text):
    """Strict anchored extraction, not the official broad regex: A, A., or (A).

    Prose and multiple choices are invalid and must count as incorrect.
    """
    if not isinstance(text, str):
        return ""
    match = re.fullmatch(r"\s*(?:([A-D])\.?|\(([A-D])\))\s*", text)
    return (match[1] or match[2]) if match else ""


build_videomme_prompt = build_prompt
parse_videomme_response = parse_answer
