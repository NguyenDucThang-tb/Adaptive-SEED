"""Prepare pinned Video-MME v1 on copyq; no GPU/runtime downloads.

Keep extracted MP4 files on GData. Each independent ZIP is downloaded to transient
JOBFS, streamed into videos/, recorded in a manifest, and removed before the next.
Rerun the same command after a copyq walltime limit to resume completed archives.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seed_video.data.videomme import (  # noqa: E402
    load_videomme, normalize_videomme_records, read_videomme_annotations,
)

REPO_ID = "lmms-eval/Video-MME"
ARCHIVES = [f"videos_chunked_{i:02d}.zip" for i in range(1, 21)]
MANIFEST = "download_manifest.json"
CHUNK_SIZE = 1024 * 1024


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".videomme-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def destination_lock(root):
    import fcntl
    with (root / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another downloader is using this destination") from exc
        yield


def file_record(path):
    crc, size = 0, 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
    return {"size": size, "crc32": crc & 0xFFFFFFFF}


def checked_video_root(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"Refusing symlink video directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def archive_complete(video_root, records):
    """Verify bytes against the ZIP's saved size/CRC, not merely marker existence."""
    if not records:
        return False
    root = checked_video_root(video_root)
    for name, expected in records.items():
        if PurePosixPath(name).name != name or not name.endswith(".mp4"):
            raise ValueError(f"Invalid manifest video name: {name}")
        path = root / name
        if path.is_symlink():
            raise ValueError(f"Refusing symlink video: {path}")
        if not path.is_file() or file_record(path) != expected:
            return False
    return True


def extract_archive(archive, video_root):
    """Preflight every member, then stream MP4s; never overwrite conflicting data.

    Flatten safe source paths to official videoID.mp4 names. This also detects
    duplicate basenames across source directories before any file is written.
    ZIP reads verify CRCs; partial writes use temporary files and atomic links.
    """
    root = checked_video_root(video_root)
    records, members = {}, []
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            name = info.filename
            path = PurePosixPath(name)
            mode = info.external_attr >> 16
            if (not name or "\\" in name or "\x00" in name or path.is_absolute()
                    or ".." in path.parts or re.match(r"^[A-Za-z]:", name)
                    or stat.S_ISLNK(mode) or info.flag_bits & 1):
                raise ValueError(f"Unsafe ZIP member: {name!r}")
            kind = stat.S_IFMT(mode)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Nonregular ZIP member: {name!r}")
            if info.is_dir():
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+\.mp4", path.name):
                raise ValueError(f"Unexpected ZIP member: {name!r}")
            if info.file_size <= 0 or path.name in records:
                raise ValueError(f"Empty or duplicate ZIP video: {name!r}")
            records[path.name] = {"size": info.file_size, "crc32": info.CRC}
            members.append(info)
        if not members:
            raise ValueError("Archive contains no MP4 videos")
        for info in members:
            target = root / PurePosixPath(info.filename).name
            if target.is_symlink():
                raise ValueError(f"Refusing symlink target: {target}")
            if target.exists():
                if not target.is_file() or file_record(target) != records[target.name]:
                    raise ValueError(f"Conflicting existing video: {target}")
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, prefix=".videomme-", delete=False) as output:
                    temporary = Path(output.name)
                    with source.open(info) as stream:
                        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if file_record(temporary) != records[target.name]:
                    raise ValueError(f"ZIP checksum mismatch: {info.filename}")
                os.link(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return records


def download_dataset(destination, jobfs=None, revision=None, annotations_only=False,
                     expected_questions=2700, expected_videos=900, api=None, download=None):
    """Download and validate a commit-pinned dataset; injected HF clients aid tests."""
    if api is None or download is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = api if api is not None else HfApi()
        download = download if download is not None else hf_hub_download
    root = Path(destination).absolute()
    if root.is_symlink():
        raise ValueError("Destination must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    with destination_lock(root):
        marker = root / MANIFEST
        manifest = json.loads(marker.read_text()) if marker.exists() else None
        if manifest:
            if manifest.get("repo_id") != REPO_ID or manifest.get("schema_version") != 1:
                raise ValueError("Incompatible dataset provenance; use a new destination")
            pinned = manifest["revision"]
            if revision is not None:
                requested = api.dataset_info(REPO_ID, revision=revision).sha
                if requested != pinned:
                    raise ValueError("Revision mismatch; use a new destination")
        else:
            if any(p.name != ".download.lock" for p in root.iterdir()):
                raise ValueError("Refusing nonempty destination without provenance manifest")
            pinned = api.dataset_info(REPO_ID, revision=revision or "main").sha
        if not isinstance(pinned, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", pinned):
            raise ValueError(f"Expected immutable HF commit hash, got {pinned!r}")
        files = sorted(api.list_repo_files(REPO_ID, repo_type="dataset", revision=pinned))
        annotations = [name for name in files if name.startswith("videomme/") and name.endswith(".parquet")]
        archives = [name for name in files if re.fullmatch(r"videos_chunked_\d{2}\.zip", name)]
        if not annotations or (not annotations_only and archives != ARCHIVES):
            raise ValueError("Official annotation shards or 20 independent ZIP archives missing")
        manifest = manifest or {"schema_version": 1, "repo_id": REPO_ID, "revision": pinned,
                                "created_utc": datetime.now(timezone.utc).isoformat(), "archives": {}}
        manifest.update(annotation_files=annotations, archive_files=archives,
                        source_url=f"https://huggingface.co/datasets/{REPO_ID}/tree/{pinned}")
        atomic_json(marker, manifest)
        rows = []
        for name in annotations:
            local = download(repo_id=REPO_ID, filename=name, repo_type="dataset",
                             revision=pinned, local_dir=str(root / "annotations"))
            rows.extend(read_videomme_annotations(local))
        rows = normalize_videomme_records(rows)
        n_videos = len({row["videoID"] for row in rows})
        if len(rows) != expected_questions or n_videos != expected_videos:
            raise ValueError(f"Expected {expected_questions} questions/{expected_videos} videos; got {len(rows)}/{n_videos}")
        atomic_json(root / "questions.json", rows)
        manifest.update(question_count=len(rows), video_count=n_videos, complete=False)
        atomic_json(marker, manifest)
        if annotations_only:
            print(f"Annotations ready: {len(rows)} questions / {n_videos} videos; revision={pinned}", flush=True)
            return manifest
        if jobfs is None:
            raise ValueError("Full download requires --jobfs or PBS_JOBFS for transient archives")
        scratch = Path(jobfs).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        if scratch == root.resolve() or root.resolve() in scratch.parents or scratch in root.resolve().parents:
            raise ValueError("Archive JOBFS must be separate from permanent destination")
        video_root = checked_video_root(root / "videos")
        used = set()
        for name in archives:
            completed = manifest["archives"].get(name)
            if completed and used.intersection(completed):
                raise ValueError(f"Video appears in multiple archives: {name}")
            if completed and archive_complete(video_root, completed):
                print(f"Verified completed {name}", flush=True)
            else:
                print(f"Downloading/extracting {name} at {pinned}", flush=True)
                with tempfile.TemporaryDirectory(prefix="videomme-", dir=scratch) as transient:
                    local = download(repo_id=REPO_ID, filename=name, repo_type="dataset",
                                     revision=pinned, local_dir=transient)
                    completed = extract_archive(local, video_root)
                    if used.intersection(completed):
                        raise ValueError(f"Video appears in multiple archives: {name}")
                    manifest["archives"][name] = completed
                    atomic_json(marker, manifest)
            used.update(completed)
        items = load_videomme(root)
        actual = list(video_root.rglob("*.mp4"))
        expected_names = {f"{row['videoID']}.mp4" for row in rows}
        if len(actual) != expected_videos or used != expected_names:
            raise ValueError(f"Expected exactly {expected_videos} referenced MP4s; found {len(actual)}")
        manifest.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(marker, manifest)
        print(f"Ready: {len(items)} questions / {len(actual)} MP4s; revision={pinned}", flush=True)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--revision", help="HF ref/commit; first run resolves and persists an immutable commit")
    parser.add_argument("--annotations-only", "--annotation-only", action="store_true")
    parser.add_argument("--jobfs", type=Path, default=os.environ.get("PBS_JOBFS"))
    args = parser.parse_args()
    download_dataset(args.destination, jobfs=args.jobfs, revision=args.revision,
                     annotations_only=args.annotations_only)


if __name__ == "__main__":
    main()
