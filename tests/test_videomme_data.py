"""Offline data and download safety tests; no network or actual videos needed."""
import copy
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from seed_video.data.videomme import (
    build_prompt, load_videomme, normalize_videomme_records, parse_answer,
    read_videomme_annotations,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_videomme.py"
SPEC = importlib.util.spec_from_file_location("videomme_download_test", SCRIPT)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def record(**changes):
    row = dict(video_id="001", videoID="abc_-123", question_id="001-1", duration="short",
               domain="Knowledge", sub_category="Art", url="https://youtube.com/watch?v=abc_-123",
               task_type="Recognition", question="What is shown?",
               options=["A. One", "B. Two", "C. Three", "D. Four"], answer="A")
    row.update(changes)
    return row


class VideoMMELoaderTests(unittest.TestCase):
    def test_official_duplicate_option_bodies_are_preserved(self):
        row = record(question_id="119-1", options=["A. Same text", "B. Same text", "C. Other", "D. Last"])
        self.assertEqual(normalize_videomme_records([row]), [row])

    def test_flat_and_nested_are_equivalent(self):
        flat = record()
        parent = {k: v for k, v in flat.items() if k not in ("question_id", "question", "task_type", "options", "answer")}
        question = {k: v for k, v in flat.items() if k not in parent}
        self.assertEqual(normalize_videomme_records([dict(parent, questions=[question])]), [flat])
        self.assertEqual(normalize_videomme_records({"data": [flat]}), [flat])

    def test_invalid_annotations_raise_value_error(self):
        cases = [dict(answer=None), dict(answer=""), dict(answer=1), dict(answer="AA"),
                 dict(domain=None), dict(sub_category=""), dict(task_type=None),
                 dict(answer="E"), dict(duration="very long"), dict(videoID="../bad"),
                 dict(question_id=""), dict(options=["A. a"]),
                 dict(options=["A. a", "A. b", "C. c", "D. d"]),
                 dict(options=["A. a", "B. b", "C. c", "D. "])]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_videomme_records([record(**changes)])

    def test_duplicate_ids_and_conflicting_video_metadata(self):
        for other in [record(), record(question_id="001-2", videoID="other"),
                      record(question_id="001-2", duration="long"),
                      record(question_id="001-2", video_id="002")]:
            with self.subTest(other=other), self.assertRaises(ValueError):
                normalize_videomme_records([record(), other])

    def test_missing_and_ambiguous_video_fail_without_skipping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "questions.json").write_text(json.dumps([record()]))
            with self.assertRaises(FileNotFoundError):
                load_videomme(root)
            self.assertEqual(len(load_videomme(root, require_videos=False)), 1)
            for directory in ("one", "two"):
                target = root / "videos" / directory / "abc_-123.mp4"
                target.parent.mkdir(parents=True)
                target.write_bytes(b"video")
                if directory == "one":
                    item = load_videomme(root)[0]
                    self.assertEqual(item["video"], str(target))
                    self.assertEqual(item["answer"], "A")
                    self.assertEqual(item["pair_id"], "videomme/001-1")
            for require in (True, False):
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    load_videomme(root, require_videos=require)

    def test_numeric_video_id_is_not_a_video_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "questions.json").write_text(json.dumps([record()]))
            (root / "videos").mkdir()
            (root / "videos" / "001.mp4").write_bytes(b"wrong video")
            with self.assertRaises(FileNotFoundError):
                load_videomme(root)

    def test_json_has_no_parquet_dependency(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"pyarrow": None}):
            path = Path(temp) / "questions.json"
            path.write_text(json.dumps([record()]))
            self.assertEqual(read_videomme_annotations(path), [record()])

    def test_parquet_import_is_lazy_and_clear_when_unavailable(self):
        with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
            with self.assertRaisesRegex(ImportError, "requires pyarrow"):
                read_videomme_annotations("questions.parquet")

    def test_prompt_and_conservative_parser(self):
        prompt = build_prompt(record())
        self.assertIn("Select the best answer", prompt)
        self.assertTrue(prompt.endswith("The best answer is:"))
        self.assertNotIn("subtitle", prompt.lower())
        for letter in "ABCD":
            for text in (letter, f" {letter}.\n", f"({letter})"):
                self.assertEqual(parse_answer(text), letter)
        for text in ("", None, 1, "a", "Answer: A", "I choose B.", "A/B", "A or B", "(A).", "ABC"):
            with self.subTest(text=text):
                self.assertEqual(parse_answer(text), "")


class VideoMMEZipTests(unittest.TestCase):
    def make_zip(self, path, entries):
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in entries:
                archive.writestr(name, content)

    def test_stream_extract_and_resume_checks_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "videos.zip"
            self.make_zip(archive, [("videos/abc_-123.mp4", b"test video bytes")])
            videos = root / "videos"
            records = downloader.extract_archive(archive, videos)
            self.assertTrue(downloader.archive_complete(videos, records))
            target = videos / "abc_-123.mp4"
            before = target.stat().st_mtime_ns
            self.assertEqual(downloader.extract_archive(archive, videos), records)
            self.assertEqual(target.stat().st_mtime_ns, before)
            target.write_bytes(b"corrupt contents")
            self.assertFalse(downloader.archive_complete(videos, records))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                downloader.extract_archive(archive, videos)
            self.assertEqual(target.read_bytes(), b"corrupt contents")

    def test_missing_completed_video_is_repaired(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "archive.zip"
            videos = Path(temp) / "videos"
            self.make_zip(archive, [("abc_-123.mp4", b"video")])
            records = downloader.extract_archive(archive, videos)
            (videos / "abc_-123.mp4").unlink()
            self.assertFalse(downloader.archive_complete(videos, records))
            downloader.extract_archive(archive, videos)
            self.assertTrue(downloader.archive_complete(videos, records))

    def test_unsafe_paths_rejected_before_any_extraction(self):
        for unsafe in ("../outside.mp4", "/absolute.mp4", "videos/../../evil.mp4",
                       "videos\\evil.mp4", "C:/evil.mp4"):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = root / "archive.zip"
                self.make_zip(archive, [("safe.mp4", b"safe"), (unsafe, b"bad")])
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    downloader.extract_archive(archive, root / "videos")
                self.assertFalse((root / "videos" / "safe.mp4").exists())

    def test_symlink_zip_member_and_destination_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "archive.zip"
            info = zipfile.ZipInfo("evil.mp4")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            self.make_zip(archive, [(info, b"../outside")])
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                downloader.extract_archive(archive, root / "videos")
            self.make_zip(archive, [("evil.mp4", b"video")])
            outside = root / "outside"
            outside.write_bytes(b"original")
            (root / "videos" / "evil.mp4").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                downloader.extract_archive(archive, root / "videos")
            self.assertEqual(outside.read_bytes(), b"original")

    def test_duplicate_basename_empty_and_nonvideo_rejected(self):
        for entries in ([('a/x.mp4', b'a'), ('b/x.mp4', b'b')],
                        [('x.mp4', b'')], [('subtitle.txt', b'no subtitles')], []):
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as temp:
                archive = Path(temp) / "archive.zip"
                self.make_zip(archive, entries)
                with self.assertRaises(ValueError):
                    downloader.extract_archive(archive, Path(temp) / "videos")


class VideoMMEDownloadTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        self.api.list_repo_files.return_value = ["videomme/test.parquet", *downloader.ARCHIVES]
        self.download = Mock(return_value="fixture.parquet")

    def prepare(self, root, **kwargs):
        return downloader.download_dataset(root, api=self.api, download=self.download,
                                           expected_questions=1, expected_videos=1, **kwargs)

    def test_annotation_only_pins_revision_normalizes_and_never_downloads_zip(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(downloader, "read_videomme_annotations", return_value=[record()]):
            root = Path(temp) / "data"
            result = self.prepare(root, annotations_only=True)
            self.assertFalse(result["complete"])
            self.assertEqual(json.loads((root / "questions.json").read_text()), [record()])
            self.assertEqual(result["revision"], "a" * 40)
            self.assertEqual(self.download.call_count, 1)
            self.assertEqual(self.download.call_args.kwargs["revision"], "a" * 40)
            self.assertEqual(self.api.list_repo_files.call_args.kwargs["revision"], "a" * 40)
            self.api.dataset_info.reset_mock()
            self.prepare(root, annotations_only=True)
            self.api.dataset_info.assert_not_called()

    def test_revision_mismatch_and_unmarked_destination_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(downloader, "read_videomme_annotations", return_value=[record()]):
            root = Path(temp)
            self.prepare(root, annotations_only=True)
            self.api.dataset_info.return_value.sha = "b" * 40
            with self.assertRaisesRegex(ValueError, "Revision mismatch"):
                self.prepare(root, annotations_only=True, revision="other")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "existing.mp4").write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "without provenance"):
                self.prepare(root, annotations_only=True)

    def test_expected_full_counts_enforced(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(downloader, "read_videomme_annotations", return_value=[record()]):
            with self.assertRaisesRegex(ValueError, "2700 questions/900"):
                downloader.download_dataset(temp, annotations_only=True, api=self.api, download=self.download)
            self.assertFalse((Path(temp) / "questions.json").exists())

    def test_full_download_one_archive_at_a_time_and_resume(self):
        archives = downloader.ARCHIVES[:2]
        rows = [record(), record(video_id="002", videoID="other", question_id="002-1", url="other")]
        self.api.list_repo_files.return_value = ["videomme/test.parquet", *archives]
        transient_dirs = []

        def fake_download(**kwargs):
            if kwargs["filename"].endswith(".parquet"):
                return "fixture.parquet"
            self.assertTrue(all(not path.exists() for path in transient_dirs))
            directory = Path(kwargs["local_dir"])
            transient_dirs.append(directory)
            path = directory / kwargs["filename"]
            video = "abc_-123" if kwargs["filename"] == archives[0] else "other"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(f"videos/{video}.mp4", b"video " + video.encode())
            return str(path)

        with tempfile.TemporaryDirectory() as temp, patch.object(downloader, "ARCHIVES", archives), patch.object(downloader, "read_videomme_annotations", return_value=rows):
            root, jobfs = Path(temp) / "data", Path(temp) / "jobfs"
            download = Mock(side_effect=fake_download)
            result = downloader.download_dataset(root, jobfs=jobfs, api=self.api, download=download,
                                                   expected_questions=2, expected_videos=2)
            self.assertTrue(result["complete"])
            self.assertTrue(all(not path.exists() for path in transient_dirs))
            self.assertEqual(len(load_videomme(root)), 2)
            download.reset_mock()
            downloader.download_dataset(root, jobfs=jobfs, api=self.api, download=download,
                                        expected_questions=2, expected_videos=2)
            self.assertEqual(download.call_count, 1)  # annotations only; ZIPs verified locally

    def test_incomplete_official_archive_listing_rejected(self):
        self.api.list_repo_files.return_value = ["videomme/test.parquet", "videos_chunked_01.zip"]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "20 independent"):
                self.prepare(temp)

    def test_pbs_uses_transient_env_storage_and_data_root_alias(self):
        text = SCRIPT.with_name("prepare_videomme_data.pbs").read_text()
        for expected in ("#PBS -q copyq", "walltime=02:00:00", "jobfs=40GB",
                         "storage=scratch/hn98+gdata/hn98", "conda-unpack", "DATA_ROOT",
                         'ENV_ROOT="${PBS_JOBFS:?}/env"', "python-deps", "pyarrow==23.0.1"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
