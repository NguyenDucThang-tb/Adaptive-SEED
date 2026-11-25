import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seed_video.data.countervid import select_freeform


def example(name="videos/a.mp4"):
    return dict(id="sample", task_type="A_freeform", preference_signal="textual",
                prompt="<video>\nDescribe the events.", chosen_answer="An event.",
                videos_file_names=[name], video_roles=["context"], base_identifier="a")


def test_select_freeform(tmp_path):
    video = tmp_path / "train" / "videos" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    rows = [example(), {**example(), "task_type": "C_binary_choice"}]
    result = select_freeform(rows, tmp_path, "train")
    assert len(result) == 1
    assert result[0]["question"] == "Describe the events."
    assert result[0]["video_path"] == str(video)


def test_rejects_missing_and_outside_video(tmp_path):
    with pytest.raises(ValueError):
        select_freeform([example()], tmp_path, "train")
    outside = tmp_path / "outside.mp4"
    outside.touch()
    with pytest.raises(ValueError):
        select_freeform([example("../outside.mp4")], tmp_path, "train")


def test_rejects_duplicate_ids(tmp_path):
    video = tmp_path / "train" / "videos" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    with pytest.raises(ValueError):
        select_freeform([example(), example()], tmp_path, "train")
