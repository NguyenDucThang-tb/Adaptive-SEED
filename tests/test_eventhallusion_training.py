import unittest

from seed_video.data.experiment_split import split_eventhallusion


class EventHallusionTrainingSplitTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            {
                "subset": subset,
                "pair_id": f"{subset}/{video}/{question}",
                "role": "eh",
                "video": f"/videos/{subset}/{video}.mp4",
                "question": f"question {question}",
                "answer": "yes" if question % 2 else "no",
            }
            for subset, n_videos in (("eh_entire", 20), ("eh_mix", 30), ("eh_misleading", 18))
            for video in range(n_videos)
            for question in range(2)
        ]

    def test_split_is_deterministic_video_disjoint_and_complete(self):
        first = split_eventhallusion(self.items, seed=123)
        second = split_eventhallusion(self.items, seed=123)
        self.assertEqual(
            [[item["pair_id"] for item in split] for split in first],
            [[item["pair_id"] for item in split] for split in second],
        )
        video_sets = [{item["video"] for item in split} for split in first]
        self.assertFalse(video_sets[0] & video_sets[1])
        self.assertFalse(video_sets[0] & video_sets[2])
        self.assertFalse(video_sets[1] & video_sets[2])
        self.assertEqual(sum(map(len, first)), len(self.items))
        self.assertTrue(all(len(split) for split in first))

    def test_all_source_subsets_are_preserved_in_each_split(self):
        train, val, test = split_eventhallusion(self.items, seed=42)
        expected = {"eh_entire", "eh_mix", "eh_misleading"}
        for split in (train, val, test):
            self.assertEqual({item["subset"] for item in split}, expected)

    def test_rejects_invalid_fractions(self):
        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            split_eventhallusion(self.items, train_fraction=0.8, val_fraction=0.2)


if __name__ == "__main__":
    unittest.main()
