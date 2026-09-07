from __future__ import annotations

import unittest

import pandas as pd

from pgnn_classifier.data.splitting import assert_no_group_leakage, make_grouped_splits


class SplittingTests(unittest.TestCase):
    def test_tic_and_injection_groups_do_not_leak(self) -> None:
        rows = []
        for index in range(40):
            label = index % 2
            rows.append(
                {
                    "sample_id": f"base-{index}",
                    "tic_id": str(index),
                    "label": label,
                    "subclass_label": -1 if label else 0,
                    "physics_eligible": bool(label),
                    "source_kind": "observed",
                    "parent_tic_id": "",
                }
            )
            rows.append(
                {
                    "sample_id": f"inj-{index}",
                    "tic_id": f"injection-{index}",
                    "label": 1,
                    "subclass_label": -1,
                    "physics_eligible": True,
                    "source_kind": "injection",
                    "parent_tic_id": str(index),
                }
            )
        split = make_grouped_splits(pd.DataFrame(rows), seed=9)
        assert_no_group_leakage(split)
        for index in range(40):
            base_split = split.loc[split.sample_id == f"base-{index}", "split"].iloc[0]
            injection_split = split.loc[split.sample_id == f"inj-{index}", "split"].iloc[0]
            self.assertEqual(base_split, injection_split)

    def test_unequal_group_sizes_are_balanced_by_samples(self) -> None:
        rows = []
        group_sizes = [20, 15, 10, 8, 6, 5, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1]
        for group, size in enumerate(group_sizes):
            for event in range(size):
                label = (group + event) % 2
                rows.append(
                    {
                        "sample_id": f"{group}-{event}", "tic_id": str(group), "label": label,
                        "subclass_label": -1 if label else event % 4, "physics_eligible": bool(label),
                        "source_kind": "observed", "parent_tic_id": "",
                    }
                )
        split = make_grouped_splits(pd.DataFrame(rows), seed=3)
        assert_no_group_leakage(split)
        counts = split["split"].value_counts()
        self.assertEqual(set(counts.index), {"train", "validation", "calibration", "test"})
        self.assertLess(abs(counts["train"] / len(split) - 0.70), 0.15)

    def test_each_split_gets_both_classes_when_target_groups_allow_it(self) -> None:
        rows = []
        # This deliberately combines large positive-only groups, small
        # negative-only groups, and mixed groups. A size-only greedy allocator
        # can leave a holdout with positives only despite ample class support.
        group_counts = [
            (22, 0), (20, 0), (19, 2), (19, 0), (18, 5), (27, 0),
            (0, 17), (0, 8), (22, 0), (23, 0), (19, 13),
        ]
        for group, (positives, negatives) in enumerate(group_counts):
            for event in range(positives):
                rows.append(
                    {
                        "sample_id": f"{group}-p-{event}", "tic_id": str(group), "label": 1,
                        "subclass_label": -1, "physics_eligible": True,
                        "source_kind": "observed", "parent_tic_id": "",
                    }
                )
            for event in range(negatives):
                rows.append(
                    {
                        "sample_id": f"{group}-n-{event}", "tic_id": str(group), "label": 0,
                        "subclass_label": event % 4, "physics_eligible": False,
                        "source_kind": "observed", "parent_tic_id": "",
                    }
                )

        manifest = pd.DataFrame(rows)
        split = make_grouped_splits(manifest, seed=3)
        for split_name in ("train", "validation", "calibration", "test"):
            labels = set(split.loc[split["split"] == split_name, "label"].astype(int))
            self.assertEqual(labels, {0, 1}, split_name)

    def test_assignment_is_deterministic_under_manifest_row_reordering(self) -> None:
        rows = []
        for group in range(32):
            for event in range(1 + group % 4):
                label = (group + event) % 2
                rows.append(
                    {
                        "sample_id": f"{group}-{event}", "tic_id": str(group), "label": label,
                        "subclass_label": -1 if label else event % 4,
                        "physics_eligible": bool(label), "source_kind": "observed",
                        "parent_tic_id": "",
                    }
                )
        manifest = pd.DataFrame(rows)
        first = make_grouped_splits(manifest, seed=17)
        reordered = make_grouped_splits(manifest.sample(frac=1.0, random_state=91), seed=17)
        first_map = first.set_index("sample_id")["split"].to_dict()
        reordered_map = reordered.set_index("sample_id")["split"].to_dict()
        self.assertEqual(first_map, reordered_map)


if __name__ == "__main__":
    unittest.main()
