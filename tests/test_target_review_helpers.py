import unittest

from app import frame_confidence, merged_box, target_candidates, target_judge_contract, target_tracking


class TargetReviewHelpersTest(unittest.TestCase):
    def test_global_assignment_can_choose_a_better_pair_than_greedy(self):
        self.assertEqual(target_tracking.hungarian([[.10, .20], [.11, .90]]), [1, 0])
        slots = [target_tracking.Slot(0, 'first_frame', box=[0, 0, 40, 50], vector=[1., 0.])]
        detections = [{'box': [60, 0, 100, 50], 'vector': [0., 1.]}]
        placed, unmatched = target_tracking.assign_global(slots, detections, .85, {0: slots[0].box}, locked=True)
        self.assertEqual((placed, unmatched, len(slots)), ([], [0], 1))

    def test_split_by_predictions_requires_two_distinct_boxes(self):
        a, b, outer = [10, 10, 40, 80], [38, 15, 80, 80], [8, 8, 82, 82]
        proposal = merged_box.propose(outer, {0: a, 1: b})
        self.assertEqual((proposal['source'], proposal['parts']), ('prediction', [{'slot_hint': 0, 'box': a}, {'slot_hint': 1, 'box': b}]))
        self.assertIsNone(merged_box.propose(outer, {0: a, 1: a}))
        self.assertIsNone(merged_box.propose(outer, {0: a, 1: b}, others=[b]))

    def test_two_pose_crops_can_propose_a_split_by_keypoint_extents(self):
        def pose(box):
            x0, y0, x1, y1 = box
            return {'box': box, 'points': [[x, y, .9] for x in (x0 + 3, x1 - 3) for y in (y0 + 3, (y0 + y1) / 2, y1 - 3)]}
        a, b = [10, 10, 38, 78], [45, 15, 83, 80]
        result = merged_box.propose([7, 7, 86, 82], {0: a, 1: b}, [pose(a), pose(b)])
        self.assertEqual((result['source'], [part['slot_hint'] for part in result['parts']]), ('dwpose', [0, 1]))

    def test_a_union_of_two_manual_anchors_cannot_be_opened_as_a_third_box(self):
        anchors = [[10, 10, 40, 75], [45, 10, 80, 75]]
        raw = [{'box': [5, 5, 87, 79], 'rank': 0, 'raw_index': 0, 'small': False}]
        candidates, _, _ = target_candidates.prepare(raw, [], (100, 80), None, None, suppressed=anchors, protected=anchors)
        self.assertEqual(candidates, [])

    def test_review_records_motion_counts_unmatched_merges_and_duplicate_names(self):
        slots = [target_tracking.Slot(i, 'first_frame', character_id='hero') for i in range(2)]
        result = frame_confidence.review(
            [{'detection': 0, 'slot': 0, 'new': False, 'scores': {'cost': .48}}, {'detection': 1, 'slot': 1, 'new': False}],
            [{'box': [80, 0, 100, 60]}, {'box': [15, 0, 40, 30]}], ['unmatched-id'], slots, 3,
            {0: [0, 0, 25, 60]}, (120, 80), ['merge-id'])
        self.assertEqual(set(result['flags']), set(frame_confidence.PENALTIES))
        self.assertEqual(result['score'], 0.)
        self.assertEqual(result['flags']['duplicate_character'], [0, 1])

    def test_judge_schema_can_only_report_known_cells_slots_and_enums(self):
        schema = target_judge_contract.schema(['S1·F2'], ['U2-1'], [0, 1])
        good = {'cells': {'S1·F2': 'same'}, 'unmatched': {'U2-1': 'slot_2'}, 'cast_changes': 'no'}
        self.assertEqual(target_judge_contract.validate(good, schema), good)
        with self.assertRaises(ValueError):
            target_judge_contract.validate({**good, 'unmatched': {'U2-1': 'slot_3'}}, schema)
        with self.assertRaises(ValueError):
            target_judge_contract.validate({**good, 'cast_changes': False}, schema)
