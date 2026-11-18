import unittest

from app import target_tracking as tracking
from app.target_tracking import Slot

SAME = .85
RED, BLUE, GREEN, GOLD = [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]


def box(x, size=40):
    return [x, 5, x + size, 75]


def person(color, x, small=False):
    return {'box': box(x), 'vector': color, 'small': small}


class TargetTrackingTest(unittest.TestCase):
    def slots(self, *people):
        return [Slot(number, 'first_frame', character, 'inherited' if character else None, box(x), color)
                for number, (color, x, character) in enumerate(people)]

    def place(self, slots, detections, **options):
        return {(item['slot'], item['detection']) for item in tracking.assign_frame(slots, detections, SAME, **options)}

    def test_order_follows_the_first_frame_whatever_order_the_detector_answers(self):
        slots = self.slots((RED, 5, 'hero'), (BLUE, 60, None))
        # AniRef lists the blue person first this time.
        self.assertEqual(self.place(slots, [person(BLUE, 60), person(RED, 5)]), {(1, 0), (0, 1)})

    def test_a_person_keeps_their_slot_after_walking_across_the_frame_and_after_leaving(self):
        slots = self.slots((RED, 5, None), (BLUE, 60, None))
        self.assertEqual(self.place(slots, [person(RED, 30)]), {(0, 0)})            # moved, still the same person
        self.assertEqual(self.place(slots, [person(BLUE, 70)]), {(1, 0)})           # red left the frame
        self.assertEqual(self.place(slots, [person(RED, 10), person(BLUE, 60)]), {(0, 0), (1, 1)})

    def test_someone_new_gets_the_next_free_slot_and_returns_to_it(self):
        slots = self.slots((RED, 5, None))
        placed = tracking.assign_frame(slots, [person(RED, 5), person(GOLD, 60)], SAME)
        self.assertEqual([(item['slot'], item['new']) for item in placed], [(0, False), (1, True)])
        self.assertEqual([slot.origin for slot in slots], ['first_frame', 'appeared'])
        self.assertEqual(self.place(slots, [person(GOLD, 20)]), {(1, 0)})

    def test_without_a_first_frame_the_first_evaluated_frame_defines_the_order(self):
        slots = []
        tracking.assign_frame(slots, [person(RED, 5), person(BLUE, 60)], SAME)
        self.assertEqual([slot.number for slot in slots], [0, 1])
        self.assertEqual(self.place(slots, [person(BLUE, 55), person(RED, 10)]), {(1, 0), (0, 1)})

    def test_a_frame_keeps_three_boxes_known_characters_first(self):
        slots = self.slots((RED, 5, None), (BLUE, 60, None), (GREEN, 110, 'hero'))
        detections = [person(RED, 5), person(BLUE, 60), person(GREEN, 110), person(GOLD, 160)]
        placed = tracking.assign_frame(slots, detections, SAME)
        self.assertEqual(len(placed), 3)
        self.assertIn(2, {item['slot'] for item in placed})                 # the known character stays
        self.assertNotIn(3, {item['slot'] for item in placed})              # the fourth person is left out
        self.assertEqual(len(slots), 3)                                     # and does not become a slot

    def test_new_people_only_fill_what_matched_people_leave_free(self):
        slots = self.slots((RED, 5, None), (BLUE, 60, None))
        placed = tracking.assign_frame(slots, [person(RED, 5), person(BLUE, 60), person(GREEN, 110), person(GOLD, 160)], SAME)
        self.assertEqual({item['slot'] for item in placed}, {0, 1, 2})
        self.assertEqual(len(slots), 3)

    def test_never_more_slots_than_colours(self):
        slots = [Slot(number, 'first_frame', None, None, None, [0.] * number + [1.]) for number in range(tracking.MAX_SLOTS)]
        newcomer = {'box': box(5), 'vector': [.5] * (tracking.MAX_SLOTS + 1), 'small': False}
        self.assertEqual(tracking.assign_frame(slots, [newcomer], SAME), [])
        self.assertEqual(len(slots), tracking.MAX_SLOTS)

    def test_small_boxes_follow_a_slot_by_overlap_but_never_start_one(self):
        slots = self.slots((RED, 5, None))
        tiny = {'box': box(7, 36), 'vector': None, 'small': True}
        self.assertEqual(self.place(slots, [tiny]), {(0, 0)})
        stray = {'box': box(120, 8), 'vector': None, 'small': True}
        self.assertEqual(tracking.assign_frame(slots, [stray], SAME), [])
        self.assertEqual(len(slots), 1)

    def test_boxes_the_user_placed_block_their_slot_and_use_up_the_cap(self):
        slots = self.slots((RED, 5, None), (BLUE, 60, None))
        # The user already placed slot 0's box in this frame, so the red-looking detection cannot take it.
        placed = tracking.assign_frame(slots, [person(RED, 5), person(BLUE, 60)], SAME, max_boxes=2, blocked={0})
        self.assertEqual({item['slot'] for item in placed}, {1, 2})
        self.assertEqual(tracking.assign_frame(slots, [person(BLUE, 60)], SAME, max_boxes=0, blocked={0, 1, 2}), [])

    def test_overlap_alone_does_not_merge_two_different_people(self):
        slots = self.slots((RED, 5, None))
        # Same place, very different look: a different person (the identity floor is below the match threshold).
        placed = tracking.assign_frame(slots, [person(BLUE, 5)], SAME)
        self.assertEqual([(item['slot'], item['new']) for item in placed], [(1, True)])


if __name__ == '__main__':
    unittest.main()
