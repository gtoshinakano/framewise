"""What may make a picture result stale: only a change of the cut's boundaries makes the first frame stale, and
audio-side work (subtitles) never makes a picture result stale (2026-11-24). Since D37 a subtitle run makes nothing stale at
all (tests/test_subtitle_pipeline.py); the marks earlier runs left are cleared once."""
import json
import unittest

import tests.test_analysis as analysis_tests
from app import db, structured_results

KINDS = ('initial_frame', 'keyframe_analysis', 'performance_beats', 'consistency_summary')


class StaleRulesTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def results(self, reason=None):
        with db.connect() as conn:
            for cut_id in ('c1', 'c2'):
                for kind in KINDS:
                    kind_name, title = structured_results.KINDS[kind]
                    conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json,stale_reason)
                        VALUES (?,?,?,?,'{}','model',?,?)""", (f'{cut_id}-{kind}', cut_id, kind_name, title,
                                                               json.dumps({'structured_type': kind}), reason))
                conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,stale_reason)
                    VALUES (?,?,'production_brief','Cut summary','x','model',?)""", (f'{cut_id}-summary', cut_id, reason))

    def stale(self):
        return {row['id']: row['stale_reason'] for row in db.all_rows('SELECT id,stale_reason FROM analysis_results WHERE stale_reason IS NOT NULL')}

    def test_marks_left_by_subtitle_runs_are_cleared_once(self):
        self.results('Episode subtitles changed')
        with db.connect() as conn:
            conn.execute("UPDATE analysis_results SET stale_reason='Transcript changed' WHERE id='c2-summary'")
            conn.execute("UPDATE episodes SET transcript_stale_reason='Excluded audio ranges changed',speaker_attribution_stale_reason='Name glossary changed'")
            conn.execute("DELETE FROM app_flags WHERE name='prep-flags-cleanup-v1'")
        db.init_db()
        self.assertEqual(self.stale(), {'c2-summary': 'Transcript changed'})  # the person's own edits leave the same reason: kept
        episode = db.one("SELECT transcript_stale_reason,speaker_attribution_stale_reason FROM episodes WHERE id='e'")
        self.assertEqual(episode, {'transcript_stale_reason': None, 'speaker_attribution_stale_reason': None})

    def test_first_frame_box_edits_never_stale_the_first_frame_description(self):
        self.results()
        with db.connect() as conn:
            structured_results.invalidate_detections(conn, 'c1')
        self.assertEqual(set(self.stale()), {'c1-keyframe_analysis', 'c1-performance_beats', 'c1-consistency_summary', 'c1-summary'})
        with db.connect() as conn:
            structured_results.invalidate_cut_timing(conn, ['c1'])
        self.assertEqual(self.stale()['c1-initial_frame'], 'Cut timing changed')

    def test_marks_the_rules_forbid_are_cleared_once(self):
        self.results('Episode subtitles changed')
        with db.connect() as conn:
            conn.execute("UPDATE analysis_results SET stale_reason='Cut timing changed' WHERE id='c2-initial_frame'")
            conn.execute("DELETE FROM app_flags WHERE name='picture-stale-rules-v1'")
            db.clear_picture_stale_marks(conn)
        stale = self.stale()
        self.assertNotIn('c1-initial_frame', stale)
        self.assertNotIn('c1-keyframe_analysis', stale)
        self.assertEqual(stale['c2-initial_frame'], 'Cut timing changed')
        self.assertEqual(stale['c1-performance_beats'], 'Episode subtitles changed')
        with db.connect() as conn:  # the flag is set: a later mark stays
            conn.execute("UPDATE analysis_results SET stale_reason='Episode subtitles changed' WHERE id='c1-initial_frame'")
            db.clear_picture_stale_marks(conn)
        self.assertIn('c1-initial_frame', self.stale())


if __name__ == '__main__':
    unittest.main()
