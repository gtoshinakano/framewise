import sqlite3
import unittest

from app import episode_events


class TriggerDefinitionTests(unittest.TestCase):
    def test_stale_trigger_is_replaced_and_current_one_kept(self):
        conn = sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, state TEXT, detail TEXT)')
        conn.execute('CREATE TABLE counter (n INTEGER)')
        conn.execute('INSERT INTO counter VALUES (0)')
        conn.execute('CREATE TRIGGER stale AFTER UPDATE OF state,online ON t WHEN NEW.online IS NOT OLD.online BEGIN UPDATE counter SET n=n+1; END')
        conn.execute('INSERT INTO t VALUES (1, "a", "")')
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute('UPDATE t SET state="b", detail="x"')
        episode_events._trigger(conn, 'stale', 't', 'UPDATE', 'state,detail', 'UPDATE counter SET n=n+1')
        conn.execute('UPDATE t SET state="b"')
        self.assertEqual(conn.execute('SELECT n FROM counter').fetchone()[0], 1)
        before = conn.execute("SELECT sql FROM sqlite_master WHERE name='stale'").fetchone()
        episode_events._trigger(conn, 'stale', 't', 'UPDATE', 'state,detail', 'UPDATE counter SET n=n+1')
        self.assertEqual(conn.execute("SELECT sql FROM sqlite_master WHERE name='stale'").fetchone(), before)


if __name__ == '__main__':
    unittest.main()
