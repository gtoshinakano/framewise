"""Small real-media API smoke test. Requires FFmpeg/FFprobe on PATH."""

import json
import os
import sqlite3
import subprocess
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient


def sample_video(path: Path):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x180:r=25:d=1",
        "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=1",
        "-f", "lavfi", "-i", "color=c=green:s=320x180:r=25:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True)


class WorkflowTest(unittest.TestCase):
    def test_low_frame_rate_cut_end_uses_decoded_frame(self):
        from app import media
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            video = path / "low-fps.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=10:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
            frame = path / "last.jpg"
            timestamp = media.extract_cut_frame(video, frame, 1.999999, 1.0, 2.0, last=True)
            self.assertTrue(frame.is_file())
            self.assertGreaterEqual(timestamp, 1.0)
            self.assertLess(timestamp, 2.0)
            self.assertAlmostEqual(timestamp, 1.9, places=2)
            offset_video = path / "offset.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-output_ts_offset", "5", str(offset_video)], check=True)
            timestamp = media.extract_cut_frame(offset_video, frame, 1.999999, 1.0, 2.0, last=True)
            self.assertAlmostEqual(timestamp, 1.96, places=2)

    def test_low_frame_rate_no_audio_workflow(self):
        from app.main import app
        from app import db, media
        with tempfile.TemporaryDirectory() as temp:
            old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
            db.DATA_DIR = media.DATA_DIR = Path(temp)
            db.DB_PATH = Path(temp) / "app.sqlite3"
            try:
                video = Path(temp) / "sample.mp4"
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=10:d=1", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=10:d=1", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
                with TestClient(app) as client:
                    response = client.post("/api/import", data={"project_title": "Low FPS"}, files={"file": ("sample.mp4", video.read_bytes(), "video/mp4")})
                    self.assertEqual(response.status_code, 200, response.text)
                    episode_id = response.json()["episode"]["id"]
                    def wait_ready():
                        for _ in range(150):
                            episode = client.get(f"/api/episodes/{episode_id}").json()
                            if episode["status"] == "ready":
                                return episode
                            if episode["status"] == "failed":
                                self.fail(episode["error"])
                            time.sleep(.1)
                        self.fail("Media worker timed out")
                    episode = wait_ready()
                    self.assertFalse(episode["has_audio"])
                    self.assertEqual(client.post(f"/api/episodes/{episode_id}/run", json={"pipeline": "hardcuts"}).status_code, 200)
                    episode = wait_ready()
                    self.assertEqual(len(episode["cuts"]), 2)
                    self.assertAlmostEqual(episode["cuts"][0]["last_frame_time"], .9, places=2)
                    last = episode["cuts"][-1]
                    self.assertAlmostEqual(last["first_frame_time"], 1.0, places=2)
                    self.assertAlmostEqual(last["last_frame_time"], 1.9, places=2)
                    self.assertEqual(client.get(f"/api/cuts/{last['id']}/frame/last").status_code, 200)
            finally:
                db.DATA_DIR, db.DB_PATH, media.DATA_DIR = old

    def test_existing_job_table_migrates(self):
        from app import db
        old_data_dir, old_db_path = db.DATA_DIR, db.DB_PATH
        with tempfile.TemporaryDirectory() as temp:
            try:
                db.DATA_DIR = Path(temp)
                db.DB_PATH = Path(temp) / "app.sqlite3"
                with closing(sqlite3.connect(db.DB_PATH)) as conn:
                    conn.execute("""CREATE TABLE jobs (
                        id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, pipeline TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'queued',
                        error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
                    conn.execute("""CREATE TABLE cuts (
                        id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, cut_index INTEGER NOT NULL,
                        start_time REAL NOT NULL, end_time REAL NOT NULL, title TEXT NOT NULL DEFAULT '',
                        description TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '',
                        results_json TEXT NOT NULL DEFAULT '{}', UNIQUE(episode_id,cut_index))""")
                    conn.execute("INSERT INTO cuts (id,episode_id,cut_index,start_time,end_time) VALUES ('old-cut','episode',0,0,1)")
                    conn.commit()
                db.init_db()
                with db.connect() as conn:
                    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
                self.assertTrue({"payload_json", "lease_owner", "lease_until", "attempt_count"}.issubset(columns))
                self.assertEqual(db.one("SELECT active FROM cuts WHERE id='old-cut'")["active"], 1)
            finally:
                db.DATA_DIR, db.DB_PATH = old_data_dir, old_db_path

    def test_job_claim_lease_and_settings_snapshot(self):
        from app import db, worker
        old_data_dir, old_db_path = db.DATA_DIR, db.DB_PATH
        with tempfile.TemporaryDirectory() as temp:
            try:
                db.DATA_DIR = Path(temp)
                db.DB_PATH = Path(temp) / "app.sqlite3"
                db.init_db()
                with db.connect() as conn:
                    conn.execute("INSERT INTO projects (id,title) VALUES ('project','Test')")
                    conn.execute("INSERT INTO episodes (id,project_id,title,original_filename,original_ext,settings_json) VALUES ('episode','project','Test','sample.mp4','.mp4',?)",
                                 ('{"scene_threshold":0.32,"chunk_duration":60}',))
                job_id = worker.enqueue("episode", "hardcuts")
                with db.connect() as conn:
                    conn.execute("UPDATE episodes SET settings_json=? WHERE id='episode'", ('{"scene_threshold":0.8,"chunk_duration":30}',))
                claimed = worker.claim_job()
                self.assertEqual(claimed[0], job_id)
                self.assertEqual(db.one("SELECT attempt_count FROM jobs WHERE id=?", (job_id,))["attempt_count"], 1)
                payload = json.loads(db.one("SELECT payload_json FROM jobs WHERE id=?", (job_id,))["payload_json"])
                self.assertEqual(payload["settings"], {"scene_threshold": 0.32, "chunk_duration": 60})
                self.assertIsNone(worker.claim_job())
                with db.connect() as conn:
                    conn.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job_id,))
                second = worker.claim_job()
                self.assertEqual(second[0], job_id)
                self.assertNotEqual(second[1], claimed[1])
                self.assertEqual(db.one("SELECT attempt_count FROM jobs WHERE id=?", (job_id,))["attempt_count"], 2)
                with db.connect() as conn:
                    conn.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job_id,))
                self.assertEqual(worker.claim_job()[0], job_id)
                with db.connect() as conn:
                    conn.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job_id,))
                self.assertIsNone(worker.claim_job())
                self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job_id,))["status"], "failed")
                self.assertEqual(db.one("SELECT status FROM episodes WHERE id='episode'")["status"], "failed")
            finally:
                db.DATA_DIR, db.DB_PATH = old_data_dir, old_db_path

    def test_import_prepare_detect_edit_and_persist(self):
        with tempfile.TemporaryDirectory() as temp:
            os.environ["DATA_DIR"] = temp
            from app.main import app
            from app import db, media
            old_data_dir, old_db_path, old_media_dir = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
            db.DATA_DIR, db.DB_PATH, media.DATA_DIR = Path(temp), Path(temp) / "app.sqlite3", Path(temp)
            self.addCleanup(setattr, db, "DATA_DIR", old_data_dir)
            self.addCleanup(setattr, db, "DB_PATH", old_db_path)
            self.addCleanup(setattr, media, "DATA_DIR", old_media_dir)
            video = Path(temp) / "sample.mp4"
            sample_video(video)
            with TestClient(app) as client:
                bad = client.post("/api/import", data={"project_title": "Bad"}, files={"file": ("notes.txt", b"text", "text/plain")})
                self.assertEqual(bad.status_code, 400)
                response = client.post("/api/import", data={"project_title": "Test film", "episode_title": "Opening"}, files={"file": ("sample.mp4", video.read_bytes(), "video/mp4")})
                self.assertEqual(response.status_code, 200, response.text)
                project_id = response.json()["project"]["id"]
                episode_id = response.json()["episode"]["id"]

                def wait_ready():
                    for _ in range(100):
                        episode = client.get(f"/api/episodes/{episode_id}").json()
                        if episode["status"] == "ready":
                            return episode
                        if episode["status"] == "failed":
                            self.fail(episode["error"])
                        time.sleep(0.1)
                    self.fail("Media job timed out")

                episode = wait_ready()
                self.assertGreater(episode["duration"], 2.9)
                status = client.get(f"/api/episodes/{episode_id}/status")
                self.assertEqual(status.json()["status"], "ready")
                self.assertEqual(status.json()["job_id"], episode["jobs"][0]["id"])
                self.assertNotIn("cuts", status.json())
                self.assertEqual(len(episode["chunks"]), 1)
                self.assertTrue(episode["has_audio"])
                self.assertEqual(client.get(f"/api/episodes/{episode_id}/media/proxy").status_code, 200)
                self.assertEqual(client.get(f"/api/episodes/{episode_id}/media/proxy", headers={"Range": "bytes=0-99"}).status_code, 206)
                self.assertEqual(client.get(f"/api/episodes/{episode_id}/media/frame_000001.jpg").status_code, 200)
                self.assertEqual(client.get(f"/api/episodes/{episode_id}/frames").json()["frames"][0]["time"], 0)
                self.assertTrue(client.get(f"/api/episodes/{episode_id}/waveform").json()["peaks"])
                response = client.post(f"/api/episodes/{episode_id}/run", json={"pipeline": "hardcuts"})
                self.assertEqual(response.status_code, 200, response.text)
                episode = wait_ready()
                self.assertGreaterEqual(len(episode["cuts"]), 2)
                cut = episode["cuts"][0]
                first_frame = client.get(f"/api/cuts/{cut['id']}/frame/first")
                last_frame = client.get(f"/api/cuts/{cut['id']}/frame/last")
                self.assertEqual(first_frame.status_code, 200)
                self.assertEqual(last_frame.status_code, 200)
                self.assertEqual(first_frame.headers["content-type"], "image/jpeg")
                self.assertLessEqual(cut["start_time"], cut["first_frame_time"])
                self.assertLess(cut["last_frame_time"], cut["end_time"])
                self.assertNotEqual(first_frame.content, client.get(f"/api/cuts/{episode['cuts'][1]['id']}/frame/first").content)
                response = client.put(f"/api/cuts/{cut['id']}", json={"title": "Opening red shot", "description": "Red field", "notes": "Keep", "results": {"manual": True}})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["results"]["manual"], True)
                response = client.post(f"/api/cuts/{cut['id']}/analysis-results", json={"kind": "first_frame", "content": "Red image"})
                self.assertEqual(response.status_code, 200, response.text)
                result_id = response.json()["id"]
                response = client.put(f"/api/analysis-results/{result_id}", json={"title": "Opening frame", "content": "Solid red image"})
                self.assertEqual(response.status_code, 200, response.text)
                revised_id = response.json()["id"]
                self.assertEqual(response.json()["parent_id"], result_id)
                self.assertEqual(len(client.get(f"/api/cuts/{cut['id']}/analysis-results").json()), 1)
                self.assertEqual(len(client.get(f"/api/cuts/{cut['id']}/analysis-results?history=true").json()), 2)
                self.assertEqual(client.post(f"/api/episodes/{episode_id}/run", json={"pipeline": "hardcuts"}).status_code, 200)
                episode = wait_ready()
                self.assertEqual(episode["cuts"][0]["title"], "Opening red shot")
                self.assertEqual(client.get(f"/api/cuts/{cut['id']}/analysis-results").json()[0]["id"], revised_id)
                self.assertEqual(db.one("SELECT title FROM projects WHERE id=?", (project_id,))["title"], "Test film")
                self.assertEqual(client.get(f"/api/projects/{project_id}").json()["episodes"][0]["id"], episode_id)
                context = client.post(f"/api/projects/{project_id}/context", json={"kind": "character", "title": "Lead", "content": "Wears a red coat", "episode_id": episode_id})
                self.assertEqual(context.status_code, 200, context.text)
                context_id = context.json()["id"]
                self.assertEqual(client.get(f"/api/projects/{project_id}/context").json()[0]["content"], "Wears a red coat")
                self.assertEqual(client.put(f"/api/context/{context_id}", json={"kind": "character", "title": "Lead", "content": "Wears a blue coat", "episode_id": episode_id}).json()["content"], "Wears a blue coat")
                self.assertEqual(client.patch(f"/api/projects/{project_id}", json={"title": "Test film", "synopsis": "A color study"}).status_code, 200)
                transcript = client.post(f"/api/episodes/{episode_id}/transcript", json={"start_time": 0, "end_time": 0.8, "text": "Opening line", "speaker_label": "Lead", "character_context_id": context_id})
                self.assertEqual(transcript.status_code, 200, transcript.text)
                revised_transcript = client.put(f"/api/transcript-segments/{transcript.json()['id']}", json={"start_time": 0, "end_time": 0.8, "text": "Updated opening line", "speaker_label": "Lead", "character_context_id": context_id})
                self.assertEqual(revised_transcript.status_code, 200, revised_transcript.text)
                self.assertEqual(revised_transcript.json()["parent_id"], transcript.json()["id"])
                self.assertEqual(len(client.get(f"/api/episodes/{episode_id}/transcript").json()), 1)
                self.assertEqual(len(client.get(f"/api/episodes/{episode_id}/transcript?history=true").json()), 2)
                self.assertEqual(client.post(f"/api/episodes/{episode_id}/transcript", json={"start_time": 0, "end_time": 9, "text": "Too long"}).status_code, 400)
                neighbor = client.post(f"/api/cuts/{episode['cuts'][1]['id']}/analysis-results", json={"kind": "cut_description", "content": "Blue image"})
                self.assertEqual(neighbor.status_code, 200, neighbor.text)
                analysis_context = client.get(f"/api/cuts/{cut['id']}/analysis-context").json()
                self.assertEqual(analysis_context["project"]["synopsis"], "A color study")
                self.assertIsNone(analysis_context["previous_cut"])
                self.assertEqual(analysis_context["next_cut"]["id"], episode["cuts"][1]["id"])
                self.assertEqual(analysis_context["neighbor_results"]["next"][0]["content"], "Blue image")
                self.assertEqual(analysis_context["context_items"][0]["id"], context_id)
                self.assertEqual(analysis_context["transcript_segments"][0]["id"], revised_transcript.json()["id"])
                second_cut_id = episode["cuts"][1]["id"]
                original_settings = episode["settings"]
                self.assertEqual(client.put(f"/api/episodes/{episode_id}/settings", json={**original_settings, "scene_threshold": 0.9}).status_code, 200)
                self.assertEqual(client.post(f"/api/episodes/{episode_id}/run", json={"pipeline": "hardcuts"}).status_code, 200)
                episode = wait_ready()
                self.assertEqual(len(episode["cuts"]), 1)
                self.assertIn(second_cut_id, {archived["id"] for archived in episode["archived_cuts"]})
                self.assertEqual(client.get(f"/api/cuts/{second_cut_id}/analysis-results").json()[0]["id"], neighbor.json()["id"])
                self.assertEqual(client.get(f"/api/cuts/{second_cut_id}/frame/first").status_code, 200)
                self.assertIsNone(client.get(f"/api/cuts/{second_cut_id}/analysis-context").json()["next_cut"])
                self.assertEqual(client.put(f"/api/episodes/{episode_id}/settings", json=original_settings).status_code, 200)
                self.assertEqual(client.post(f"/api/episodes/{episode_id}/run", json={"pipeline": "hardcuts"}).status_code, 200)
                episode = wait_ready()
                self.assertIn(second_cut_id, {active["id"] for active in episode["cuts"]})
                self.assertEqual(client.get(f"/api/cuts/{second_cut_id}/analysis-results").json()[0]["id"], neighbor.json()["id"])
                # Manual boundary changes preserve complete coverage and result history.
                first, second = episode["cuts"][:2]
                boundary = (first["start_time"] + first["end_time"]) / 2
                invalid = client.put(f"/api/cuts/{first['id']}/timing", json={"start_time": 0, "end_time": second["end_time"]})
                self.assertEqual(invalid.status_code, 400)
                response = client.put(f"/api/cuts/{first['id']}/timing", json={"start_time": 0, "end_time": boundary})
                self.assertEqual(response.status_code, 200, response.text)
                episode = wait_ready()
                self.assertAlmostEqual(episode["cuts"][0]["end_time"], boundary)
                self.assertAlmostEqual(episode["cuts"][1]["start_time"], boundary)
                self.assertEqual(client.get(f"/api/cuts/{second_cut_id}/frame/first").status_code, 200)
                self.assertEqual(client.post(f"/api/cuts/{first['id']}/merge-next").status_code, 200)
                episode = wait_ready()
                self.assertIn(second_cut_id, {item["id"] for item in episode["archived_cuts"]})
                self.assertEqual(client.get(f"/api/cuts/{second_cut_id}/analysis-results").json()[0]["id"], neighbor.json()["id"])
                if len(episode["cuts"]) > 1:
                    removed = episode["cuts"][0]
                    self.assertEqual(client.delete(f"/api/cuts/{removed['id']}").status_code, 200)
                    episode = wait_ready()
                    self.assertIn(removed["id"], {item["id"] for item in episode["archived_cuts"]})
                self.assertEqual(episode["cuts"][0]["start_time"], 0)
                self.assertAlmostEqual(episode["cuts"][-1]["end_time"], episode["duration"])
                for left, right in zip(episode["cuts"], episode["cuts"][1:]):
                    self.assertAlmostEqual(left["end_time"], right["start_time"])
                if len(episode["cuts"]) == 1:
                    self.assertEqual(client.delete(f"/api/cuts/{episode['cuts'][0]['id']}").status_code, 400)
                self.assertEqual(client.delete(f"/api/transcript-segments/{revised_transcript.json()['id']}").status_code, 200)
                self.assertEqual(client.get(f"/api/episodes/{episode_id}/transcript").json(), [])
                self.assertEqual(len(client.get(f"/api/episodes/{episode_id}/transcript?history=true").json()), 3)
                silent = Path(temp) / "silent.mp4"
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=11", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True)
                response = client.post(f"/api/projects/{project_id}/episodes", data={"title": "Silent scene"}, files={"file": ("silent.mp4", silent.read_bytes(), "video/mp4")})
                self.assertEqual(response.status_code, 200, response.text)
                silent_id = response.json()["id"]
                for _ in range(100):
                    silent_episode = client.get(f"/api/episodes/{silent_id}").json()
                    if silent_episode["status"] in ("ready", "failed"):
                        break
                    time.sleep(0.1)
                self.assertEqual(silent_episode["status"], "ready", silent_episode["error"])
                self.assertFalse(silent_episode["has_audio"])
                self.assertEqual(client.get(f"/api/episodes/{silent_id}/waveform").json()["peaks"], [])
                settings = {**silent_episode["settings"], "chunk_duration": 10}
                self.assertEqual(client.put(f"/api/episodes/{silent_id}/settings", json=settings).status_code, 200)
                self.assertEqual(client.post(f"/api/episodes/{silent_id}/run", json={"pipeline": "hardcuts"}).status_code, 200)
                for _ in range(100):
                    silent_episode = client.get(f"/api/episodes/{silent_id}").json()
                    if silent_episode["status"] in ("ready", "failed"):
                        break
                    time.sleep(0.1)
                self.assertEqual(silent_episode["status"], "ready", silent_episode["error"])
                self.assertEqual(len(silent_episode["chunks"]), 2)
                chunk = silent_episode["chunks"][0]
                self.assertEqual(client.put(f"/api/chunks/{chunk['id']}", json={"title": "Opening chunk", "notes": "Manual note"}).status_code, 200)
                self.assertEqual(client.get(f"/api/episodes/{silent_id}").json()["chunks"][0]["notes"], "Manual note")
                db.init_db()
                self.assertEqual(client.get(f"/api/episodes/{silent_id}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
