"""Cut Assistance: a per-cut chat with tool calling over saved analysis.

The API (``api.py``) records sessions, user messages and queued turns in
SQLite. A thread in the inference worker (``runner.py``) claims turns with the
shared lease discipline, streams provider output into SQLite, and runs tools
from ``tools.py``. Skills are Markdown guides in ``skills/`` that the model
loads on demand. Edits to stored results are only proposals until the user
applies them.
"""
