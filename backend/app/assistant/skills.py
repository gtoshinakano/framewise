"""Markdown skills loaded by the model on demand.

Each ``skills/*.md`` file starts with a small front matter block::

    ---
    name: h3_animation_prompt
    title: H3 animation prompt
    description: One line shown to the model in the skill index.
    examples: Prompt chip one | Prompt chip two
    ---

Only the name and description are sent with every request; the body is
returned by the ``load_skill`` tool. Adding a skill means adding a file.
"""
import re
from functools import lru_cache
from pathlib import Path

SKILL_DIR = Path(__file__).parent / 'skills'
NAME = re.compile(r'^[a-z][a-z0-9_]{1,40}$')


def parse(text):
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.S)
    if not match:
        raise ValueError('Skill file is missing front matter')
    meta = {}
    for line in match.group(1).splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            meta[key.strip()] = value.strip()
    if not NAME.match(meta.get('name', '')) or not meta.get('description'):
        raise ValueError('Skill front matter requires a snake_case name and a description')
    return {'name': meta['name'], 'title': meta.get('title') or meta['name'], 'description': meta['description'],
            'scope': meta.get('scope') if meta.get('scope') in ('cut', 'sequence', 'any') else 'cut',
            'examples': [item.strip() for item in meta.get('examples', '').split('|') if item.strip()],
            'body': text[match.end():].strip()}


@lru_cache(maxsize=1)
def _load_all():
    skills = {}
    for path in sorted(SKILL_DIR.glob('*.md')):
        skill = parse(path.read_text(encoding='utf-8'))
        skills[skill['name']] = skill
    return skills


def all_skills():
    return list(_load_all().values())


def get(name):
    return _load_all().get(name)


def names():
    return list(_load_all())


def index_text(scope='cut'):
    """The skills offered to a chat: those of its scope (a cut or a sequence) and those that fit both."""
    return '\n'.join(f"- {skill['name']}: {skill['description']}" for skill in all_skills() if skill['scope'] in (scope, 'any'))
