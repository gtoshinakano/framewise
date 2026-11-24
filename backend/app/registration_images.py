"""Small JPEGs captured from the editor's stopped video."""
import io
import logging
import tempfile

from fastapi import HTTPException

from . import db

MAX_BYTES = 1024 * 1024
MAX_SIDE = 1600
log = logging.getLogger(__name__)


def read_image(upload):
    from PIL import Image, UnidentifiedImageError

    raw = upload.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, 'Registration image is too large.')
    try:
        with Image.open(io.BytesIO(raw), formats=['JPEG']) as image:
            if not (1 <= image.width <= MAX_SIDE and 1 <= image.height <= MAX_SIDE):
                raise HTTPException(422, 'Registration image dimensions are invalid.')
            result = io.BytesIO()
            image.convert('RGB').save(result, format='JPEG', quality=88)
            return result.getvalue(), list(image.size)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, 'Could not read the registration image.') from exc


def thumbnail(project_id, character_id):
    return db.DATA_DIR / 'projects' / project_id / 'character-thumbnails' / f'{character_id}.jpg'


def store_thumbnail(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.partial', delete=False) as output:
        temp = path.parent / output.name
        try:
            output.write(data)
            output.flush()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def unlink(path):
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning('Could not remove registration asset %s', path)
