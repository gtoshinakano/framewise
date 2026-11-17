"""Visual engine settings: which model runs each visual role (see ``visual_models``)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import db, visual_models

router = APIRouter()


class VisualModelChoice(BaseModel):
    role: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=80)


@router.get('/api/visual-models')
def get_visual_models():
    return visual_models.overview()


@router.put('/api/visual-models')
def put_visual_model(body: VisualModelChoice):
    """Choose the model for one role. Jobs already queued keep the model they snapshotted."""
    if body.role not in visual_models.ROLES:
        raise HTTPException(422, f'Unknown role {body.role!r}')
    if not visual_models.supports(body.model, body.role):
        raise HTTPException(422, f'{body.model!r} cannot run {visual_models.ROLE_LABELS[body.role].lower()}')
    if body.model != visual_models.DEFAULTS[body.role]:
        # The worker reports which model files exist; choosing a missing one would fail every job of that role.
        report, age = visual_models.reported()
        info = report.get(body.model) or {}
        if not info.get('available') or age is None or age > visual_models.STALE_SECONDS:
            detail = info.get('detail') if age is not None and age <= visual_models.STALE_SECONDS else 'the inference worker has not reported it'
            raise HTTPException(409, f'{visual_models.MODELS[body.model].label} is not available yet ({detail})')
    with db.connect() as conn:
        conn.execute('INSERT INTO visual_model_settings(role,model) VALUES (?,?) ON CONFLICT(role) DO UPDATE SET model=excluded.model',
                     (body.role, body.model))
    return visual_models.overview()
