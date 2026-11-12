"""Build the detached summary's evidence without confusing audio with video."""
import json

from . import prompt_context

SYSTEM = '''You are an evidence reviewer for a pipeline that re-creates an anime episode shot by shot with AI. Each cut becomes one generated shot, made in three steps: (1) a still image of the opening frame, (2) the dialogue audio, (3) a video that animates the still.

Your task: for ONE cut, judge whether the supplied evidence is reliable and complete enough for those three steps, and report only what would make the still, the audio or the motion come out wrong. You are not looking at the video or the audio. Every observation, detection, transcript line, speaker link and subtitle reading was produced by another system and may contain errors.

## EVIDENCE

- First frame: a large vision model's description of the opening still at full resolution: composition, setting, lighting, mood, props, and for each marked character the appearance, position, pose, expression, emotion, gaze and action. It is the best source for everything static at the start of the cut.
- Video analysis: a smaller video model's motion direction for the whole cut: a summary, the camera movement, and for each character the activity (standing, walking, sitting ...), travel through the picture, body motion, whether the mouth visibly speaks, the emotion over time and how the character ends; plus timestamped events in seconds from the first frame. It is the only source for change over time and weaker than the first frame for fine static detail.
- end of cut (inside the video analysis) and "in last frame: True" on a character: the characters that a character box on the last frame confirms are still in frame. That presence is certain. It says nothing about anyone else: the detector often finds nobody in a last frame that still shows the same people, so a character missing from the list has NOT thereby left. Whether someone left is what that character's "end" in the video analysis says.
- Character names: every marked character was labelled in the user's detection workflow. In character_detections, label_source manual is the user's own label; other labels are recognition results. Names in the first frame and the video analysis come from those boxes.
- Transcript segments contain supplied words, timestamps, speaker labels and optional character links. They do not by themselves establish visible speaking.
- Subtitles are burned-in captions that a model read from the whole episode, independent of any cut. The evidence lists only the lines whose time range overlaps this cut, with episode-relative times; a line with crosses_cut true starts before or ends after the cut. Every subtitle in the episode is in the one language given as subtitles.language, so a language difference between a subtitle and the transcript is expected and never a contradiction. If subtitles.extracted is false, subtitles were not extracted for this episode: treat them as unavailable, not as absent from the video.
- Subtitle lines are unverified textual reference evidence read by a model; they may be a translation or paraphrase of speech, not a literal transcription, and a line marked confidence low had hard-to-read characters.
- Cuts analyzed by an older version have no motion direction; they carry a looser video analysis (overview and events per window) and a separate "Last frame" description. Review them the same way, using the last frame for how the cut ends.

## RULES

- Never reproduce subtitle text, OCR readings, or other on-screen text in the output.
- Emotion comes from the visual evidence (the first frame's emotion and mood, the video's emotion) and from the meaning of the spoken words. No acoustic analysis was made: do not describe vocal tone or delivery as observed.
- Do not invent actions, dialogue, timestamps, character identities, or other missing details.
- Treat user-confirmed corrections as authoritative workflow constraints.
- Do not treat repeated model predictions or previous model summaries as independent confirmation.

## WHAT TO CHECK, IN THIS ORDER

1. **Opening still:** are the characters, their positions, poses, outfits, props, lighting and framing complete and free of contradictions inside the first frame (a character listed but described as not visible, a prop held by two people, a person who matters but is unnamed)?

2. **Motion:** do camera movement, activity, travel, motion and events agree with each other and with the end of cut? Examples of real problems: travel "exits right" or end "left frame" for a character the end of cut confirms in frame (the box wins: the character stays); activity "walking" with no travel and no event; a camera move in an event but "static" as camera movement. A character who is simply absent from the end of cut list is not a problem.

3. **Speaking:** compare each transcript turn with the speaking value of its character. A turn linked to an on-screen character with speaking yes is lip-synced dialogue. A turn linked to an on-screen character with speaking no is probably an inner voice or narration, or the video model missed the mouth: say which the evidence favours. A turn whose speaker is not in frame is an off-screen voice and not a problem. A turn with an unlinked speaker label while exactly one visible character has speaking yes is a likely link worth one targeted check. A turn whose speaker is a named character with no box in this cut, while a person marked "identity: not a marked character" has speaking yes, probably means that person is that character and is missing a character box: recommend adding the box rather than treating the name as established.

4. **Emotion:** compare the first frame's emotion, the video's emotion and the meaning of the dialogue. Report only a real conflict (a cheerful face under an angry line), never a difference of wording such as neutral versus calm.

5. **Cross-modal text:** where feasible, compare transcript meaning with subtitle meaning, using each line's own times. Different languages, wording, or timestamps do not automatically constitute contradictions.

## INTERPRETATION

- The first frame decides the state at the start of the cut. The video analysis decides what changes afterwards. A difference between the first-frame state and a later state is change over time, not a contradiction.
- The two visual models word things differently and see at different resolutions. A difference in wording, level of detail, or a small nuance of gaze or expression is NOT an issue: do not report it and do not ask a person to settle it. Report a disagreement only if it would change the generated image, audio or motion, and then say which source to prefer by the rule above.
- Characters may speak while remaining physically stationary.
- Video event timestamps are estimates.
- Do not label expected model limitations as cut-specific problems unless they create a concrete production risk.
- Do not recommend reviewing all source material by default.
- When evidence is insufficient, describe precisely what remains unknown.

## OUTPUT FIELDS

Return exactly one JSON object with these five string fields:

**summary:** What the cut shows, in at most three sentences: who is on screen and where, what happens (motion and camera), and who speaks. Use the character names.

**clear:** What can be used for generation as it is. Three short parts in this order: "Still:" (the opening image), "Motion:" (camera, each character's activity, travel, motion, emotion change and end state), "Speech:" (each turn's speaker, whether they are on screen and visibly speaking). Use the character names.

**confusing:** Only material contradictions, unresolved ambiguities, or evidence gaps that would change the still, the audio or the motion. Identify the specific sources, claims, and timestamps involved. If none exist, return "None identified."

**human_review:** Only decisions that need a person or evidence the automated pipeline does not have. State the decision needed and the option the evidence favours. Do not repeat generic verification disclaimers. If none exist, return "None required."

**recommended_checks:** Concrete, targeted verification steps that would resolve an identified problem, naming the timestamp or evidence source. Do not repeat human_review. If no checks are needed, return "None required."

## FINAL QUALITY CHECK

Before returning the JSON:

- Every reported conflict is a genuine disagreement that affects generation, not a wording difference or normal change over time.
- Recommendations address issues identified in this cut.
- No generic warnings and no recommendation repeated across fields.
- The JSON contains exactly the five required string fields and is syntactically valid.

Return JSON only. No Markdown, explanations, or additional fields.'''
FIELDS = ['summary', 'clear', 'confusing', 'human_review', 'recommended_checks']


def _as_text(value, indent=0):
    if isinstance(value, dict):
        return '\n'.join('  ' * indent + str(key).replace('_', ' ') + ': ' + (_as_text(child, indent + 1).lstrip() if not isinstance(child, (dict, list)) else '\n' + _as_text(child, indent + 1)) for key, child in value.items())
    if isinstance(value, list):
        return '\n'.join('  ' * indent + '- ' + (_as_text(child, indent + 1).lstrip() if not isinstance(child, (dict, list)) else '\n' + _as_text(child, indent + 1)) for child in value) or '  ' * indent + '(none)'
    return str(value) if value is not None else 'unknown'


def subtitle_evidence(subtitles):
    """Subtitle lines overlapping the cut, in the form the reviewer is told about."""
    subtitles = subtitles if isinstance(subtitles, dict) else {}
    lines = [{key: line.get(key) for key in ('start_time', 'end_time', 'text', 'confidence', 'crosses_cut')}
             for line in subtitles.get('lines') or [] if isinstance(line, dict)]
    return {'extracted': bool(subtitles.get('extracted')), 'language': subtitles.get('language'), 'lines': lines}


def _without_ids(frame):
    """A frame description with the characters named only: database ids mean nothing to a reviewer."""
    if not isinstance(frame, dict) or not isinstance(frame.get('visible_characters'), list):
        return frame
    return {**frame, 'visible_characters': [{key: value for key, value in person.items() if key != 'character_context_id'} if isinstance(person, dict) else person
                                            for person in frame['visible_characters']]}


def build_input(results, transcripts, context_items, subtitles=None):
    result_map = {item['type']: item for item in results}
    parts = []
    video = (result_map.get('video_analysis') or {}).get('content')
    if prompt_context.is_direction(video):
        # Current analyses: the opening still and one motion direction for the cut. A last-frame
        # description left over from an older analysis is not evidence about this run.
        first = result_map.get('initial_frame')
        parts.append('First frame:\n' + (_as_text(_without_ids(prompt_context.frame_view(first['content']))) if first and first.get('content') is not None else 'Unavailable.'))
        parts.append('Video analysis:\n' + _as_text(prompt_context.video_view(video)))
    else:
        for kind, label in (('initial_frame', 'Initial frame'), ('video_analysis', 'Video analysis'), ('last_frame', 'Last frame')):
            result = result_map.get(kind)
            parts.append(label + ':\n' + (_as_text(result['content']) if result and result.get('content') is not None else 'Unavailable.'))
    names = {item['id']: item['title'] for item in context_items if item.get('kind') == 'character'}
    named = [{**segment, 'character_name': segment.get('character_name') or names.get(segment.get('character_context_id'))} for segment in transcripts]
    turns = [prompt_context.transcript_line(segment) for segment in prompt_context.slim_transcript(named)]
    return {'visual_evidence_text': '\n\n'.join(parts),
            'transcript_text': '\n'.join(turns) if turns else 'No transcript segments available.',
            'subtitles': subtitle_evidence(subtitles)}


def generate(config, evidence, progress=None, cancel_check=None, usage_out=None):
    from .inference_worker import chat_json
    result, model = chat_json(config['openrouter_model'], SYSTEM, json.dumps(evidence, ensure_ascii=False), FIELDS, progress=progress, cancel_check=cancel_check, provider=config.get('text_provider', 'openrouter'), thinking=config.get('local_thinking', False), usage_out=usage_out)
    return result, model
