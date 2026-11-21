"""The director's side of sequence direction: prompts, answer schemas and validators.

Five calls direct a sequence: ``compile`` (the dialogue is reconciled with the subtitles and reviewed by the user), ``bible`` (what the sequence is and who is in it), ``plan`` (one shot per
cut: where each boundary sits at a hard cut that falls inside speech, and each shot's intent), one ``shot`` call per planned shot, and ``review`` (continuity across the
written shots). The model decides and writes prose; ids, times, labels and spoken words are the
app's (``sequence_timeline``, ``sequence_prompt``). Each validator returns the normalised answer
or raises ``ValueError`` with sentences that are sent back to the model, or shown to a person who
edited the result by hand.
"""
import math
import re

from . import sequence_compile, sequence_evidence, sequence_prompt, sequence_timeline, still_vocabulary

FRAMING = """You are the director of a short video of at most three minutes that re-creates a sequence of an episode with AI, shot by shot. Each shot is generated on its own: first a still image of its opening frame, then a MiniMax H3 video of at most 15 seconds that animates that still in sync with the ORIGINAL audio of the episode. The original audio is kept unchanged under the whole video, so it is the spine every decision follows.

You do not see pictures and you do not hear audio. Your evidence is text written by other systems (vision models, a transcription, the user's notes); it can be incomplete or wrong. The user's own notes, character links and decisions are authoritative.

The prompts you write are suggestions for whoever generates the pictures and the video. The visual style belongs to them: never name a medium, an art style or a rendering technique, and describe a look only as "the look of the supplied picture"."""

RULES = """## RULES
- Base everything on the supplied evidence. Never invent characters, places, actions, props or dialogue.
- Emotion comes from the pictures and from the meaning of the words. No acoustic analysis was made: never describe vocal tone or delivery as heard.
- Subtitles and other on-screen text are reference only and never part of a picture.
- Write in English. Keep character names exactly as they are given, in full.
- The user's own spoken words and timings are absolute. When the input says dialogue_reviewed, the dialogue lines (their words, speakers and seconds), the voice ranges in fixed_by_user and the shot boundaries the user placed are final: never change, move, reword or contradict them, and never raise a disagreement with other evidence as a question or a problem.
- Every other source of speech or of its timing is supporting material only: the automatic transcription, subtitles, a cut summary, the motion analysis (its mouth events and speaking flags) and a neighbouring shot's lines. It can show whether a mouth is seen moving. It never supplies words, never moves a line, and where it differs from the dialogue, the dialogue is right. The user places the cuts where they want them, so a line seldom runs across one: when supporting material suggests it does and the dialogue lines do not say so, trust the dialogue."""

COMPILE_SYSTEM = FRAMING + """

Before anything is directed you COMPILE THE DIALOGUE of the sequence. The transcript was produced by speech recognition and is often wrong: wrong kanji for the spoken word, the wrong speaker, one sentence cut into several lines. The same speech was also read from the burned-in subtitles of the picture, which is a second, independent source. You reconcile them. A person reviews your result afterwards, so mark what you are unsure of instead of guessing.

## WHAT YOU RECEIVE
One JSON object for one window of the sequence (times are seconds on the sequence clock):
- window: which part of the sequence this is, spoken_language (the original audio) and subtitle_language (the language the subtitles are written in, possibly a translation).
- lines: the recognised speech as `L<n> start-end VOICE LABEL / Character: words`. A character after the slash was linked by the user and is certain. A line with only a voice label has no known speaker yet.
- subtitles: the subtitle lines read from the picture as `U<n> start-end: words`. They are written by people, so their spelling and kanji are more reliable than the recognised text, but they may be shortened, may hold two speakers in one line, and are shown a little before or after the speech.
- pairs: for each line the subtitle lines that overlap it, with the overlap in seconds and a text similarity from 0 to 1.
- hints: consecutive lines where one utterance may have been cut in two, with the reasons the app found.
- cuts: per source cut its time span, the named characters on screen and the named characters whose mouth visibly moves.
- characters: the exact names you may use as speaker.
- voices: per voice label the characters the user linked, and who visibly spoke while it was heard.
- previous: the last compiled lines of the window before this one, and voices_decided so far. A voice label keeps the character decided for it.

## WHAT YOU DECIDE
Return the lines of this window as an ordered list of groups. Every input line belongs to exactly one group, in order; a group is one or more CONSECUTIVE lines.
1. Grouping: inspect syntax, meaning and hard-cut times even when a brief pause separates the fragments. Use next_window only to judge continues_next. Join lines into one group when they are one utterance that was cut in half (a sentence that continues, a word split across the boundary, one subtitle covering both), including across a hard cut. Keep complete sentences separate, and never join lines of different speakers. Mark continues_next false for a complete phrase, even if its speaker keeps talking right away.
2. Text: the final words of the group, written in the language of the original audio (the language the lines were spoken and recognised in, `spoken_language`), never in the language of the subtitles. For translated subtitles, compare what each clause means against the recognised words before settling the text. Actively look for an ASR homophone or broken phrase that makes poor sense in context. If a nearby translation identifies its intended meaning and there is a plausible sound-alike in the spoken language, repair the wrong token there. For example, if Japanese recognition says 講演を散歩した and the English subtitle says "I walked through the park", write 公園を散歩した. For English speech, “we took the fairy across the bay” becomes “we took the ferry across the bay” when the translated subtitle refers to a boat and the scene is a crossing. A translation conveys meaning even when the text similarity is zero. Repair a clear contextual spelling or word-choice error even when it sounds like the recognition and the rest of the sentence is already correct. If the distinction is ambiguous, retain the audio-language recognition and flag the uncertainty. Preserve its register, names, hesitations and code-switching. Never replace a whole sentence with a back-translation or copy translated words. Use the subtitle spelling only when a paired subtitle is written in the spoken language and clearly says the same words (a similarity above about 0.5, or the same sounds written differently); otherwise keep the recognised text, correcting only what the meaning of the subtitle shows to be a recognition error, in the spoken language. Never add words nobody said, never copy words that belong to the other speaker of a shared subtitle, and keep the spoken language. Join the pieces of a split utterance into one natural line. Write no speaker names, no brackets and no stage directions.
3. Speaker: the exact character name, "Narrator" for a narration voice, or null when nothing supports a choice. A line linked by the user keeps its character. For others weigh the voice label's other lines, who visibly speaks, who is alone on screen, and who would say these words to whom. A wrong speaker is worse than none.
4. Subtitles used: the ids of nearby subtitles you relied on for wording or meaning.
5. confidence: high when the text and speaker are supported by two sources, medium when by one, low when you guessed or the sources disagree. note: one short sentence when you changed the words or when sources disagree; otherwise empty.
6. continues_next: true when the group ends mid-utterance because the next group (in this or the next window) continues it.
7. voices: one entry per voice label that appears in this window, with the character it belongs to or null, its kind (character, narrator, unknown), confidence and the reason.
8. open_questions: only what a person must decide because the evidence cannot.
Whether a speaker is seen speaking is not decided here: each shot decides it later for its own picture.

## RULES
- Base everything on the supplied evidence. Never invent dialogue, characters or speakers.
- The words of "text", "note" and "reason" stay in the spoken language of the lines; "note" and "reason" may be English. Character names are copied exactly as given.
- Never use acoustic claims: you did not hear the audio.


## OUTPUT
Return exactly one JSON object:
{"lines": [{"ids": ["L3", "L4"], "text": "final words", "speaker": "exact name, Narrator or null",
            "subtitle_ids": ["U2"], "confidence": "high | medium | low", "note": "", "continues_next": false}],
 "voices": [{"speaker_label": "SPEAKER_01", "character": "exact name or null", "kind": "character | narrator | unknown", "confidence": "high | medium | low", "reason": "..."}],
 "open_questions": ["..."]}
Return JSON only."""


TIMING_SYSTEM = """Choose where each shot starts on the unchanged original audio. The reconciled dialogue identifies hard cuts that fall inside an unfinished phrase. Use the supplied options to place each such boundary just before or after the whole phrase. Prefer the picture that shows its speaker and preserve important reactions and actions. When both pictures fit equally well, prefer the smaller shift. Consider adjacent boundaries together: every cut remains in source order, each shot needs enough of its own picture and must fit the generator's time limit. Keep the hard cut only when no shifted option fits. User-placed boundaries are final.
Answer every listed decision once. Format: {"decisions":[{"id":"K1","choice":"before_line | after_line | keep_cut","reason":"the reason for the placement"}]}. Return JSON only."""

BIBLE_SYSTEM = FRAMING + """

Before any shot is planned you write the SEQUENCE BIBLE: the shared understanding that keeps every shot consistent.

## WHAT YOU RECEIVE
One JSON object:
- project, episode and director_instruction (the user's intent for this sequence; follow it when present).
- sequence: its length in seconds and its first and last cut number.
- characters: the project's character records (exact names, notes). For a character who appears in these cuts, seen_in_cuts lists where, and appearance_in_first_frames quotes how the vision model described that character in the opening frames. scenario_notes and context_notes: the user's notes on places and story.
- cuts: one digest per source cut, in order. t is its start and end in seconds from the start of the sequence; shot is the size and angle of its opening frame; camera its movement; on_screen the named characters the user marked; unnamed_people true when other people are visible; speaking the named characters whose mouth visibly moves; action what happens; setting, lighting, mood and props describe the opening frame. "analysis": "missing" means the cut was not analysed.
- transcript: every spoken line as `L<n> start-end VOICE LABEL / Character: words`; with dialogue_reviewed these are the user's final words and seconds. A character after the slash was linked by the user and is certain. A line with only a voice label (such as SPEAKER_03) has no known speaker yet. A line shown as "Unknown speaker" has no voice label at all: such lines may come from different people, so they get no entry in voices.
- voices: per voice label, how much it says, which characters the user linked it to (linked_by_user), and for its unlinked lines who was visibly speaking meanwhile and who was the only named person on screen. These are hints, not proof.
- pauses: quiet stretches of the audio, as start and end seconds.

## YOUR DECISIONS
1. Story (logline, synopsis, arc): what this sequence is about and how tension and emotion develop from the first cut to the last.
2. Scenes: group consecutive cuts that share one place and time. Every cut belongs to exactly one scene and scenes follow each other without gaps. Name the place concretely, with its time of day and light. If a scenario note describes that place, put its exact title in "scenario"; otherwise null.
3. Characters: every named character who is on screen in these cuts (seen_in_cuts), with the exact name of the character record. anchors are 3 to 6 visible features that stay constant in this sequence and identify the character at any angle and distance (build, hair, face, main garment and its colours); small accessories such as hairpins, clips, jewellery or badges are not anchors, because they show in some frames and not in others. outfit is what the character wears here. Take both from appearance_in_first_frames, which is what the frames show; the record notes describe the series and may show another outfit. When the frames disagree about a garment or a colour, keep what most of them say and name the disagreement in open_questions. arc is what the character wants or feels across the sequence, in one sentence.
4. Voices: when the input says voices_are_final, copy them exactly and decide nothing. Otherwise decide who owns each voice label. A label the user linked to one character belongs to that character. For an unlinked label weigh who visibly speaks while it is heard, who is alone on screen, what the words mean and who would say them to whom, and whether another label already belongs to that character. kind is character, narrator (narration, a title voice) or unknown. confidence is high only when two independent hints agree, otherwise medium or low. When nothing supports a choice, use kind unknown and character null: a wrong owner is worse than none.
5. spoken_language: the language of the spoken lines, as one English word (for example Japanese).
6. Pacing: how the shots should breathe with the audio: where the sequence is quick, where it rests, where a held look matters.
7. Continuity: concrete rules every shot must respect, limited to what changes the animation (outfits, props carried or used from one cut to the next, who stands where, time of day, weather). Make no rule about a small detail that appears or disappears with the angle, the distance or the crop.
8. open_questions: only what a person must decide because the evidence cannot.

""" + RULES + """

## OUTPUT
Return exactly one JSON object:
{"logline": "one sentence",
 "synopsis": "at most 120 words",
 "spoken_language": "Japanese",
 "scenes": [{"id": "S1", "from_cut": 12, "to_cut": 15, "location": "...", "time_of_day": "...", "lighting": "...", "mood": "...", "scenario": "exact scenario note title or null"}],
 "characters": [{"name": "exact record name", "role": "who this is in the sequence", "anchors": "3 to 6 visible features", "outfit": "...", "arc": "..."}],
 "voices": [{"speaker_label": "SPEAKER_01", "character": "exact record name or null", "kind": "character | narrator | unknown", "confidence": "high | medium | low", "reason": "the hints that decided it"}],
 "arc": "...", "pacing": "...",
 "continuity": ["one rule per item"],
 "open_questions": ["..."]}
Return JSON only."""

PLAN_SYSTEM = FRAMING + """

The bible is written. Now you plan the SHOTS. Every source cut becomes exactly one shot, in order: shots are never merged, and only a cut longer than 15 seconds is divided by the app into consecutive parts of one take. Each shot is generated on its own, from the opening still of its own cut, with the original audio of exactly its own time span underneath it.

## WHAT YOU DECIDE
1. Where each shot boundary sits at a hard cut that falls inside speech. On the hard cut, the picture changes exactly where the source edit changes it; if a phrase is being spoken at that moment, the phrase is divided between two separately generated shots. To keep the phrase whole, the boundary can move into the silence around it instead:
   - before_line: the boundary moves to the silence just before the phrase begins. The later shot starts early and carries the whole phrase in its own picture; the last moment of the earlier cut's picture is not shown.
   - after_line: the boundary moves to the silence just after the phrase ends. The earlier shot runs on and carries the whole phrase; the first moment of the later cut's picture is not shown.
   - keep_cut: the boundary stays on the hard cut and the phrase is divided at that exact moment.
   Keep phrases whole. Choose keep_cut only when the scene needs the edit to land inside the phrase: someone interrupts the speaker, an action or a reaction lands on a word, a reply cuts in, or the picture change is a beat the timing of the scene depends on. Give the reason, naming that moment. Between before_line and after_line choose the side that keeps the phrase with the picture of its speaker and does not hide an action that matters (each option says what its hidden picture shows).
2. intent: for every cut, one sentence on what its shot must achieve in the sequence.
3. notes: one or two sentences on the rhythm of the sequence.

## WHAT YOU RECEIVE
One JSON object: director_instruction, the bible, sequence_seconds, the cuts (t in sequence seconds, shot, camera, setting, on_screen, speaking = the named characters whose mouth visibly moves, action, and parts when the app divides a cut longer than one shot), the transcript (`L<n> start-end speaker: words`, times in sequence seconds; with dialogue_reviewed they are the user's final words and seconds) and decisions.
decisions lists every hard cut that falls inside speech and is still yours to decide: its id (K<n>), the time of the hard cut, the cut before and after it, the speech it falls into with the words heard before and after the cut, and options, which describes what each possible choice does. A choice missing from options is not possible there; not_possible says why. Other boundaries stay at the hard cut or use dialogue_timing, the automatic positions chosen when dialogue was compiled. Write intents around those positions; do not include them in decisions.
fixed_by_user lists the hard cuts whose shot boundary the user placed by hand: where it sits, what each shot hears on either side of it, the picture it hides and the frame the later shot opens on. These boundaries are final, even where they divide a phrase: never answer them in decisions, and write the intents of the shots beside them around those edges.

""" + RULES + """

## OUTPUT
Return exactly one JSON object:
{"decisions": [{"id": "K<n>", "choice": "keep_cut | before_line | after_line", "reason": "one sentence"}],
 "cuts": [{"cut": 12, "intent": "what this shot must achieve, one sentence"}],
 "notes": "one or two sentences on the rhythm"}
Answer every decision exactly once and give every cut one entry, in order. With no decisions, return an empty "decisions" list. Never answer a hard cut listed in fixed_by_user.
Return JSON only."""

SHOT_SYSTEM = FRAMING + """

The bible and the shot plan are written. Now you direct ONE shot. Your answer is structured direction, and the app turns it into the final MiniMax H3 prompt: it writes the section headers, the reference labels, the timecodes, the camera sentence, the spoken words and the whole sound section itself. The H3 prompt animates a still that already exists, so it is about what happens: who moves, how and when. Appearance in detail belongs to the still and the character references. Therefore write names and never reference labels, put no timecodes inside your sentences, and never quote dialogue.

## WHAT YOU RECEIVE
One JSON object:
- sequence: synopsis, the spoken language, the arc and the continuity rules of the bible. director_instruction is the user's intent for the sequence. instruction_for_this_shot is the user's newest correction for this shot and earlier_instructions_for_this_shot (oldest first) are the corrections the user gave in earlier rewrites of it: every one of them stays in force, and where two differ the later one wins. review_suggestion, when present, is a proposal of the automatic review: follow it unless it contradicts the user's instructions or the dialogue.
- scene and scenario_reference (a picture of the place that the generator receives, when there is one).
- shot: its number, its length in seconds, its cut (and part, when the app divides a cut longer than one take), its intent from the plan, initial_frame (whether the opening still is the cut's first frame or a later moment of it), and opens and closes: how the shot meets its neighbours (see TIMING AT THE EDGES).
- previous_shot and next_shot: what surrounds this shot. ends_with is how the analysis says the previous shot's characters end, opens_with is how the opening frame of the next shot shows them, and hands_over is the state the previous shot was written to end in.
- characters: the named characters that appear in the covered cuts, with the bible's anchors and outfit and the user's notes.
- cuts: the evidence for the shot's own cut. plays_at is when that cut's picture and audio run on THIS shot's clock. first_frame describes the cut's opening still; each character in it may carry pose_detail: state (standing, sitting ...), torso and head facing as a view of the camera with the screen side it turns toward, eyes, looking_at (who or what the character looks at) and check (a disagreement the app found inside those facts). motion is the motion direction: summary, camera, per character activity, travel, motion, gaze (where the eyes and head go, and when), objects (what the character touches, holds or moves), speaking (yes, no, unclear), emotion and end, state_correction when the clip shows a different state than the still read, plus events whose seconds are already on this shot's clock, each with a kind (body, gaze, mouth, expression, object, travel, camera). cut_summary is a written brief of the cut (its seconds count from the start of the cut, not of this shot, and it holds no spoken words) and user_decisions are the user's answers to open review points. A neighbouring cut appears only when this shot borrows some of its audio to keep a phrase whole; its picture is not shown.
- Each own cut can also contain keyframes (states and visible changes) and performance_beats (staged acting and delivery hints). Their times are already on this shot's clock. States immediately before zero or after the end are reference poses, not new events. Still camera is the Cut summary's camera and finish. The app adds it verbatim to the still, never to H3. Use it to describe the depth of field, textures, palette, lighting and atmosphere of the frozen image.
- dialogue: the lines heard during this shot, with id, t (start and end on this shot's clock), the speaker and how the speaker is known. With dialogue_reviewed these words, speakers and seconds were fixed by the user and are final. subtitles_for_meaning_only appears only when the dialogue was not reviewed; it may help you understand the words, never show them.
- speakers: one entry per speaker heard in this shot: the lines and seconds they speak, and per covered cut whether they are on screen there, whether their mouth visibly moves (yes, no, unclear) and whether that cut's picture is this shot's picture. A neighbouring cut whose audio this shot borrows is marked not shown: this shot's own picture goes on over those seconds. fixed_by_user lists time ranges in which the user decided how the speaker is heard.

## PRECEDENCE
1. What the user decided, which is final: dialogue (its words, speakers and seconds), fixed_by_user, the edges decided by the user, user_decisions, earlier_instructions_for_this_shot and instruction_for_this_shot.
2. The opening frame decides the initial composition and state. For a shifted start, use the nearest timed keyframe states and changes to infer what is already in progress before second 0. Keyframes decide physical changes and their times. The current performance beats supply acting intent, face, gaze, body and delivery within those changes. If they disagree, use the observed keyframes. The Cut summary supplies acting notes and constraints. Older cuts use motion direction. Use neighbouring shots for continuity and review_suggestion as advice subordinate to the source and user decisions.
Speech and its timing come only from the dialogue. The cut_summary and the motion (mouth events, speaking) tell you whether a mouth is seen moving and what the picture does, never what is said or when it is said: when one of them places speech at other seconds or in other words than the dialogue, follow the dialogue and say nothing about the difference. The first frame decides how the shot starts, the motion decides what changes over time, and what the take shows decides whether lips can be seen moving. When pose_detail and a free-text field disagree about facing, gaze or state, pose_detail wins; a character's state_correction means the video's activity is what the character does from the first second.

## WHAT YOU WRITE
- subjects: every character or person who is visible at any point of the shot, the most important first, at most 5. name is the exact name from "characters"; for someone without a record use a short visual label such as "the shop clerk", never a guessed name. anchors: 3 to 6 large visible features that identify the character at any angle and distance (build, hair, main garment and colour) and agree with the bible and the first frame; small accessories (hairpins, clips, jewellery, badges, buttons) go in still.subjects, not here. keep: what must stay identical to that character's reference, as one clause.
- picture_line: who is where in the opening still, in one sentence, with only what is visible in it. keep_picture: what the video must preserve from the opening still (composition, palette, lighting), as one clause.
- still: the opening frame as ONE frozen moment for an image generator. shot: size, angle, camera height and where the frame crops the subject. placement: for each visible character, in words, where they are in the frame (left, centre or right, foreground, midground or background, relative to the others and to objects), their body state, which way the torso faces and which way the head points as a view of the camera with the image side it turns toward, where the eyes look and at whom or what. Take these from pose_detail and the user's answers, never from your own reading of the free text, and say them in words from the KEYWORDS below, never as degrees (an image generator does not understand degrees); pose_detail already gives torso, head and eyes in those words. When state_correction says the clip shows motion from the first second, place the character in that state caught mid-motion. subjects: for each visible character the expression, hands and visible outfit, with the small accessories and details the first frame shows (hairpins, jewellery, badges); not placement. layout: foreground, midground and background and how they relate. setting. lighting: source, direction, softness, time of day, palette, highlights and shadows. Write 170 to 350 words across the still's fields when there is evidence to support them, as concrete connected visual prose for Qwen-Image-2.1. Start from the subject and frozen pose, establish the setting and spatial layers, then describe framing, lighting, atmosphere and photographic detail (fabric, skin, reflections or surfaces where visible). Make screen direction and depth unambiguous. No lettering.
- summary: one sentence with what happens in the shot and how the camera behaves.
- shot_line: shot size, angle and who is where at the start, in one or two sentences. framing_lock: one short sentence that locks the crop you gave in still.shot, naming the same upper and lower edge of the frame and ending with "never wider" (or "never closer" for a wide shot); it must agree with still.shot. light: one sentence on the light.
- camera: move is ONE of static, push_in, pull_out, pan_left, pan_right, truck_left, truck_right, tilt_up, tilt_down, pedestal_up, pedestal_down, arc, tracking, shake. If the motion evidence says static, use static. amplitude is small, medium or large; speed is slow, medium or fast. detail is one optional sentence on where the move ends or what it reveals; leave it empty for a static shot.
- beats: use the supplied performance_beats and keyframes on the shot clock. Follow the distinct acting turns and their seconds within the visible shot. Give each important beat a playable start, change and arrival. For the people who matter to it, direct weight shifts, posture, hands and prop contact, expression changes (eyes, brows, cheeks or jaw), where the eyes settle, mouth shape, and how an action lands. Use intent and mood to choose intensity, giving quiet reactions room. In clearly comic or kinetic moments you may heighten timing, anticipation, recoil or follow-through while preserving the evidenced action and the shot's fixed timing. Keep beats contiguous; end the last beat at the shot duration. Combine nearby beats only when they are one acting change. The app inserts each spoken line once. In the acting beats write only the visible mouth action, for example "her lips part in sync", without quoting the line or paraphrasing its content. Keep sound and camera directions outside acting beats. Use mouth motion only during the confirmed dialogue of a visible lip-synced speaker.
- voices: how each speaker of "speakers" is heard in THIS shot. Give every speaker an entry from second 0; add a later entry for the same speaker only when that changes during the take (a camera move or an action brings them into view, turns them away or takes them out of frame), with from = the second it changes. delivery is one of:
  - on_screen: the speaker is one of your subjects, in the picture with the mouth visible, and visibly speaks; the mouth moves in sync with their words.
  - off_screen: the speaker is not in the picture at that time. Also use it for a voice nobody confirmed.
  - inner_voice: the speaker is in the picture but the mouth does not move for the words: a thought, or the face is turned away from the camera or hidden.
  - narrative: the speaker may or may not be in the picture but is too far from the camera, or too small, for lip sync; the words simply play over the picture. Use it also for a narrator.
  subject is the exact subject name who carries the voice for on_screen and inner_voice (for a named speaker, that speaker), may name the distant subject for narrative, and is null for off_screen. Decide from this shot's own picture: a speaker who is visible only in a neighbouring cut whose audio this shot borrows is off screen here, and a speaker who is seen speaking in this shot's picture and finishes a phrase over borrowed seconds stays on_screen until the last word. Being the only person on screen is not proof of speaking; for an unconfirmed voice choose off_screen and add an open question. Ranges in fixed_by_user were decided by the user: repeat them exactly and write the picture and the acting around them.
- scenario_anchors and scenario_keep: only when scenario_reference is present: 3 to 5 visible features of the place, and what must stay as in its reference.
- continuity: from_previous is what must match the end of the previous shot; to_next is the state this shot hands over (positions, poses, props, gaze). One sentence each. When the evidence of this shot contradicts a neighbour (someone sits here and stands in the next opening frame, an outfit differs), follow this shot's own evidence and name the contradiction in open_questions instead of hiding it.
- open_questions: only decisions the evidence cannot make.

## CONTINUITY, ANIMATION AND AUDIO
- Continuity: open on the state previous_shot.hands_over (or ends_with) and close on the state the next shot opens with. Carry positions, held props, eyelines, clothing state and time of day across the boundary; when a line of dialogue continues from the previous shot, the speaker's body and gaze continue too.
- Animation direction: build the shot on the clock first. Anchor every beat to the seconds of the motion events and their kind, in order, so that the picture stays in sync with the original audio; a gaze shift, a mouth movement and a handled object each land at their own second. Position precisely: say where each character starts and ends in the frame (zone, depth, relative to the others and to objects) and which way the torso and head face when a beat begins and ends. Act precisely: name which body part leads, in what order, in which direction, how far and how fast; give a gaze shift its target (who or what, from where to where) and its second, say when the mouth moves (only for an on_screen speaker) and when it stops, and give small acting (a breath, a hand tightening, a nod) its own beat only when the evidence places it. Name every object that is touched, held, moved, pointed at or looked at, and what happens to it, from the video's objects and the first frame's props. Give the camera a start framing and an end framing when it moves. Hold still where the source holds still; do not add motion the evidence does not show.
- Audio treatment: the original audio is the spine. A mouth moves only while its speaker is on_screen and their words are heard, which is during the seconds t of that speaker's dialogue lines; a mouth beat starts and ends there, also where a mouth event or the cut summary puts the speech elsewhere. An off_screen voice needs a picture without that speaker; an inner_voice speaker keeps the lips still (or the mouth hidden) while the words play; a narrative speaker stays far enough away that no lip movement is expected. Beats must agree with the voices: never describe lips moving in speech for someone who is not on_screen at that second.
- Sound: the app writes the sound section itself. It tells the generator to follow the original audio exactly, with whatever background sound, music or effects it holds. You cannot hear that audio, so never describe or assume a background sound, a room tone, the ambience of a place or music anywhere in your answer. Time the action to the dialogue and the motion evidence.

## TIMING AT THE EDGES
The edges of the shot are exact and lie on the original audio, so the acting must land on them. opens and closes give their seconds on this shot's clock.
- When speech is divided at an edge (split), its words stop or resume at that exact second. When the director gave a reason for dividing it there (an interruption, an action that cuts the speaker off, a reaction that lands on a word), the moment that causes it gets its own beat ending at that second, or starting at 0 in the shot that opens on it.
- When the shot carries a phrase past the end of its cut, it keeps its own framing until the phrase ends and the speaker's acting continues to the last word; a speaker who is on_screen there stays on_screen until the end. When it starts before its cut, hold its opening composition and gross body position until its own motion begins. Let the visible speaker lip-sync and play subtle facial reactions during the borrowed seconds when their dialogue and delivery call for it.
- When the shot starts a moment into its cut, the action before that moment has already happened: open on the state it left.
- A part of a long cut continues the take of the part before it: same framing, same light, the motion goes on.
- An edge the user placed (decided_by the user) is exact and final, even when it divides a phrase: the acting meets it like any other edge, and the shot never asks to move it.

## RULES FOR H3
- Never write the word "cut", not even "no cut": H3 reads it as an edit. Say "shot" or "take".
- No lettering anywhere: write "blank" or "unmarked". Subtitles and on-screen text never appear.
- Never name a medium, art style or rendering technique; "keep_picture" and "framing_lock" refer to the supplied pictures, not to a style.
- When the initial frame is a later moment of its cut, derive the opening state from first_frame plus the motion up to that moment, and add an open question if that state is uncertain.
- Write 180 to 480 words across shot_line, framing_lock, light and the beats together. Every sentence must be concrete and supported; no filler.

""" + RULES + "\n\n" + still_vocabulary.shot_block() + """

## OUTPUT
Return exactly one JSON object:
{"subjects": [{"name": "...", "anchors": "...", "keep": "..."}],
 "picture_line": "...", "keep_picture": "...",
 "still": {"shot": "...", "placement": "...", "subjects": "...", "layout": "...", "setting": "...", "lighting": "..."},
 "summary": "...", "shot_line": "...", "framing_lock": "...", "light": "...",
 "camera": {"move": "static", "amplitude": "small", "speed": "slow", "detail": ""},
 "beats": [{"end": 2.4, "action": "..."}],
 "voices": [{"speaker": "exact speaker from speakers", "from": 0, "delivery": "on_screen", "subject": "exact subject name or null"}],
 "scenario_anchors": "", "scenario_keep": "",
 "continuity": {"from_previous": "...", "to_next": "..."},
 "open_questions": ["..."]}
Return JSON only."""

REVIEW_SYSTEM = FRAMING + """

Every shot of the sequence is written. Now you REVIEW the whole sequence before anything is generated, as the director who has to answer for it. This is optional advice. Default to an empty list. Report only an evidenced contradiction inside a shot, or across a continuous take, that would visibly spoil the acting or lip sync, and give an actionable correction. Cuts between shots are jumps by design; a difference across a cut is not a contradiction.

## WHAT YOU RECEIVE
One JSON object: the bible (story, scenes, characters with anchors and outfits, continuity rules), the voices it decided, and shots in order. Each shot has t (start and end in sequence seconds), its cut, scene, intent, opens and closes (how it meets its neighbours: on the hard cut, or moved before or after a phrase so that the phrase stays whole, or placed by the user, with the speech divided there and the director's reason), opening_still, opens_with and ends_with (how the analysis says each character is at the start and at the end of the shot: state, torso and head facing, who or what they look at; present only when the shot opens on its cut's first frame or reaches the end of its cut), subjects, summary, camera, beats ("end second: action", seconds on the shot's own clock), voices (per speaker how they are heard and from which second: on_screen = lip-synced, off_screen = not in the picture, inner_voice = visible with the mouth still, narrative = too far for lip sync; "(user)" marks a range the user decided), lines, user_instructions (what the user asked for when rewriting that shot; present only then) and continuity (from_previous, to_next). A shot without "written" has no direction yet.
lines lists every line spoken in the shot's original audio as "line id, start-end seconds on the shot's clock, speaker: first words". "none" means nobody speaks anywhere in that shot. The lines are the only evidence of speech you have: the original audio plays unchanged under every shot and you cannot hear it, so never assume a voice, a sound or music it does not list.

## WHAT TO CHECK
1. Each shot is one cut of the source edit, and a cut is a jump: the next shot may show another angle, distance, pose, gaze, expression or hand position, and that is normal editing, never a continuity error. Compare two shots only when they are one continuous take: parts of one long cut, or a boundary where the same action or phrase visibly runs on across it. Then report only a contradiction of what the take itself establishes: a person who appears or vanishes mid-action, a state that jumps without the action that causes it (sitting, then standing, with no beat between), the wrong person moving, or the light or time of day changing. Between two separate cuts report nothing about pose, hands, gaze, facing, expression, position or framing, even when ends_with and opens_with differ.
2. Voices, only at the seconds a line of that shot covers: on_screen or inner_voice given to someone who is not a subject of that shot; one voice on_screen for two different subjects; a speaker on_screen while the beats keep the face hidden, turned away or far away.
3. Timing at the boundaries: where speech or an action continues across a boundary, the end of one shot and the start of the next agree on who speaks, where the words stop and resume, and the action that interrupts and when it lands; a part of a long cut continues the part before it; a camera move that fights the intent.
4. Bible: the characters' identity (build, hair, main garment and colours), the scene and those continuity rules that concern the points above.

## WHAT IS NOT AN ISSUE
- A voice, speech, lip sync or sound where the shot's lines show none: a shot whose lines are "none", and every second outside its lines, has no voice to add, move or sync. Never suggest adding a voice, a speaker, dialogue or lip sync there.
- Any difference between the end of one shot and the start of the next when they are separate cuts: pose, hand or arm position, posture, facing, gaze, expression, where a character stands, what is in the frame. The source edit made that jump; the shot follows its own cut's evidence.
- A speaker heard differently in different shots, or differently inside one shot when the camera moves (on screen in one, off screen, narrative or an inner voice in another): each shot decides it from its own framing.
- A voice range marked "(user)": the user decided it.
- Anything a shot's user_instructions ask for, and the words and seconds of the lines listed: the user decided them, and your suggestions never go against them.
- Where an edge "placed by user" sits, even when it divides a phrase: the user placed it. Report only what the shots on either side get wrong around it.
- A difference of shade, tone or wording of a colour or material (brown-red against grey-red, dark against navy, blonde against light brown), a difference between how two shots describe the same garment, hair or eyes, and any appearance detail that does not change who the character is: such a difference is never an issue unless it is a different garment or a different person. The same holds for light described in different words.
- A small detail that appears or disappears with the angle, the distance or the crop: hairpins, hair clips, ribbons, earrings, jewellery, buttons, badges, pockets, stitching, small patterns. Such details belong to the opening still and the character references, not to the animation. Report one only when the action of the shot depends on it (a character pulls out the hairpin, shows the badge).
- A shot that leaves out a feature another shot or the bible mentions, and any difference of wording or of detail between shots.
- Anything you are not sure would change the generated picture, its motion or its lip sync. When in doubt, leave it out. An empty issues list is the right answer for a sequence that holds together.

## HOW TO WRITE AN ISSUE
- shot: the shot that must be rewritten to repair it. When two shots disagree, choose the one that departs from its own evidence or from the bible, and name the other one in problem.
- severity: "fix" when the shot as written would generate a visibly wrong video (a person appears or vanishes between consecutive shots of one scene, a pose or a position jumps, the wrong person moves the lips for a line, the light or the time of day changes inside a scene). "check" only when the evidence cannot decide and a person must choose; never use "check" for a minor point, leave it out instead.
- problem: one sentence naming the shots, the seconds and what disagrees.
- suggestion: one instruction that the writer of that shot can follow on its own, without seeing this review: the part to change (a beat and its second, a speaker's voice, the subjects, the camera, the opening or the closing state) and what to write instead, with names. Never "check", "verify" or "consider", and never ask to add a detail, a sound or a voice.

""" + RULES + """

## OUTPUT
Return exactly one JSON object:
{"verdict": "two or three sentences on whether the sequence holds together",
 "issues": [{"shot": 4, "severity": "fix | check", "problem": "what disagrees, naming the shots and seconds",
             "suggestion": "the change the writer of that shot makes"}]}
Return JSON only."""

# max_tokens includes the model's reasoning; temperature is lower where the answer is a decision and higher where it is prose.
STEPS = {
    'compile': {'system': COMPILE_SYSTEM, 'max_tokens': 6144, 'temperature': 0.2},
    'timing': {'system': TIMING_SYSTEM, 'max_tokens': 6144, 'temperature': 0.15},
    'bible': {'system': BIBLE_SYSTEM, 'max_tokens': 6144, 'temperature': 0.3},
    'plan': {'system': PLAN_SYSTEM, 'max_tokens': 8192, 'temperature': 0.3},
    'shot': {'system': SHOT_SYSTEM, 'max_tokens': 6144, 'temperature': 0.6},
    'review': {'system': REVIEW_SYSTEM, 'max_tokens': 6144, 'temperature': 0.2},
}
STRING = {'type': 'string'}
STILL_KEYS = ('shot', 'placement', 'subjects', 'layout', 'setting', 'lighting')  # shots written before 2026-11-16 have no placement
LANGUAGE = re.compile(r'^[A-Za-z][A-Za-z -]{1,30}$')
BANNED = (
    (re.compile(r'\bcuts?\b', re.IGNORECASE), 'contains the word "cut"; say "shot" or "take"'),
    (re.compile(r'<\s*(?:Subject|Picture|Audio|Video)\b', re.IGNORECASE), 'contains a reference label; write names, the app adds the labels'),
    (re.compile(r'</?d>'), 'quotes dialogue; describe the acting only, the app adds the spoken words'),
    (re.compile(r'\b\d{2}:\d{2}(?:\.\d+)?\b'), 'contains a timecode; the app writes the times'),
)


def _object(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(required if required is not None else properties), 'additionalProperties': False}


def _format(name, schema):
    return {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True, 'schema': schema}}


def _str(value):
    return re.sub(r'\s+', ' ', value).strip() if isinstance(value, str) else ''


def _strings(value, limit):
    return [_str(item) for item in value if _str(item)][:limit] if isinstance(value, list) else []


def _fail(errors):
    raise ValueError(' '.join(f'({number}) {error}' if len(errors) > 1 else error for number, error in enumerate(errors, 1)))


def _canonical(name, names):
    """The record name a model meant, matched without regard to case."""
    wanted = _str(name).casefold()
    return next((item for item in names if item.casefold() == wanted), None)


# --------------------------------------------------------------------- compile

def compile_format(line_ids, subtitle_ids, names):
    """json_schema of the compile answer: ids are the supplied line and subtitle ids, speakers the roster."""
    speaker = {'anyOf': [{'type': 'string', 'enum': [*names, sequence_compile.NARRATOR]}, {'type': 'null'}]}
    return _format('dialogue_compilation', _object({
        'lines': {'type': 'array', 'items': _object({
            'ids': {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'enum': list(line_ids)}}, 'text': STRING, 'speaker': speaker,
            'subtitle_ids': {'type': 'array', 'items': {'type': 'string', 'enum': list(subtitle_ids)} if subtitle_ids else STRING},
            'confidence': {'type': 'string', 'enum': list(sequence_compile.CONFIDENCE)}, 'note': STRING, 'continues_next': {'type': 'boolean'}})},
        'voices': {'type': 'array', 'items': _object({'speaker_label': STRING, 'character': {'type': ['string', 'null']}, 'kind': {'type': 'string', 'enum': ['character', 'narrator', 'unknown']},
                                                      'confidence': {'type': 'string', 'enum': list(sequence_compile.CONFIDENCE)}, 'reason': STRING})},
        'open_questions': {'type': 'array', 'items': STRING}}))


def validate_compile(output, line_ids, subtitle_ids, names, source=()):
    """Check a window's answer: every input line once and in order inside consecutive groups, real speakers and subtitle ids."""
    if not isinstance(output, dict) or not isinstance(output.get('lines'), list):
        _fail(['The answer must be one JSON object with a "lines" list'])
    errors, groups, position = [], [], 0
    by_id = {line['id']: line for line in source}
    for number, group in enumerate(output['lines'], 1):
        if not isinstance(group, dict) or not isinstance(group.get('ids'), list) or not group['ids']:
            errors.append(f'group {number} needs "ids"')
            continue
        ids = [_str(item) for item in group['ids']]
        expected = line_ids[position:position + len(ids)]
        if ids != expected:
            errors.append(f'group {number} lists {", ".join(ids)} but the next lines in order are {", ".join(line_ids[position:position + max(1, len(ids))]) or "none"}; every line belongs to exactly one group, in order, and a group holds consecutive lines')
            break
        position += len(ids)
        text = _str(group.get('text'))
        if not text:
            errors.append(f'group {number} has no text')
        originals = [by_id[key] for key in ids if key in by_id]
        linked = {line['character_name'] for line in originals if line.get('character_name')}
        labels = {line['speaker_label'] for line in originals if line.get('speaker_label')}
        if len(linked) > 1 or (len(labels) > 1 and not all(line.get('character_name') in linked and line.get('character_name') for line in originals)):
            errors.append(f'group {number} merges different speakers; keep these lines separate')
        if originals:
            original_script = sequence_compile.scripts(' '.join(line['text'] for line in originals))
            output_script = sequence_compile.scripts(text)
            if original_script and original_script.isdisjoint(output_script):
                errors.append(f'group {number} must use the script of the original spoken words')
        speaker = _str(group.get('speaker')) or None
        if speaker and speaker.lower() == 'null':
            speaker = None
        if speaker and speaker.casefold() != sequence_compile.NARRATOR.casefold():
            speaker = _canonical(speaker, names)
            if speaker is None:
                errors.append(f'group {number}: "{_str(group.get("speaker"))}" is not a character name; use an exact name, Narrator or null')
        elif speaker:
            speaker = sequence_compile.NARRATOR
        groups.append({'ids': ids, 'text': text, 'speaker': speaker, 'subtitle_ids': [item for item in (_str(x) for x in group.get('subtitle_ids') or []) if item in subtitle_ids],
                       'confidence': group.get('confidence') if group.get('confidence') in sequence_compile.CONFIDENCE else 'medium', 'note': _str(group.get('note'))[:300],
                       'continues_next': bool(group.get('continues_next'))})
    if position != len(line_ids) and not errors:
        errors.append(f'the lines {", ".join(line_ids[position:position + 6])}{"…" if len(line_ids) - position > 6 else ""} are missing; every input line belongs to exactly one group')
    if errors:
        _fail(errors)
    voices = [{'speaker_label': _str(voice.get('speaker_label')), 'character': voice.get('character'), 'kind': voice.get('kind'), 'confidence': voice.get('confidence'), 'reason': _str(voice.get('reason'))}
              for voice in output.get('voices') or [] if isinstance(voice, dict) and _str(voice.get('speaker_label'))]
    labels = {line['speaker_label'] for line in source if line.get('speaker_label')}
    if source:
        voices = [voice for voice in voices if voice['speaker_label'] in labels]
    return {'lines': groups, 'voices': voices, 'open_questions': _strings(output.get('open_questions'), 8)}


# ---------------------------------------------------------------------- bible

def bible_format():
    return _format('sequence_bible', _object({
        'logline': STRING, 'synopsis': STRING, 'spoken_language': STRING,
        'scenes': {'type': 'array', 'items': _object({'id': STRING, 'from_cut': {'type': 'integer'}, 'to_cut': {'type': 'integer'}, 'location': STRING, 'time_of_day': STRING,
                                                      'lighting': STRING, 'mood': STRING, 'scenario': {'type': ['string', 'null']}})},
        'characters': {'type': 'array', 'items': _object({'name': STRING, 'role': STRING, 'anchors': STRING, 'outfit': STRING, 'arc': STRING})},
        'voices': {'type': 'array', 'items': _object({'speaker_label': STRING, 'character': {'type': ['string', 'null']}, 'kind': {'type': 'string', 'enum': ['character', 'narrator', 'unknown']},
                                                      'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']}, 'reason': STRING})},
        'arc': STRING, 'pacing': STRING, 'continuity': {'type': 'array', 'items': STRING}, 'open_questions': {'type': 'array', 'items': STRING}}))


def validate_bible(output, ctx, timeline):
    """Check and normalise a bible against the character records, scenario notes, cuts and voice labels of the sequence."""
    if not isinstance(output, dict):
        _fail(['The answer must be one JSON object'])
    errors = []
    bible = {key: _str(output.get(key)) for key in ('logline', 'synopsis', 'spoken_language', 'arc', 'pacing')}
    for key in ('logline', 'synopsis', 'spoken_language'):
        if not bible[key]:
            errors.append(f'"{key}" is empty')
    if bible['spoken_language'] and not LANGUAGE.match(bible['spoken_language']):
        errors.append('"spoken_language" must be the language name in English letters, for example Japanese')
    first, last = timeline['cuts'][0]['number'], timeline['cuts'][-1]['number']
    scenario_titles = [item['title'] for item in ctx['scenarios']]
    scenario_ids = {item['title']: item['id'] for item in ctx['scenarios']}
    scenes, expected = [], first
    for scene in output.get('scenes') if isinstance(output.get('scenes'), list) else []:
        if not isinstance(scene, dict) or not all(isinstance(scene.get(key), int) and not isinstance(scene.get(key), bool) for key in ('from_cut', 'to_cut')):
            errors.append('every scene needs integer from_cut and to_cut')
            continue
        if scene['from_cut'] != expected or scene['to_cut'] < scene['from_cut'] or scene['to_cut'] > last:
            errors.append(f"scenes must cover cuts {first} to {last} in order without gaps or overlaps; the scene starting at cut {scene['from_cut']} should start at cut {expected} and end by cut {last}")
            break
        expected = scene['to_cut'] + 1
        title = None
        if _str(scene.get('scenario')) and _str(scene.get('scenario')).lower() != 'null':
            title = _canonical(scene['scenario'], scenario_titles)
            if title is None and scenario_titles:
                errors.append(f"scenario \"{_str(scene['scenario'])}\" is not a scenario note; use one of: {', '.join(scenario_titles)}, or null")
        scenes.append({'id': f'S{len(scenes) + 1}', 'from_cut': scene['from_cut'], 'to_cut': scene['to_cut'], **{key: _str(scene.get(key)) for key in ('location', 'time_of_day', 'lighting', 'mood')},
                       'scenario': title, 'scenario_context_id': scenario_ids.get(title)})
    if not scenes:
        errors.append('"scenes" must list at least one scene')
    elif expected != last + 1 and not errors:
        errors.append(f'scenes stop at cut {expected - 1}; they must cover every cut up to {last}')
    names = [character['name'] for character in ctx['characters']]
    ids = {character['name']: character['id'] for character in ctx['characters']}
    characters = []
    for person in output.get('characters') if isinstance(output.get('characters'), list) else []:
        name = _canonical(person.get('name'), names) if isinstance(person, dict) else None
        if name is None:
            errors.append(f"character \"{_str(person.get('name')) if isinstance(person, dict) else person}\" has no character record; use exact names from: {', '.join(names) or 'none (there are no records)'}")
            continue
        if any(item['name'] == name for item in characters):
            continue
        if not _str(person.get('anchors')):
            errors.append(f'character "{name}" needs anchors: 3 to 6 visible features')
        characters.append({'name': name, 'character_context_id': ids[name], **{key: _str(person.get(key)) for key in ('role', 'anchors', 'outfit', 'arc')}})
    evidence = {voice['speaker_label']: voice for voice in sequence_evidence.voice_evidence(ctx, timeline)}
    given = {_str(voice.get('speaker_label')): voice for voice in output.get('voices') if isinstance(voice, dict)} if isinstance(output.get('voices'), list) else {}
    voices = []
    for label, facts in evidence.items():
        voice = given.get(label) or {}
        linked = facts.get('linked_by_user') or {}
        if len(linked) == 1:
            # The user's own link decides; the model's answer is only kept as the reason it gave.
            name = next(iter(linked))
            voices.append({'speaker_label': label, 'character': name, 'character_context_id': ids.get(name), 'kind': 'character', 'confidence': 'high', 'source': 'transcript',
                           'reason': 'Linked to this character in the transcript.'})
            continue
        name = _canonical(voice.get('character'), names) if _str(voice.get('character')) and _str(voice.get('character')).lower() != 'null' else None
        if _str(voice.get('character')) and _str(voice.get('character')).lower() != 'null' and name is None:
            errors.append(f"voice {label}: \"{_str(voice['character'])}\" has no character record; use an exact name or null")
        kind = voice.get('kind') if voice.get('kind') in ('character', 'narrator', 'unknown') else 'unknown'
        kind = 'character' if name else ('narrator' if kind == 'narrator' else 'unknown')
        voices.append({'speaker_label': label, 'character': name, 'character_context_id': ids.get(name), 'kind': kind,
                       'confidence': voice.get('confidence') if voice.get('confidence') in ('high', 'medium', 'low') else 'low', 'source': 'director', 'reason': _str(voice.get('reason'))})
    if errors:
        _fail(errors)
    return {**bible, 'scenes': scenes, 'characters': characters, 'voices': voices, 'continuity': _strings(output.get('continuity'), 16), 'open_questions': _strings(output.get('open_questions'), 12)}


# ----------------------------------------------------------------------- plan

def open_boundaries(timeline):
    """The hard cuts inside speech that the director decides: every one the user did not place."""
    return [boundary for boundary in timeline['boundaries'] if boundary['speech'] and not boundary.get('fixed') and boundary['id'] not in timeline.get('dialogue_timing', {})]


def plan_format(timeline, bible):
    """json_schema of the plan answer: decision ids are the open hard cuts inside speech, and every cut gets one intent."""
    numbers = [cut['number'] for cut in timeline['cuts']]
    return _format('shot_plan', _object({
        'decisions': decision_items(timeline),
        'cuts': {'type': 'array', 'minItems': len(numbers), 'maxItems': len(numbers), 'items': _object({'cut': {'type': 'integer', 'enum': numbers}, 'intent': STRING})},
        'notes': STRING}))


def decision_items(timeline):
    ids = [boundary['id'] for boundary in open_boundaries(timeline)]
    decision = _object({'id': {'type': 'string', 'enum': ids} if ids else STRING, 'choice': {'type': 'string', 'enum': list(sequence_timeline.CHOICES)}, 'reason': STRING})
    return {'type': 'array', 'items': decision, 'minItems': len(ids), 'maxItems': len(ids)}


def timing_format(timeline):
    return _format('dialogue_timing', _object({'decisions': decision_items(timeline)}))


def validate_decisions(output, timeline, timing=False):
    if not isinstance(output, dict):
        _fail(['The answer must be one JSON object'])
    errors = []
    by_id = {boundary['id']: boundary for boundary in timeline['boundaries']}
    open_ids = [boundary['id'] for boundary in open_boundaries(timeline)]
    decisions = {}
    for item in output.get('decisions') if isinstance(output.get('decisions'), list) else []:
        if not isinstance(item, dict):
            continue
        key = _str(item.get('id'))
        if key in timeline.get('dialogue_timing', {}) or (key in by_id and by_id[key].get('fixed')):
            continue  # placed by the user: final, and not the director's to answer
        if key not in open_ids:
            errors.append(f'decision "{key}" is not one of the listed decisions ({", ".join(open_ids) or "there are none"})')
            continue
        if key in decisions:
            errors.append(f'{key} is answered twice; answer each decision once')
            continue
        choice, reason, boundary = _str(item.get('choice')), _str(item.get('reason')), by_id[key]
        if choice not in boundary['options']:
            errors.append(sequence_timeline.unavailable(boundary, choice))
            continue
        if choice == 'keep_cut' and not reason:
            movable = [name for name in boundary['options'] if name != 'keep_cut']
            errors.append(f"{key}: keep_cut divides {' and '.join(boundary['speech']['lines'])}; give the reason the scene needs it divided at the hard cut"
                          + (f", or keep the phrase whole with {' or '.join(movable)}" if movable else ''))
        decisions[key] = {'id': key, 'choice': choice, 'reason': reason, 'by': 'director'}
    missing = [key for key in open_ids if key not in decisions]
    if missing:
        errors.append(f"answer every decision: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing")
    decisions.update(timeline.get('dialogue_timing', {}))
    choices = {key: item['choice'] for key, item in decisions.items()}
    if not errors:
        try:
            errors.extend(sequence_timeline.check_spans(sequence_timeline.decide(timeline, choices), timeline))
            if timing and not errors:
                for key, choice in choices.items():
                    if choice != 'keep_cut':
                        continue
                    for alternative in by_id[key]['options']:
                        if alternative != choice and not sequence_timeline.check_spans(sequence_timeline.decide(timeline, {**choices, key: alternative}), timeline):
                            errors.append(f'{key}: use {alternative} to keep the unfinished phrase whole')
                            break
        except sequence_timeline.PlanError as exc:
            errors.extend(exc.errors)
    if errors:
        _fail(errors)
    return {'decisions': [decisions[key] for key in by_id if key in decisions]}


def validate_plan(output, timeline, bible):
    """Check one decision per open boundary and one intent per source cut."""
    answer = validate_decisions(output, timeline)
    numbers = [cut['number'] for cut in timeline['cuts']]
    intents = {}
    for item in output.get('cuts') if isinstance(output.get('cuts'), list) else []:
        if isinstance(item, dict) and item.get('cut') in numbers and _str(item.get('intent')):
            intents.setdefault(item['cut'], _str(item['intent']))
    lacking = [number for number in numbers if number not in intents]
    if lacking:
        listed = ', '.join(map(str, lacking[:12])) + ('…' if len(lacking) > 12 else '')
        _fail([f"give every cut an intent: cut{'s' if len(lacking) > 1 else ''} {listed} {'has' if len(lacking) == 1 else 'have'} none"])
    return {**answer, 'cuts': [{'cut': number, 'intent': intents[number]} for number in numbers], 'notes': _str(output.get('notes'))}


def default_choice(boundary, ctx, timeline, bible):
    """What the app chooses for a hard cut inside speech: keep the phrase whole on the side where its speaker is visibly speaking, else with the smaller move."""
    movable = [choice for choice in ('after_line', 'before_line') if choice in boundary['options']]
    if not movable:
        return 'keep_cut'
    line = next(item for item in timeline['lines'] if item['id'] == boundary['speech']['lines'][0])
    name = sequence_evidence.speaker_of(line, bible)['name']
    sides = {'after_line': boundary['after_cut_id'], 'before_line': boundary['before_cut_id']}
    for choice in movable:
        if name in sequence_evidence.speaking_in(ctx['results'].get(sides[choice], {})):
            return choice
    return min(movable, key=lambda choice: abs(boundary['options'][choice]['shift']))


def complete_plan(answer, timeline, bible, ctx, timing=False):
    """A usable plan from an answer that failed validation: its valid choices, the app's default for the rest, the hard cut where neither fits."""
    answer = answer if isinstance(answer, dict) else {}
    given = {}
    for item in answer.get('decisions') if isinstance(answer.get('decisions'), list) else []:
        if isinstance(item, dict) and _str(item.get('id')) not in given:
            given[_str(item.get('id'))] = (_str(item.get('choice')), _str(item.get('reason')))
    decisions = list(timeline.get('dialogue_timing', {}).values())
    choices = {item['id']: item['choice'] for item in decisions}
    for boundary in open_boundaries(timeline):
        tries = []
        choice, reason = given.get(boundary['id'], (None, ''))
        if choice in boundary['options'] and (choice != 'keep_cut' or (reason and not timing)):
            tries.append((choice, reason, 'director'))
        fallback = default_choice(boundary, ctx, timeline, bible)
        tries.append((fallback, 'Chosen by the app: keeps the phrase whole.' if fallback != 'keep_cut' else 'Chosen by the app: no position nearby keeps the phrase whole.', 'app'))
        if timing:
            tries.extend((other, 'Keeps the phrase whole within the shot limits.', 'app') for other in boundary['options'] if other not in (fallback, 'keep_cut'))
        tries.append(('keep_cut', 'Chosen by the app: keeping the phrase whole did not fit next to the other boundaries.', 'app'))
        for choice, reason, by in tries:
            trial = {**choices, boundary['id']: choice}
            try:
                fits = not sequence_timeline.check_spans(sequence_timeline.decide(timeline, trial), timeline)
            except sequence_timeline.PlanError:
                fits = False
            if fits or choice == 'keep_cut':
                choices = trial
                decisions.append({'id': boundary['id'], 'choice': choice, 'reason': reason, 'by': by})
                break
    numbers = [cut['number'] for cut in timeline['cuts']]
    intents = {}
    for item in answer.get('cuts') if isinstance(answer.get('cuts'), list) else []:
        if isinstance(item, dict) and item.get('cut') in numbers and _str(item.get('intent')):
            intents.setdefault(item['cut'], _str(item['intent']))
    return {'decisions': decisions, 'cuts': [{'cut': number, 'intent': intents.get(number) or 'Follow the source.'} for number in numbers], 'notes': _str(answer.get('notes'))}


def plan_content(answer, timeline, bible):
    """The stored plan for a checked (or completed) answer: one span per cut (parts for a long cut) and the decision at each hard cut inside speech."""
    by_id = {boundary['id']: boundary for boundary in timeline['boundaries']}
    intents = {item['cut']: item['intent'] for item in answer['cuts']}
    scenes = {cut['number']: (sequence_evidence.scene_for(bible, cut['number']) or {}).get('id') for cut in timeline['cuts']}
    spans = sequence_timeline.decide(timeline, {item['id']: item['choice'] for item in answer['decisions']}, intents, scenes)
    boundaries = [{'after_cut_id': by_id[item['id']]['after_cut_id'], 'choice': item['choice'], 'reason': item['reason'], 'by': item['by']} for item in answer['decisions']]
    return {'shots': spans, 'boundaries': boundaries, 'notes': answer.get('notes') or ''}


# ----------------------------------------------------------------------- shot

def shot_format(shot, roster, dialogue):
    heard = sequence_prompt.speakers(dialogue)
    voice = _object({'speaker': {'type': 'string', 'enum': heard} if heard else STRING, 'from': {'type': 'number'},
                     'delivery': {'type': 'string', 'enum': list(sequence_prompt.DELIVERIES)}, 'subject': {'type': ['string', 'null']}})
    return _format('shot_direction', _object({
        'subjects': {'type': 'array', 'maxItems': 5, 'items': _object({'name': STRING, 'anchors': STRING, 'keep': STRING})},
        'picture_line': STRING, 'keep_picture': STRING,
        'still': _object({key: STRING for key in STILL_KEYS}),
        'summary': STRING, 'shot_line': STRING, 'framing_lock': STRING, 'light': STRING,
        'camera': _object({'move': {'type': 'string', 'enum': list(sequence_prompt.CAMERA_MOVES)}, 'amplitude': {'type': 'string', 'enum': list(sequence_prompt.AMPLITUDES)},
                           'speed': {'type': 'string', 'enum': list(sequence_prompt.SPEEDS)}, 'detail': STRING}),
        'beats': {'type': 'array', 'minItems': 1, 'items': _object({'end': {'type': 'number'}, 'action': STRING})},
        'voices': {'type': 'array', 'items': voice},
        'scenario_anchors': STRING, 'scenario_keep': STRING,
        'continuity': _object({'from_previous': STRING, 'to_next': STRING}), 'open_questions': {'type': 'array', 'items': STRING}}))


def _prose(content):
    """Every field the model writes as prose, with the name an error message gives it."""
    yield from ((key, content.get(key)) for key in ('picture_line', 'keep_picture', 'summary', 'shot_line', 'framing_lock', 'light', 'scenario_anchors', 'scenario_keep'))
    yield 'camera.detail', content['camera'].get('detail')
    yield from ((f'still.{key}', value) for key, value in content['still'].items())
    yield from ((f'beat {number}', beat['action']) for number, beat in enumerate(content['beats'], 1))
    for subject in content['subjects']:
        yield f"subject {subject['name']}", f"{subject['anchors']} {subject['keep']}"
    yield from ((f'continuity.{key}', value) for key, value in content['continuity'].items())


def _covers(ranges, duration):
    """True when time ranges together cover ``[0, duration]``."""
    reach = 0.0
    for item in sorted(ranges, key=lambda item: item['from']):
        if item['from'] > reach + 0.05:
            return False
        reach = max(reach, item['to'])
    return reach >= duration - 0.05


def _voices(output, shot, dialogue, subjects, errors):
    """The shot's voices: the director's entries checked against the speakers and subjects, with the user's ranges laid over them."""
    heard = sequence_prompt.speakers(dialogue)
    if not heard:
        return []
    duration, names = shot['duration'], [subject['name'] for subject in subjects]
    listed = ', '.join(names) or 'you listed none'
    fixed = [item for item in shot.get('fixed_voices') or [] if item.get('speaker') in heard and sequence_prompt.delivery_of(item.get('delivery'))]
    fixed = [{**item, 'delivery': sequence_prompt.delivery_of(item['delivery'])} for item in fixed]
    entries = []
    for item in output.get('voices') if isinstance(output.get('voices'), list) else []:
        if not isinstance(item, dict):
            continue
        speaker = _canonical(item.get('speaker'), heard)
        if speaker is None:
            errors.append(f"voices: \"{_str(item.get('speaker'))}\" is not heard in this shot; the speakers are {', '.join(heard)}")
            continue
        delivery = sequence_prompt.delivery_of(item.get('delivery'))
        if delivery is None:
            errors.append(f"voices {speaker}: delivery must be one of {', '.join(sequence_prompt.DELIVERIES)}")
            continue
        start = item.get('from')
        start = float(start) if isinstance(start, (int, float)) and not isinstance(start, bool) and math.isfinite(start) else 0.0
        if not -0.05 <= start <= duration + 0.05:
            errors.append(f'voices {speaker}: "from" {start:g} s is outside the shot; seconds run from 0 to {duration:g}')
            continue
        if speaker == sequence_prompt.NARRATOR and delivery in sequence_prompt.NEEDS_SUBJECT:
            errors.append(f'voices {speaker}: a narrator is off_screen or narrative')
            continue
        raw = _str(item.get('subject'))
        raw = '' if raw.lower() == 'null' else raw
        subject = None
        if delivery != 'off_screen':
            subject = _canonical(raw, names) if raw else _canonical(speaker, names)
            if subject is None and (raw or delivery in sequence_prompt.NEEDS_SUBJECT):
                errors.append(f"voices {speaker}: {delivery} needs \"subject\" to be one of your subjects ({listed}); use off_screen while {speaker} is not in the picture")
                continue
        entries.append({'speaker': speaker, 'from': round(min(max(0.0, start), duration), 3), 'delivery': delivery, 'subject': subject})
    for speaker in heard:
        if not any(entry['speaker'] == speaker for entry in entries) and not _covers([item for item in fixed if item['speaker'] == speaker], duration):
            spoken = ', '.join(line['line'] for line in dialogue if sequence_prompt.speaker_key(line) == speaker)
            errors.append(f'voices needs an entry from second 0 for {speaker}, who speaks {spoken} in this shot')
    voices = sequence_prompt.overlay(entries, fixed, duration)
    for entry in voices:
        if entry.get('fixed') and entry['delivery'] in sequence_prompt.NEEDS_SUBJECT:
            subject = _canonical(entry.get('subject'), names)
            if subject is None:
                errors.append(f"the user set {entry['speaker']} {entry['delivery']} from {entry['from']:g} s to {entry['to']:g} s: the speaker must be in the picture then, so list {entry['speaker']} "
                              f"among your subjects (or name the subject who carries this voice in its voices entry) and write the acting around it")
            else:
                entry['subject'] = subject
    return voices


def validate_shot(output, shot, bible, roster, scenario, dialogue, generated=False):
    """Check and normalise one shot's direction, render it and apply the H3 format rules.

    ``roster`` are the named characters of the covered cuts and ``dialogue`` the resolved lines of the
    shot. Returns the normalised direction; problems (including those of the rendered prompt) are raised.
    ``generated`` is true for a model's answer, which must give the still's placement; a hand or assistant
    revision of a shot written before that key existed may leave it empty.
    """
    if not isinstance(output, dict):
        _fail(['The answer must be one JSON object'])
    errors = []
    names = [person['name'] for person in roster]
    subjects = []
    for item in output.get('subjects') if isinstance(output.get('subjects'), list) else []:
        name = _str(item.get('name')) if isinstance(item, dict) else ''
        name = _canonical(name, names) or name
        if not name or any(subject['name'].casefold() == name.casefold() for subject in subjects):
            continue
        if not _str(item.get('anchors')):
            errors.append(f'subject "{name}" needs anchors: 3 to 6 visible features')
        subjects.append({'name': name, 'anchors': _str(item.get('anchors')), 'keep': _str(item.get('keep'))})
    if len(subjects) > 5:
        errors.append('list at most 5 subjects, the most important first')
    # No soundscape: the app writes the sound section, which follows the original audio instead of a guess about it.
    content = {'subjects': subjects[:5], **{key: _str(output.get(key)) for key in ('picture_line', 'keep_picture', 'summary', 'shot_line', 'framing_lock', 'light', 'scenario_anchors', 'scenario_keep')}}
    for key in ('picture_line', 'summary', 'shot_line'):
        if not content[key]:
            errors.append(f'"{key}" is empty')
    still = output.get('still') if isinstance(output.get('still'), dict) else {}
    content['still'] = {key: _str(still.get(key)) for key in STILL_KEYS}
    for key in ('shot', 'setting', 'lighting', *(('placement',) if generated else ())):
        if not content['still'][key]:
            errors.append(f'"still.{key}" is empty')
    camera = output.get('camera') if isinstance(output.get('camera'), dict) else {}
    if camera.get('move') not in sequence_prompt.CAMERA_MOVES:
        errors.append(f"camera.move must be one of: {', '.join(sequence_prompt.CAMERA_MOVES)}")
    content['camera'] = {'move': camera.get('move') if camera.get('move') in sequence_prompt.CAMERA_MOVES else 'static',
                         'amplitude': camera.get('amplitude') if camera.get('amplitude') in sequence_prompt.AMPLITUDES else 'small',
                         'speed': camera.get('speed') if camera.get('speed') in sequence_prompt.SPEEDS else 'slow',
                         'detail': '' if camera.get('move') == 'static' else _str(camera.get('detail'))}
    duration, beats, previous = shot['duration'], [], 0.0
    spoken = [sequence_compile.normalize(line['text']).casefold() for line in dialogue if line.get('text')]
    for number, beat in enumerate(output.get('beats') if isinstance(output.get('beats'), list) else [], 1):
        end = beat.get('end') if isinstance(beat, dict) else None
        if isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end) or not _str(beat.get('action')):
            errors.append(f'beat {number} needs a numeric "end" in seconds and an "action"')
            continue
        if end <= previous + min(0.2, duration / 2):
            errors.append(f'beat {number} ends at {end:g} s, which is not after the previous beat ({previous:g} s)')
            continue
        if end > duration + 0.5:
            errors.append(f'beat {number} ends at {end:g} s but the shot lasts {duration:g} s; times are seconds from the start of this shot')
            continue
        action = _str(beat['action'])
        if generated:
            quotes = re.findall(r"""["“「『]([^"”」』]+)["”」』]|(?<!\w)'([^']+)'(?!\w)""", action)
            quoted = {sequence_compile.normalize(value).casefold() for matches in quotes for value in matches if value}
            normalized_action = sequence_compile.normalize(action).casefold()
            if any(words and (words in quoted or len(words) >= 6 and words in normalized_action) for words in spoken):
                errors.append(f'beat {number} repeats the dialogue; describe the visible mouth action and leave the words to the app')
        previous = min(float(end), duration)
        beats.append({'end': round(previous, 3), 'action': action})
    if not beats:
        errors.append(f'"beats" needs at least one beat; the last one ends at {duration:g} s')
    else:
        beats[-1]['end'] = duration
        limit = max(2, math.ceil(duration / 1.25))
        if len(beats) > limit:
            errors.append(f'{len(beats)} beats are too many for {duration:g} s; write at most {limit}, each with a visible start, change and arrival')
    content['beats'] = beats
    content['voices'] = _voices(output, shot, dialogue, content['subjects'], errors)
    continuity = output.get('continuity') if isinstance(output.get('continuity'), dict) else {}
    content['continuity'] = {key: _str(continuity.get(key)) for key in ('from_previous', 'to_next')}
    content['open_questions'] = _strings(output.get('open_questions'), 8)
    if scenario and not content['scenario_anchors']:
        errors.append('"scenario_anchors" is empty: give 3 to 5 visible features of the place, because the shot receives its scenario picture')
    if not scenario:
        content['scenario_anchors'] = content['scenario_keep'] = ''
    for field, value in _prose(content):
        for pattern, problem in BANNED:
            if value and pattern.search(value):
                errors.append(f'{field} {problem}')
    if errors:
        _fail(errors)
    _, verdict, _ = sequence_prompt.build(shot, content, bible, roster, scenario, dialogue)
    if not verdict['ok']:
        _fail([error.replace('detailed_description has', 'shot_line, framing_lock, light and beats together have') for error in verdict['errors']])
    return content


# --------------------------------------------------------------------- review

def review_format(count):
    issue = _object({'shot': {'type': 'integer', 'minimum': 1, 'maximum': max(1, count)}, 'severity': {'type': 'string', 'enum': ['fix', 'check']}, 'problem': STRING, 'suggestion': STRING})
    return _format('sequence_review', _object({'verdict': STRING, 'issues': {'type': 'array', 'items': issue}}))


def validate_review(output, count):
    if not isinstance(output, dict) or not _str(output.get('verdict')):
        _fail(['"verdict" is empty'])
    issues = []
    for item in output.get('issues') if isinstance(output.get('issues'), list) else []:
        if not isinstance(item, dict) or isinstance(item.get('shot'), bool) or not isinstance(item.get('shot'), int) or not 1 <= item['shot'] <= count or not _str(item.get('problem')):
            continue
        issues.append({'shot': item['shot'], 'severity': item.get('severity') if item.get('severity') in ('fix', 'check') else 'check',
                       'problem': _str(item['problem']), 'suggestion': _str(item.get('suggestion'))})
    return {'verdict': _str(output['verdict']), 'issues': issues}
