# Replay: review_answers

2026-11-22T1808 UTC, commit 1388af0

## Ground truth

74 cuts where the user answered a current review's questions; 0 of them have a shadow decision.

## The reviewer's recommended option against the user

| topic | cites a picture | text only | user wrote own text |
|---|---|---|---|
| camera | 1/1 (100%) | — | 1 |
| emotion | 3/5 (60%) | 1/1 (100%) | 0 |
| gaze | 8/9 (89%) | 1/1 (100%) | 2 |
| identity | 3/3 (100%) | 2/2 (100%) | 0 |
| other | 1/1 (100%) | — | 0 |
| pose | 11/18 (61%) | — | 0 |
| speaking | 27/40 (68%) | 12/18 (67%) | 7 |
| state | 6/11 (55%) | 1/2 (50%) | 1 |

All: 77/112 (69%). The resolver applies an answer alone only at 90% measured agreement (D12).

## The local decider

No shadow decisions yet: run `fw review-decisions run EP --shadow` on an episode the user answered (Episode 2), wait for it, copy the database again (`FRESH=1`) and re-run this replay.

## What the resolver would do today (profile and trust as stored)

| topic | apply | right of applied | Claude | person |
|---|---|---|---|---|
| camera | 1 | 1/1 (100%) | 0 | 0 |
| emotion | 0 | — | 0 | 6 |
| gaze | 10 | 9/10 (90%) | 0 | 0 |
| identity | 0 | — | 5 | 0 |
| other | 0 | — | 0 | 1 |
| pose | 0 | — | 18 | 0 |
| speaking | 0 | — | 0 | 58 |
| state | 0 | — | 0 | 13 |

## Trust per topic (as the resolver reads it)

- camera: — from 1 (too few)
- emotion: — from 6 (too few)
- gaze: 90% from 10 (recommended)
- identity: — from 5 (too few)
- other: — from 1 (too few)
- pose: 61% from 18 (recommended)
- speaking: 67% from 58 (recommended)
- state: 54% from 13 (recommended)

## Where the user chose another option than the recommended one (35)

- Episode 2 cut 2 [speaking · Rio Futaba] Is Rio Futaba's mouth moving when she speaks the transcript line?
  - recommended: No, her mouth remains closed while the line is spoken.
  - user: Yes, her mouth moves to lip-sync the line.
  - why: The line strip and keyframes (F2, F3) show Rio's mouth closed, while the transcript indicates she is speaking. The app checks also flag 'mouth_without_line'.
- Episode 2 cut 2 [speaking · Rio Futaba] When does Rio Futaba start speaking?
  - recommended: She starts speaking later, around 2.6s, matching the transcript line.
  - user: She starts speaking immediately at the beginning of the cut (0.0s).
  - why: The transcript line starts at 2.592s, and the visual evidence shows no mouth movement before that time, contradicting the beats which suggest speaking starts at 0.22s.
- Episode 2 cut 3 [speaking · Rio Futaba] When does Rio Futaba's line start?
  - recommended: 2.592s
  - user: 0.01s
  - why: Transcript says 2.592s, beats say 0.01s. Transcript is the primary source for timing.
- Episode 2 cut 4 [speaking · Rio Futaba] When does Rio Futaba start speaking the line about the cat and radioactive material?
  - recommended: Rio starts speaking around 8.9s, after placing the first items.
  - user: Rio starts speaking immediately at the beginning of the cut (0.0s).
  - why: The transcript line S1 starts at 8.912s, while the performance beats claim speaking from 0.24s. The subtitles also start at 8.625s.
- Episode 2 cut 5 [speaking · Rio Futaba] When does Rio Futaba stop speaking?
  - recommended: Rio Futaba stops speaking around 2.0s into the cut, as her mouth is closed in the line strip at +2.08s.
  - user: Rio Futaba stops speaking around 3.16s into the cut, as stated in the performance beats.
  - why: The line strip shows Rio's mouth closed at +2.08s and +2.81s, contradicting the beats which claim she speaks until 3.16s. The keyframes also show her mouth closed at 3.59s.
- Episode 2 cut 6 [pose · Sakuta Azusagawa] Is Sakuta Azusagawa standing or sitting in this cut?
  - recommended: Standing
  - user: Sitting
  - why: First frame description and previous/next cuts show him standing. Keyframes say sitting, but pictures are chest-up and context favors standing.
- Episode 2 cut 9 [state · Sakuta Azusagawa] Is Sakuta Azusagawa sitting or standing in the opening still?
  - recommended: Standing
  - user: Sitting
  - why: Keyframes describe him as 'Standing still' and the next cut shows him standing with his back to the camera.
- Episode 2 cut 12 [speaking · Rio Futaba] Is Rio Futaba's delivery an inner voice or on-screen lip-synced speech?
  - recommended: Rio Futaba's delivery is an inner voice.
  - user: Rio Futaba's delivery is on-screen lip-synced speech.
  - why: The app check flags 'line_without_visible_speaking', and the mouth shape is static, suggesting an inner voice rather than active lip-syncing.
- Episode 2 cut 13 [speaking · Sakuta Azusagawa] How should Sakuta Azusagawa's mouth behave during the line '人間も同じってことか。'?
  - recommended: His mouth remains static and slightly open (inner_voice).
  - user: His mouth moves to lip-sync the line (on_screen).
  - why: The line strip shows Sakuta's mouth shape is static and slightly open throughout the line, contradicting the beats which claim his mouth is 'speaking'.
- Episode 2 cut 14 [speaking · Sakuta Azusagawa] How is Sakuta Azusagawa's speech delivered given that he is visible in the background but his mouth is closed?
  - recommended: Inner voice (visible, mouth still)
  - user: Off-screen (character not visible)
  - why: The line strip shows Sakuta's mouth is closed, and he is visible in the background, making 'inner voice' the most likely delivery type compared to 'off-screen' (which implies invisibility) or 'on-screen' (which requires mouth movement).
- Episode 2 cut 129 [speaking · Sakuta Azusagawa] How is Sakuta Azusagawa's dialogue delivered in this cut?
  - recommended: Off-screen or inner voice, as Sakuta is not visible in the frames.
  - user: On-screen lip-synced, as Sakuta is visible peeking over a barrier.
  - why: The pictures show only a shower head. Sakuta is not visible, so he cannot be lip-synced on-screen.
- Episode 2 cut 131 [speaking · Sakuta Azusagawa] Who speaks the line 'Yahho!' in this cut?
  - recommended: Kaede Azusagawa
  - user: Sakuta Azusagawa
  - why: The transcript attributes the line to Sakuta, but the speaker label SPEAKER_03 is linked to Kaede in the surrounding dialogue, and the 'voices' section links SPEAKER_03 to Kaede (2 lines) and Sakuta (1 line).
- Episode 2 cut 137 [pose · Sakuta Azusagawa] What is Sakuta's mouth state in the first frame?
  - recommended: Mouth is closed.
  - user: Mouth is slightly open.
  - why: The first frame picture shows Sakuta's mouth is closed, contradicting the first frame description which claimed it was slightly open.
- Episode 2 cut 137 [pose · Sakuta Azusagawa] What is Sakuta's mouth state in keyframe F3 (1.58s)?
  - recommended: Mouth is closed.
  - user: Mouth is slightly open.
  - why: The keyframe F3 picture shows Sakuta's mouth is closed, contradicting the keyframe description which claimed it was slightly open.
- Episode 2 cut 137 [speaking · Sakuta Azusagawa] Is Sakuta's mouth moving consistently during the dialogue line?
  - recommended: No, his mouth is closed at the start and end of the line strip.
  - user: Yes, his mouth is open and moving throughout the line.
  - why: The line strip shows Sakuta's mouth is closed in the first and last tiles, contradicting the 'on_screen' delivery which implies continuous speaking.
- Episode 2 cut 139 [pose · Sakuta Azusagawa] Does Sakuta Azusagawa turn his head to face the camera during the cut?
  - recommended: No, he remains facing screen-left throughout
  - user: Yes, he turns to face the camera
  - why: Pictures F2, F3, F4, and F5 all show Sakuta facing screen-left, contradicting the keyframes claim that he turns to face the camera.
- Episode 2 cut 141 [state · Crying Lost Child] Is the Crying Lost Child standing or sitting/crouching in the first frame?
  - recommended: The Crying Lost Child is sitting or crouching on the ground.
  - user: The Crying Lost Child is standing.
  - why: The picture F1 clearly shows the child sitting/crouching, contradicting the first frame description.
- Episode 2 cut 141 [gaze · Sakuta Azusagawa] Where is Sakuta Azusagawa looking in the first frame?
  - recommended: Sakuta Azusagawa is looking at the Crying Lost Child.
  - user: Sakuta Azusagawa is looking up and to the left.
  - why: The picture F1 shows Sakuta looking at the child, contradicting the keyframe F1 description.
- Episode 2 cut 144 [pose · Crying Lost Child] Is the Crying Lost Child standing or sitting in the opening still?
  - recommended: Sitting on the ground behind Sakuta
  - user: Standing behind Sakuta
  - why: The first frame description says standing, but the keyframes and picture F1 show the character sitting.
- Episode 2 cut 144 [pose · Tomoe Koga] What is Tomoe Koga's arm position in the opening still?
  - recommended: Right arm extended forward pointing
  - user: Arms slightly bent and fists clenched
  - why: The first frame description says fists clenched, but the keyframes and picture F1 show her pointing.
- Episode 2 cut 148 [speaking · Tomoe Koga] Is Tomoe Koga's line 'え？' (Eh?) delivered as an inner voice (mouth closed) or on-screen (mouth moving)?
  - recommended: Inner voice: The line is an inner thought, and Tomoe's mouth remains closed throughout.
  - user: On-screen: The line is spoken aloud, and Tomoe's mouth moves to form the sound.
  - why: The keyframes show Tomoe's mouth closed at F2 (0.58s) and F3 (1.96s), which brackets the transcript time of the line (567.567s). The app_check 'line_without_visible_speaking' and the beats' 'inner_voice' label support this.
- Episode 2 cut 154 [state · Sakuta Azusagawa] What is visible in the frame at 2.21s (F4)?
  - recommended: A white skirt (Tomoe's skirt).
  - user: Sakuta Azusagawa with his mouth slightly open.
  - why: The picture F4 shows a white skirt, contradicting the keyframe description which claims Sakuta is visible.
- Episode 2 cut 155 [state · Tomoe Koga] What is the final state of the cut?
  - recommended: Tomoe Koga's face in profile, eyes closed, blushing.
  - user: Tomoe Koga's lower body with skirt fallen back down.
  - why: Picture F5 shows the face, contradicting the keyframe description of the skirt.
- Episode 2 cut 155 [state · Tomoe Koga] Does the cut contain a transition to a face shot?
  - recommended: Yes, it transitions to Tomoe Koga's face.
  - user: No, it remains on the skirt the whole time.
  - why: Picture F5 shows the face, while keyframes F1-F4 show the skirt.
- Episode 2 cut 176 [state · Sakuta Azusagawa] What happens at the end of the cut?
  - recommended: The camera cuts to Mai Sakurajima.
  - user: Sakuta remains in frame with mouth closed.
  - why: Picture F5 shows Mai Sakurajima, contradicting the keyframe text which says Sakuta is still in frame.
- Episode 2 cut 178 [speaking · Mai Sakurajima] How is Mai Sakurajima's line delivered?
  - recommended: inner_voice (visible, mouth still)
  - user: on_screen (lip-synced)
  - why: The line strip shows Mai's mouth is static during her line, contradicting the 'on_screen' delivery in the beats.
- Episode 2 cut 197 [speaking · Shoko Makinohara] How is Shoko Makinohara's speech delivered in this cut?
  - recommended: Inner voice (visible character, mouth remains closed).
  - user: Lip-synced (on_screen) with her mouth moving.
  - why: The line strip images show Shoko's mouth is closed throughout the speech, contradicting the beats' claim of 'on_screen' and 'mouth: speaking'.
- Episode 2 cut 204 [speaking · Sakuta Azusagawa] For the first line (0.27-2.03s), is the delivery inner_voice with a closed mouth, or on_screen with a moving mouth?
  - recommended: inner_voice with a closed mouth
  - user: on_screen with a moving mouth
  - why: The line strip shows a closed mouth, contradicting the app check but supporting the beats.
- Episode 2 cut 207 [emotion · Mai Sakurajima] What is Mai Sakurajima's expression when she begins speaking?
  - recommended: Neutral and serious
  - user: Slightly surprised
  - why: The performance beats describe a 'slightly surprised look', but the keyframes and line strip show her mouth open with a neutral/serious expression, without raised brows or widened eyes typical of surprise.
- Episode 2 cut 217 [speaking · Mai Sakurajima] How should Mai Sakurajima's dialogue be delivered given her back is to the camera?
  - recommended: On-screen, but mouth not visible (no lip sync needed)
  - user: Off-screen voice
  - why: Mai is on screen but facing away, so her mouth is not visible. The transcript attributes the line to her. It is not an inner voice as she is addressing others.
- Episode 2 cut 221 [pose · Mai Sakurajima] Where is Mai Sakurajima's left hand at keyframe F4?
  - recommended: Beside her body
  - user: In front of her torso
  - why: Picture F4 shows her hand down by her side, contradicting the keyframe description of it being in front of her torso.
- Episode 2 cut 114 [emotion · Mai Sakurajima] What is Mai Sakurajima's emotion in the first frame?
  - recommended: Calm and neutral.
  - user: Mild contentment.
  - why: First frame says mild contentment, but the picture shows a neutral face, and the context is a serious conversation.
- Episode 2 cut 122 [speaking · Mai Sakurajima] How does Mai Sakurajima deliver her lines?
  - recommended: Inner voice (mouth closed)
  - user: On-screen lip-synced
  - why: The line strip shows Mai's mouth closed while the transcript attributes speech to her.
- Episode 2 cut 130 [speaking · Sakuta Azusagawa] Does Sakuta Azusagawa's mouth move during his inner monologue?
  - recommended: Yes, it opens slightly at the end of the line.
  - user: No, it remains closed throughout the line.
  - why: The line strip shows the mouth opening slightly at +3.08s, while the beats claim it remains closed.
- Episode 2 cut 133 [speaking · Sakuta Azusagawa] How is Sakuta Azusagawa's line 'Hey, don't peek!' delivered?
  - recommended: Inner voice (mouth closed)
  - user: On-screen lip-synced
  - why: The line strip for Sakuta's line shows his mouth is closed in all frames, contradicting the transcript's implication of speaking and the beats' potential on-screen delivery.
