---
name: look_at_frames
title: Look at the frames
scope: any
vision: true
description: Look at the real video frames (first frame, keyframes, any moment, zoomed details) to check a result, a prompt or what the user says against the picture.
examples: Does the first frame really look like the analysis says? | Look at the keyframes and tell me what changes | Zoom into her hands in the first frame
---
# Looking at the video

`view_frames` shows you real frames, cut from the original video, in the message that follows its result. You have it only while the model answering can read images. If a user asks you to look and the tool is not in your list, say so and work from the saved results.

## What to ask for
- `first` / `last`: the cut's boundary frames, the ones its results were made from.
- `F1`…`Fn`: the keyframes the keyframe analysis describes (`get_result(type="keyframe_analysis")` lists them with their times). To confirm a change, ask for the two keyframes around it, for example `["F2", "F3"]`.
- Seconds from the start of the cut (`"1.25"`): any other moment, such as where a beat says something happens.
- `crop` `[x1, y1, x2, y2]` (fractions of the frame, from the top left): zoom into one region of every frame you ask for, to read a hand, an object, an expression or small text. Look at the whole frame first, then zoom where the detail is.
- `cut`: another cut by its number, for continuity: the `last` frame of the previous cut beside the `first` of this one.

At most 4 frames per call. Each picture costs context and only the latest few stay attached, so ask only for what the question needs. Never ask for the same frame twice in a conversation unless you need a zoom.

## How to use what you see
1. Read the saved result first, then look to confirm or contradict a specific claim in it. Do not describe the whole picture when one detail is in question.
2. Say which picture shows what ("in F3 her left hand is on the cup"). Describe only what is visible; if it is dark, small or cut off, zoom once, then say you cannot tell.
3. Who someone is comes from the character boxes (`get_characters`), the saved results and the user, never from a face or an outfit alone. Position and pose you may state from the picture.
4. A frame is one instant. It cannot show that something moves, speaks or sounds: compare two frames for a change, use the transcript for speech, and never assume the audio.
5. Subtitles and on-screen text are reference only; never put them into an image or animation prompt.
6. When the picture contradicts a result or a prompt, tell the user which frame shows it and propose the smallest correction with the usual proposal tool. The user's review answers still override what you see, and a doubt is a question, not an edit.

## Answer
Name the frames you looked at, what is visible in them, how that compares with the result or prompt, and what you propose (or that nothing needs to change). Keep it short.
