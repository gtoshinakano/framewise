# Composition brief size measurement (draft)

Episode `e4d40c01-3a6e-493d-8e7b-c891db2659ae`, cuts 44–68. UTF-8 JSON with compact serialization. The before columns count each legacy brief cut and the corresponding separately fetched structured-analysis response; after counts the current composition cut. Range totals include their JSON envelopes. The source snapshots were captured at different times while analysis could continue. The fixture size-budget test and evidence coverage are not yet satisfied; these measurements do not establish completion.

| Cut | Before brief bytes | Before structured bytes | After cut bytes |
| ---: | ---: | ---: | ---: |
| 44 | 4785 | 12678 | 1133 |
| 45 | 7233 | 17567 | 3412 |
| 46 | 7100 | 14978 | 2585 |
| 47 | 9724 | 18174 | 4457 |
| 48 | 30124 | 13084 | 2561 |
| 49 | 9020 | 15333 | 3123 |
| 50 | 5790 | 15442 | 2736 |
| 51 | 10075 | 21715 | 3454 |
| 52 | 8782 | 15888 | 3507 |
| 53 | 8657 | 16980 | 3157 |
| 54 | 9134 | 14114 | 3289 |
| 55 | 9313 | 16214 | 3142 |
| 56 | 4478 | 11537 | 1101 |
| 57 | 10193 | 17637 | 4063 |
| 58 | 5729 | 14808 | 2576 |
| 59 | 6342 | 14274 | 2042 |
| 60 | 6066 | 14914 | 3074 |
| 61 | 6057 | 11959 | 2212 |
| 62 | 7432 | 10826 | 2167 |
| 63 | 5117 | 11131 | 1809 |
| 64 | 12304 | 14774 | 3551 |
| 65 | 6246 | 15396 | 2403 |
| 66 | 4991 | 11934 | 1927 |
| 67 | 5302 | 11576 | 1607 |
| 68 | 8424 | 17740 | 4476 |

Legacy range: 208,523 bytes; 25 structured-analysis responses: 370,673 bytes; combined: 579,196 bytes. Composition range: 70,182 bytes. All 25 cuts were marked ready in this snapshot.

## Cut 49 draft example

This record is a work in progress, not a claim that all available evidence is represented.

```json
{
  "id": "a8f99bcc-4bdd-4697-8d8d-3d8430bbe890",
  "cut_number": 49,
  "duration": 3.461792,
  "ready": true,
  "updated_at": "2026-11-06T11:24:18.384Z",
  "analysis_rev": "qCMJtoQ32xo",
  "initial_frame_id": "ff53b624-8988-4f09-9b08-bf66915a4f84",
  "last_frame_id": "96f0b336-9f4b-45af-9493-83549b87192a",
  "shot": {
    "size": "MCU",
    "angle": "eye",
    "camera": "static"
  },
  "setting": "supermarket or grocery store; refrigerated display cases with packaged meat and produce shelves stocked with green vegetables and fruits background",
  "lighting": "Bright, diffuse overhead lighting; The light source appears to be from above",
  "subjects": [
    {
      "character_context_id": "1ab61da1-6b7b-44af-89cb-9ea73ebb83a9",
      "name": "Mai Sakurajima",
      "identity": "confirmed",
      "frame_pos": "left mid",
      "facing": "camera",
      "pose": "standing",
      "hands": "right hand raised, holding an orange carrot",
      "gaze": "forward",
      "expression": "neutral to slightly concerned or questioning",
      "outfit_visible": "brown school blazer over white collared shirt with red tie"
    },
    {
      "character_context_id": null,
      "name": "unknown",
      "frame_pos": "left mid",
      "pose": "Standing next to the main character",
      "outfit_visible": "white garment (possibly shirt or coat)"
    }
  ],
  "end_state": {
    "subjects": [
      {
        "subject": "Mai Sakurajima",
        "frame_pos": "right fg",
        "facing": "right; three-quarter view",
        "gaze": "at the carrot",
        "expression": "slight smile"
      }
    ],
    "size": "MS"
  },
  "beats": [
    {
      "t0": 0.0,
      "t1": 1.978,
      "subject": "Mai Sakurajima",
      "action": "Holds a carrot in her right hand, initially looking forward with a neutral expression; turns her head slightly to her right, closes her eyes briefly, and smiles while looking down at the carrot; Man in white shirt: Remains in the background, facing away from the camera, looking down at produce"
    },
    {
      "t0": 1.978,
      "t1": 3.462,
      "subject": "Mai Sakurajima",
      "action": "Holding a carrot in her right hand, looking toward the right side of the frame with a slight smile; Her head is turned to her right, showing a profile view; Her left hand is partially visible near her waist, holding a small object; The man: Holding a bunch of bananas in both hands, looking down at them; His head is slightly tilted downward, and his gaze is directed at the bananas; His body is angled slightly toward the woman"
    }
  ],
  "dialogue": [
    {
      "speaker_label": "SPEAKER_07",
      "character_context_id": "290d19bd-b44d-4500-ad8e-f1b45fd6057a",
      "character_name": "Sakuta Azusagawa",
      "text": "人にはどう見えてるんですか？",
      "t0": 0,
      "t1": 1.297,
      "crosses_cut": true
    },
    {
      "speaker_label": "SPEAKER_04",
      "character_context_id": "1ab61da1-6b7b-44af-89cb-9ea73ebb83a9",
      "character_name": "Mai Sakurajima",
      "text": "見えてないみたいよ",
      "t0": 2.057,
      "t1": 3.097,
      "crosses_cut": true
    }
  ],
  "subtitles": [
    {
      "text": "O que acontece com essa cenoura em sua mão, se pessoas não te enxergam?",
      "lang": "pt",
      "t": 0.0
    },
    {
      "text": "Parece que não enxergam.",
      "lang": "pt",
      "t": 3.42
    }
  ],
  "issues": [
    "Subtitle and transcript may disagree"
  ],
  "links": [
    {
      "label_source": "auto",
      "stale": false,
      "subject": 0,
      "first": [
        [
          252.2,
          0,
          925.4,
          720
        ]
      ],
      "last": [
        [
          322.6,
          5.0,
          985.6,
          720
        ]
      ]
    }
  ],
  "confidence": {
    "shot": "low",
    "setting": "low",
    "lighting": "low",
    "subjects": "low",
    "beats": "low",
    "end_state": "low"
  },
  "unavailable": {
    "lighting": "lp",
    "end_state": "ia"
  }
}
```
