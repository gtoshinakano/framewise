"""WeVisOne subtitle localization and conditional OCR for selected frames."""
from pathlib import Path
from . import media, subtitle_evidence


def inspect_subtitles(original, cut, config, has_transcript, scratch, model, on_frame=None):
    from PIL import Image
    from wevisone_vl.visualization import parse_predictions
    from .inference_worker import sample_times
    selected = sample_times(cut['start_time'], cut['end_time'], config)
    first, last = media.cut_frame_times(cut['start_time'], cut['end_time'])
    times = sorted(set([first, last, *selected]))
    samples = []
    ocr_performed = False
    for index, requested in enumerate(times):
        if on_frame:
            on_frame(index + 1, len(times))
        target = Path(scratch) / 'subtitle-frame.jpg'
        last_frame = len(times) > 1 and index == len(times) - 1
        actual = media.extract_cut_frame(Path(original), target, requested, cut['start_time'], cut['end_time'], width=1280, last=last_frame)
        with Image.open(target) as image:
            image = image.convert('RGB')
            detection = model.predict(image, task='detection', query=['subtitles'], max_new_tokens=512, temperature=0.0, decode_matting=False)
            regions = parse_predictions(detection.response, image.width, image.height)
            readings = []
            selected_for_ocr = requested in selected
            if has_transcript and regions and selected_for_ocr:
                ocr = model.predict(image, task='ocr', max_new_tokens=1024, temperature=0.0, decode_matting=False)
                ocr_performed = True
                readings = subtitle_evidence.subtitle_readings(regions, parse_predictions(ocr.response, image.width, image.height))
            samples.append({'time': actual, 'requested_time': requested, 'selected_for_ocr': selected_for_ocr, 'image_size': [image.width, image.height],
                            'regions': [entry['coords'] for values in regions.values() for entry in values if entry.get('type') == 'box'],
                            'readings': readings})
    return {'samples': samples, 'events': subtitle_evidence.consolidate([sample for sample in samples if sample['selected_for_ocr']]),
            'ocr_requested': bool(has_transcript), 'ocr_performed': ocr_performed, 'language_note': 'Language is unknown until verified.',
            'limitations': 'Times are sampled observations; predicted subtitle boxes and OCR may be incomplete or incorrect.'}
