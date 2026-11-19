"""The target-frame judge's prompt and response grammar, independent of the job runner."""

PROMPT_VERSION = 'target-frame-judge-v1'
CELL = ('same', 'different_person', 'not_a_person', 'merged_two_people', 'unclear')
UNMATCHED = ('background_extra', 'not_a_person', 'new_character')
SYSTEM = '''You review a tracker over one uninterrupted anime shot. You have one contact sheet. In each slot row the leftmost crop is its identity reference from the reviewed opening frame. F1, F2, and so on are consecutive keyframes. The label S2·F4 means slot 2, frame 4. The bottom strip shows unassigned crops (U labels) and suspected merged boxes (M labels). Skip blank cells and give exactly one choice for each ID named by the user.

For a tracked crop, compare it with that slot's reference and other appearances. same means the same person; different_person means one visible person with a different identity; not_a_person means no real person or visible part of a person; merged_two_people means one box encloses two distinct tracked people and neither is just an incidental edge; unclear is for insufficient evidence. Partial body crops and back views can still be the same person. A new person without a reference is unclear unless the crop is clearly not a person or merged. For an M crop, judge the same choices, with merged_two_people only if there really are two people.

For an unassigned crop choose slot_N only if it belongs to that exact established slot, background_extra for an incidental or very distant extra, not_a_person when it is scenery or an object, or new_character for a distinct foreground character. Cast changes yes means a new foreground character outside the opening-frame cast enters the shot; someone in the original cast temporarily leaving or becoming occluded is not a cast change. Only the visual evidence decides. Return the requested JSON only.'''


def schema(cells, unmatched, slots):
    def enum(values):
        return {'type': 'string', 'enum': list(values)}

    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}

    output = {'cast_changes': enum(('yes', 'no'))}
    if cells:
        output['cells'] = obj({cell: enum(CELL) for cell in cells})
    if unmatched:
        options = (*UNMATCHED, *(f'slot_{slot + 1}' for slot in slots))
        output['unmatched'] = obj({candidate: enum(options) for candidate in unmatched})
    return obj(output)


def evidence(cells, unmatched, slots):
    return ('Tracked and suspected merge crop IDs: ' + (', '.join(cells) or 'none') + '\nUnassigned crop IDs: '
            + (', '.join(unmatched) or 'none') + '\nOpening cast slots: ' + (', '.join(f'S{n + 1}' for n in slots) or 'none'))


def validate(value, grammar):
    """The same requirements apply if an endpoint falls back to plain JSON."""
    def check(node, spec, path):
        if spec['type'] == 'object':
            if not isinstance(node, dict) or set(node) != set(spec['properties']):
                raise ValueError(f'{path} must have exactly these keys: {", ".join(spec["properties"])}')
            for key, child in spec['properties'].items():
                check(node[key], child, f'{path}.{key}')
        elif not isinstance(node, str) or node not in spec['enum']:
            raise ValueError(f'{path} must be one of: {", ".join(spec["enum"])}')

    check(value, grammar, 'answer')
    return value
