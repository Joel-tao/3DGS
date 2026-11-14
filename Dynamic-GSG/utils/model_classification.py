"""Strict parsing and provenance for model-derived object types."""
import hashlib
import json

TYPES = {'Asset', 'Ordinary', 'Standalone'}


def parse_classification(text):
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[-1].strip() == '```':
            text = '\n'.join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict) or result.get('Object caption') not in TYPES:
        raise ValueError('Model must return a JSON object with a valid Object caption')
    return result['Object caption']


def cache_key(category, model, revision, prompt):
    return hashlib.sha256(json.dumps([category, model, revision, prompt], ensure_ascii=False).encode()).hexdigest()


def model_overrides(nodes, annotations):
    entries = annotations['objects']
    expected = {str(n['id']):n for n in nodes}
    if set(entries) != set(expected):
        raise ValueError('Model annotations must cover exactly the live instance IDs')
    overrides = {}
    for i, node in expected.items():
        entry = entries[i]
        if entry['category'] != node['category']:
            raise ValueError(f'Category mismatch for instance {i}')
        if entry['object_type'] not in TYPES:
            raise ValueError(f'Invalid model type for instance {i}')
        overrides[i] = dict(object_type=entry['object_type'])
    return overrides
