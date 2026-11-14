"""Deterministic, inspectable hierarchy approximation; no LVLM claims."""
import numpy as np

ASSETS = {'chair', 'sofa chair', 'armchair', 'couch', 'sofa', 'table', 'coffee table', 'desk', 'cabinet', 'shelf', 'bookshelf', 'bed', 'dresser'}
STANDALONE = {'stool', 'window', 'door', 'closet door', 'picture', 'curtain', 'mirror', 'ceiling light'}


def build_hierarchy(nodes, *, xy_margin=.08, vertical_tolerance=.15, overrides=None):
    """Nodes must be in a gravity-aligned, Z-up coordinate frame, in meters.

    An ordinary object's center must lie in an asset footprint and its bottom
    must lie within the asset's height range, allowing a small tolerance. This
    approximates attachment, not verified physical contact or containment.
    """
    overrides = overrides or {}
    nodes = [dict(n) for n in nodes]
    for n in nodes:
        override = overrides.get(str(n['id']), {})
        n['category'] = override.get('category', n['category'])
        label = n['category'].lower().strip()
        n['object_type'] = override.get('object_type', 'Asset' if label in ASSETS else 'Standalone' if label in STANDALONE else 'Ordinary')
        if n['object_type'] not in {'Asset', 'Standalone', 'Ordinary'}:
            raise ValueError(f"Invalid object_type for {n['id']}")
        n['type_source'] = 'manual_override' if override else 'category_rules'
    assets = [n for n in nodes if n['object_type'] == 'Asset']
    edges = []
    for n in nodes:
        candidates = []
        center, lo, hi = (np.asarray(n[k]) for k in ['center_m', 'bounds_min_m', 'bounds_max_m'])
        if n['object_type'] == 'Ordinary':
            for a in assets:
                ac, al, ah = (np.asarray(a[k]) for k in ['center_m', 'bounds_min_m', 'bounds_max_m'])
                horizontal = np.all(center[:2] >= al[:2]-xy_margin) and np.all(center[:2] <= ah[:2]+xy_margin)
                vertical = al[2]-vertical_tolerance <= lo[2] <= ah[2]+vertical_tolerance and center[2] >= al[2]
                if horizontal and vertical:
                    score = float(np.linalg.norm(center[:2]-ac[:2]) + .25*abs(lo[2]-ah[2]))
                    candidates.append(dict(parent=a['id'], score=score, bottom_to_parent_top_m=float(lo[2]-ah[2])))
        candidates.sort(key=lambda c: (c['score'], c['parent']))
        n['attachment_candidates'] = candidates
        if candidates:
            chosen = candidates[0]
            n['parent'] = chosen['parent']
            n['attachment_status'] = 'geometric_candidate'
            edge = dict(source=n['id'], target=n['parent'], relation='attached_to', source_method='footprint_and_height_heuristic', verified=False)
        else:
            n['parent'] = 'room'
            n['attachment_status'] = 'unresolved' if n['object_type'] == 'Ordinary' else 'room_member'
            edge = dict(source=n['id'], target='room', relation='in_room', source_method='room_membership', verified=False)
        edges.append(edge)
    return dict(schema_version=1, directed=True, edge_direction='child_to_parent', method='Local category rules and geometric attachment approximation; not paper LVLM inference', nodes=[dict(id='room', category='room', object_type='Room', parent=None)]+nodes, edges=edges, parameters=dict(xy_margin_m=xy_margin, vertical_tolerance_m=vertical_tolerance))


def query_hierarchy(graph, category=None, parent_category=None, object_id=None, parent_id=None):
    """Exact category matching plus a small documented alias set."""
    aliases = {'sofa': 'couch', 'sofa chair': 'chair', 'armchair': 'chair'}
    def match(label, query):
        return query is None or aliases.get(label.lower(), label.lower()) == aliases.get(query.lower(), query.lower())
    lookup = {n['id']: n for n in graph['nodes']}
    results = []
    for n in graph['nodes']:
        if n['id'] == 'room' or (object_id is not None and n['id'] != object_id) or not match(n['category'], category):
            continue
        parent = lookup[n['parent']]
        if (parent_id is not None and parent['id'] != parent_id) or not match(parent['category'], parent_category):
            continue
        results.append(dict(id=n['id'], category=n['category'], object_type=n['object_type'], parent_id=parent['id'], parent_category=parent['category'], attachment_status=n['attachment_status']))
    return results
