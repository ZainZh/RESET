'''Explicit G1 body-role sets for the independent scorer. Built from the body tree, never from name prefixes.

left_/right_ also prefix shoulder and leg bodies, and the model mixes *_Link (left) with *_link (right), so every
set is a descendant set of an explicitly named root body. Pure function of the compiled model: no world builder, no data.

Held-contact groups per hand (frozen G1 anatomical equivalent of the original two-sided finger contact):
  thumb            = descendants of {left_thumb_metacarpal_Link, right_thumb_metacarpal_link}
  opposing_digits  = union of descendants of the index/middle/ring/pinky proximal bodies
Palm (base_link), base2, wrist and arm belong to NO held group; they all stay in the robot-cube exclusion set.
'''
import mujoco

ROBOT_ROOT = 'pelvis'
THUMB_ROOT = {'left': 'left_thumb_metacarpal_Link', 'right': 'right_thumb_metacarpal_link'}
DIGIT_ROOTS = {
    'left': ('left_index_proximal_Link', 'left_middle_proximal_Link', 'left_ring_proximal_Link', 'left_pinky_proximal_Link'),
    'right': ('right_index_proximal_link', 'right_middle_proximal_link', 'right_ring_proximal_link', 'right_pinky_proximal_link'),
}
GROUPS = ('thumb', 'opposing_digits')
ALLOWED_STATIC = ('table',)   # non-robot, non-object bodies tolerated besides world; anything else fails closed
REFERENCE_ROBOT_BODIES = 72   # g1_scene.xml: pelvis (id 1) .. imu_in_pelvis (id 72)


def subtree(m, root):
    '''Body ids of root and all its descendants (MuJoCo parents precede children).'''
    r = int(m.body(root).id)
    out = {r}
    for b in range(r + 1, m.nbody):
        if int(m.body_parentid[b]) in out:
            out.add(b)
    return out


def robot_ids(m):
    return subtree(m, ROBOT_ROOT)


def classify(m, allowed_static=ALLOWED_STATIC):
    '''{body_id: role} with roles world|robot|object|static. Fails closed on any unmapped body.'''
    robot = robot_ids(m)
    out, bad = {}, []
    for b in range(m.nbody):
        n = m.body(b).name or ''
        if b == 0:
            out[b] = 'world'
        elif n.startswith('object_'):
            if b in robot:
                raise ValueError('object body inside robot subtree: ' + n)
            out[b] = 'object'
        elif b in robot:
            out[b] = 'robot'
        elif n in allowed_static:
            out[b] = 'static'
        else:
            bad.append(n or str(b))
    if bad:
        raise ValueError('unmapped non-world non-object bodies (fail closed): ' + ', '.join(bad))
    return out


def hand_groups(m, side):
    '''{body_id: group} for one hand; thumb and opposing_digits are non-empty and disjoint.'''
    thumb = subtree(m, THUMB_ROOT[side])
    digits = set()
    for r in DIGIT_ROOTS[side]:
        digits |= subtree(m, r)
    if not thumb or not digits or thumb & digits:
        raise ValueError('bad hand groups for ' + side)
    robot = robot_ids(m)
    if not (thumb | digits) <= robot:
        raise ValueError('hand group outside robot subtree: ' + side)
    out = {b: 'thumb' for b in thumb}
    out.update({b: 'opposing_digits' for b in digits})
    return out


def roles(m, allowed_static=ALLOWED_STATIC):
    '''Public role summary used by every scorer entry point.'''
    cls = classify(m, allowed_static)
    return {
        'robot': {b for b, r in cls.items() if r == 'robot'},
        'objects': {b for b, r in cls.items() if r == 'object'},
        'groups': {'left': hand_groups(m, 'left'), 'right': hand_groups(m, 'right')},
    }
