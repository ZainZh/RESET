"""Immutable campaign inputs, paired adoption, and a durable independent budget.

Preparation never steps physics. An attempt is an fsynced reservation, including
attempts killed before producing a result. There is deliberately no retry flag.
"""
import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('collect_apple', 'collect_cubes', 'stack_cups')
SIM_CAPS = dict(zip(TASKS, (30., 90., 60.)))
OBJECTS = {'collect_apple': ('apple', 'bowl'),
           'collect_cubes': ('bowl', 'yellow_cube', 'red_cube_left', 'blue_cube', 'orange_cube', 'red_cube_right'),
           'stack_cups': ('yellow_cup', 'purple_cup', 'blue_cup')}
SPLITS = {'DEV': (101, 102, 103), 'dev3': (201, 202, 203),
          'VAL': (301, 302, 303), 'TEST': (901, 902, 903, 904, 905, 906)}
OFFSETS = dict(zip(TASKS, (10000, 20000, 30000)))
PROTOCOL = {'id': 'practice_20261031', 'task_cap': 30, 'total_cap': 90,
            'wall_cap_sec': 300., 'sim_caps': SIM_CAPS, 'dwell_sec': .5,
            'sample_sec': .05, 'versions': ['v0', 'v1'], 'splits': SPLITS}


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    def default(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=default) + '\n').encode()


def digest_json(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    # Persist the directory entry as well as the file contents.
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    return json.loads(Path(path).read_text())


def hash_files(paths):
    result = {}
    for raw in sorted(set(map(str, paths))):
        path = Path(raw).resolve(strict=True)
        require(path.is_file(), f'input is not a file: {path}')
        result[str(path)] = sha256(path)
    return result


def verify_files(files):
    for path, expected in files.items():
        require(Path(path).is_file() and sha256(path) == expected, f'frozen input changed or missing: {path}')


def scene_dependencies(scene):
    """Resolve all XML assets; unsupported includes fail rather than escape freeze."""
    scene = Path(scene).resolve(strict=True)
    tree = ET.parse(scene).getroot()
    require(not tree.findall('.//include'), 'scene includes require an explicit dependency resolver')
    compiler = tree.find('compiler')
    paths = [scene]
    for node in tree.iter():
        if node.get('file'):
            path = Path(node.get('file'))
            if not path.is_absolute():
                key = 'texturedir' if node.tag == 'texture' else 'meshdir'
                prefix = compiler.get(key, '') if compiler is not None else ''
                path = scene.parent / prefix / path
            paths.append(path.resolve(strict=True))
    return paths


def validate_layouts(task, manifest):
    require(manifest.get('task') == task, 'layout task mismatch')
    entries = manifest['layouts']
    expected = {f'{split}_{i}': (split, i, scale) for split, ids in SPLITS.items()
                for i, scale in zip(ids, (.4, 1., 1.6) if split != 'TEST' else (.4, .4, 1., 1., 1.6, 1.6))}
    require(len(entries) == 15 and {x['id'] for x in entries} == set(expected), 'missing/duplicate layout IDs')
    for item in entries:
        split, seed_id, scale = expected[item['id']]
        require(item['split'] == split and item['seed'] == OFFSETS[task] + seed_id
                and item['scale'] == scale, f'layout protocol mismatch: {item["id"]}')
        require(set(item.get('poses', {})) == {'object_' + name for name in OBJECTS[task]},
                'layout does not include every required task object')
        require(item.get('validity', {}).get('valid') is True, 'layout lacks positive geometric validity evidence')
        require(1 <= item.get('attempt', 0) <= 200, 'layout rejection sampling attempt missing/out of range')
        for pose in item['poses'].values():
            require(len(pose) == 7 and all(math.isfinite(float(x)) for x in pose), 'invalid layout pose')
            require(abs(sum(float(x)**2 for x in pose[3:]) - 1.) < 1e-6, 'invalid layout quaternion')
    return entries


def audit_program(path):
    """Enforce the declared policy dependency boundary; not an OS sandbox."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    banned = {'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals',
              'getattr', 'setattr', 'vars', 'input', 'breakpoint'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or '']
            require(all(n.split('.')[0] in ('numpy', 'math', '__future__') for n in names),
                    f'candidate imports outside numpy/math: {names}')
            require(not getattr(node, 'level', 0), 'relative candidate imports are unsupported')
        if isinstance(node, ast.Name):
            require(node.id not in banned, f'candidate uses forbidden builtin: {node.id}')
        if isinstance(node, ast.Attribute):
            require(not node.attr.startswith('_'), 'candidate uses private attributes')
            require(node.attr not in ('load', 'save', 'savez', 'savez_compressed', 'fromfile', 'tofile',
                                      'memmap', 'loadtxt', 'savetxt', 'genfromtxt', 'ctypeslib'),
                    f'candidate filesystem/native access: {node.attr}')
    runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    require(len(runs) == 1, 'program.py must export synchronous run(io, params)')
    args = runs[0].args
    require([a.arg for a in args.posonlyargs + args.args] == ['io', 'params']
            and not args.vararg and not args.kwarg and not args.kwonlyargs,
            'candidate signature must be run(io, params)')


def candidate_manifest(directory):
    directory = Path(directory).resolve(strict=True)
    children = list(directory.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in children), 'candidate must contain ordinary files only')
    require({p.name for p in children} <= {'program.py', 'params.json', 'CANDIDATE.json'},
            'candidate closure is program.py + params.json; inline helpers')
    require((directory / 'params.json').is_file(), 'candidate params missing')
    audit_program(directory / 'program.py')
    require(isinstance(read_json(directory / 'params.json'), dict), 'candidate params must be a JSON object')
    return {name: sha256(directory / name) for name in ('program.py', 'params.json')}


def freeze_candidate(source, target, task, version, evidence=None):
    require(task in TASKS and version in ('v0', 'v1'), 'invalid task/candidate')
    source, target = Path(source), Path(target)
    files = candidate_manifest(source)
    target.mkdir(parents=True, exist_ok=False)
    for name in files:
        shutil.copyfile(source / name, target / name)
    require(candidate_manifest(target) == files, 'candidate copy changed')
    record = dict(task=task, version=version, files=files, candidate_sha256=digest_json(files),
                  evidence=evidence, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_new(target / 'CANDIDATE.json', record)
    for path in target.iterdir():
        path.chmod(0o444)
    return record


def verify_candidate(directory, task=None):
    record = read_json(Path(directory) / 'CANDIDATE.json')
    require(task is None or record['task'] == task, 'candidate task mismatch')
    require(candidate_manifest(directory) == record['files'], 'candidate source changed')
    require(record['candidate_sha256'] == digest_json(record['files']), 'invalid candidate manifest')
    return record


def _priors_params(config, prior):
    params = dict(config)
    params['priors'] = prior
    return params


def prepare(output):
    """Generate all layouts and offline priors; no physics stepping or perception."""
    from . import layouts, priors
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = {}
    for task in TASKS:
        home = output / 'tasks' / task
        home.mkdir(parents=True)
        manifest = layouts.make_layouts(task, home / 'layouts')
        validate_layouts(task, manifest)
        write_new(home / 'LAYOUTS.json', manifest)
        prior = priors.build(task, home / 'priors')
        config = read_json(ROOT / 'configs/practice' / f'{task}.json')
        staging = home / 'initial_source'
        staging.mkdir()
        shutil.copyfile(ROOT / 'twin/practice/tasks' / f'{task}.py', staging / 'program.py')
        write_new(staging / 'params.json', _priors_params(config, prior))
        freeze_candidate(staging, home / 'candidates/v0', task, 'v0')
        tasks[task] = {'layouts': str(home / 'LAYOUTS.json'), 'home': str(home),
                       'scene': str(ROOT / 'runs/camera_reference_20261025/review_v2/scenes' / task / 'scene.xml'),
                       'record': str(ROOT / 'runs/physics_v2' / task / 'reconstruction.json'),
                       'reference': str(ROOT / 'runs/physics_v2' / task / 'reference.npz')}
    prepared = dict(protocol=PROTOCOL, tasks=tasks, campaign=str(output))
    write_new(output / 'PREPARED.json', prepared)
    return prepared


def _referenced_paths(value):
    """Freeze named file provenance recursively, never load demonstration trajectories."""
    result = []
    if isinstance(value, dict):
        for item in value.values():
            result.extend(_referenced_paths(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_referenced_paths(item))
    elif isinstance(value, str) and Path(value).is_absolute() and Path(value).is_file():
        result.append(Path(value))
    return result


def verify_provenance(value):
    """Validate conventional source-hash records before sealing derived inputs."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            path = Path(value['path'])
            if not path.is_absolute():
                path = ROOT / path
            require(path.is_file() and sha256(path) == value['sha256'], f'derived source changed: {path}')
        hashes = value.get('inputs_sha256')
        if isinstance(hashes, dict):
            for name, expected in hashes.items():
                path = Path(name)
                if not path.is_absolute():
                    path = ROOT / path
                require(path.is_file() and sha256(path) == expected, f'layout source changed: {path}')
        for child in value.values():
            verify_provenance(child)
    elif isinstance(value, list):
        for child in value:
            verify_provenance(child)


def freeze(campaign, approval):
    """Explicit root acceptance record is attached before formal inputs can run."""
    campaign, approval = Path(campaign).resolve(), Path(approval).resolve(strict=True)
    require(not (campaign / 'INPUTS.json').exists(), 'campaign already frozen')
    prepared = read_json(campaign / 'PREPARED.json')
    require(prepared['protocol'] == json.loads(json_bytes(PROTOCOL)), 'prepared protocol changed')
    paths = [approval, campaign / 'PREPARED.json', ROOT / 'configs/practice/scoring.json',
             ROOT / 'docs/route/THREE_TASK_PRACTICE.md', ROOT / 'docs/route/PRACTICE_INTERFACE.md',
             ROOT / 'configs/one_shot.json']
    required_modules = ('__init__.py', 'runtime.py', 'motion.py', 'engine.py', 'layouts.py', 'scoring.py',
                        'perception.py', 'sam_worker.py', 'priors.py', 'tasks/__init__.py',
                        *(f'tasks/{task}.py' for task in TASKS))
    paths.extend(ROOT / 'twin/practice' / name for name in required_modules)
    paths.extend((ROOT / 'twin').rglob('*.py'))
    paths.append(ROOT / 'scripts/run_practice.py')
    for task, item in prepared['tasks'].items():
        layout_manifest = read_json(item['layouts'])
        validate_layouts(task, layout_manifest)
        verify_provenance(layout_manifest)
        verify_provenance(read_json(Path(item['home']) / 'candidates/v0/params.json'))
        verify_candidate(Path(item['home']) / 'candidates/v0', task)
        paths.extend(scene_dependencies(item['scene']))
        paths.extend(Path(item[key]) for key in ('record', 'reference', 'layouts'))
        paths.extend(p for p in (Path(item['home']) / 'priors').rglob('*') if p.is_file())
        paths.extend(p for p in (Path(item['home']) / 'layouts').rglob('*') if p.is_file())
        paths.extend((Path(item['home']) / 'candidates/v0').iterdir())
        paths.extend(_referenced_paths(read_json(Path(item['home']) / 'candidates/v0/params.json')))
        for key in ('record', 'layouts'):
            paths.extend(_referenced_paths(read_json(item[key])))
        paths.append(ROOT / 'configs/practice' / f'{task}.json')
    report_path = ROOT / 'runs/camera_reference_20261025/review_v2/camera_export_report.json'
    report = read_json(report_path)
    paths.append(report_path)
    verify_provenance(report.get('reference', {}))
    paths.extend(_referenced_paths(report.get('reference', {})))
    for task, item in prepared['tasks'].items():
        exported = report['tasks'][task]['export']
        require(Path(exported['output_xml']).resolve() == Path(item['scene']).resolve()
                and sha256(item['scene']) == exported['output_sha256'], 'unaccepted task camera export')
        source = Path(exported['source_xml'])
        require(sha256(source) == exported['source_sha256_before'] == exported['source_sha256_after'],
                'accepted camera export source changed')
        paths.extend(_referenced_paths(exported))
    # Freeze the actual configured SAM interpreter, checkpoint, configs and source.
    segmentation = read_json(ROOT / 'configs/one_shot.json')['segmentation']
    for key in ('python', 'checkpoint'):
        paths.append(Path(segmentation[key]))
    sam_source = Path(segmentation['source'])
    require(sam_source.is_dir(), 'SAM source missing')
    sam_paths = list(sam_source.rglob('*.py')) + list(sam_source.rglob('*.yaml'))
    require(bool(sam_paths), 'SAM code/config closure missing')
    model_config = segmentation['model_config']
    config_name = Path(model_config).name
    require(any(p.name == config_name for p in sam_paths), 'configured SAM model YAML missing from source closure')
    paths.extend(sam_paths)
    budget_root = ROOT / 'runs/practice_20261031/budget'
    budget_root.mkdir(parents=True, exist_ok=True)
    frozen = {**prepared, 'budget_root': str(budget_root), 'approval': str(approval), 'files': hash_files(paths),
              'environment': environment(), 'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    frozen['inputs_sha256'] = digest_json(frozen)
    write_new(campaign / 'INPUTS.json', frozen)
    return frozen


def environment():
    return {'python': sys.version, 'executable_sha256': sha256(sys.executable),
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'mujoco')}}


def load_inputs(campaign):
    frozen = read_json(Path(campaign) / 'INPUTS.json')
    checksum = frozen.pop('inputs_sha256')
    require(digest_json(frozen) == checksum, 'input manifest changed')
    frozen['inputs_sha256'] = checksum
    require(frozen['protocol'] == json.loads(json_bytes(PROTOCOL)), 'campaign protocol mismatch')
    require(frozen['budget_root'] == str(ROOT / 'runs/practice_20261031/budget'), 'campaign budget path changed')
    require(frozen['environment'] == environment(), 'runtime environment changed')
    verify_files(frozen['files'])
    # Added source files can alter import resolution too.
    current = {str(p.resolve()) for p in (ROOT / 'twin').rglob('*.py')}
    require(current <= set(frozen['files']), 'shared source closure grew after freeze')
    return frozen


def budget_root(campaign):
    path = Path(campaign) / 'INPUTS.json'
    return Path(read_json(path)['budget_root']) if path.is_file() else Path(campaign)


@contextmanager
def campaign_lock(campaign, name='ledger'):
    path = (budget_root(campaign) if name == 'ledger' else Path(campaign)) / f'.{name}.lock'
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def attempts(campaign):
    path = budget_root(campaign) / 'attempts.jsonl'
    if not path.exists():
        return []
    # A torn record fails closed; never guess away a reserved attempt.
    result = [json.loads(line) for line in path.read_text().splitlines()]
    require(all(x.get('event') == 'reserved' for x in result), 'invalid ledger')
    return result


def selected(campaign, task):
    path = Path(campaign) / 'tasks' / task / 'SELECTION.json'
    require(path.is_file(), 'TEST sealed: finalize selection first')
    selection = read_json(path)
    seal = read_json(path.with_name('SELECTION_SEAL.json'))
    require(sha256(path) == seal['selection_sha256'], 'final selection artifact changed')
    return selection


def reserve_attempt(task, candidate_dir, layout, output, frozen):
    """Called by run_episode BEFORE initializing or stepping a task world."""
    require(task in TASKS, 'unknown task')
    campaign = Path(frozen['campaign'])
    candidate = verify_candidate(candidate_dir, task)
    version = candidate['version']
    expected_dir = Path(frozen['tasks'][task]['home']) / 'candidates' / version
    require(Path(candidate_dir).resolve() == expected_dir.resolve(), 'candidate is not registered in this campaign')
    manifest = read_json(frozen['tasks'][task]['layouts'])
    require(any(x == layout for x in manifest['layouts']), 'layout is not a frozen campaign entry')
    with campaign_lock(campaign):
        rows = attempts(campaign)
        if layout['split'] != 'TEST':
            require(not any(x['task'] == task and x.get('split') == 'TEST' for x in rows),
                    'TEST has been opened for this task; development is sealed across output directories')
        if layout['split'] == 'TEST':
            selection = selected(campaign, task)
            require(selection['inputs_sha256'] == frozen['inputs_sha256'], 'selection belongs to another input version')
            require(version in ('v0', selection['selected']), 'candidate was not selected for paired TEST')
            require(selection['candidate_hashes'][version] == candidate['candidate_sha256'], 'selected candidate changed')
        else:
            require(not (Path(frozen['tasks'][task]['home']) / 'SELECTION.json').exists(), 'selection finalized; tuning sealed')
        require(len(rows) < 90 and sum(x['task'] == task for x in rows) < 30, 'episode budget exhausted')
        require(not any(x['task'] == task and x['version'] == version and x['layout_id'] == layout['id'] for x in rows),
                'attempt already reserved; failures and missing results cannot be retried')
        out = Path(output).resolve()
        out.mkdir(parents=True, exist_ok=False)
        row = dict(event='reserved', task=task, version=version, layout_id=layout['id'], split=layout['split'],
                   candidate_sha256=candidate['candidate_sha256'], inputs_sha256=frozen['inputs_sha256'],
                   output=str(out), reserved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        with (budget_root(campaign) / 'attempts.jsonl').open('ab') as stream:
            stream.write(json_bytes(row).replace(b'\n', b'') + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        ledger_fd = os.open(budget_root(campaign), os.O_RDONLY)
        try:
            os.fsync(ledger_fd)
        finally:
            os.close(ledger_fd)
        write_new(out / 'ATTEMPT.json', row)
    return row


def results(campaign, task, version, split):
    rows = [x for x in attempts(campaign) if x['task'] == task and x['version'] == version and x['split'] == split]
    indexed = {x['layout_id']: x for x in rows}
    answer = []
    for i in SPLITS[split]:
        ident = f'{split}_{i}'
        row = indexed.get(ident)
        path = Path(row['output']) / 'result.json' if row else None
        if path is None or not path.is_file():
            answer.append(dict(layout_id=ident, success=False, status='missing', reserved=row is not None,
                               reason='reserved without result' if row else 'not attempted'))
            continue
        override = path.parent / 'watchdog_failure.json'
        result = read_json(override if override.is_file() else path)
        manifest_path = Path(campaign) / 'INPUTS.json'
        if manifest_path.is_file() and row['inputs_sha256'] != read_json(manifest_path)['inputs_sha256']:
            answer.append(dict(layout_id=ident, success=False, status='missing', reserved=True,
                               reason='reserved under another engine/input version; incompatible evidence retained'))
            continue
        require(result['candidate_sha256'] == row['candidate_sha256'] and
                result['inputs_sha256'] == row['inputs_sha256'] and result['layout_id'] == ident,
                'result provenance mismatch')
        answer.append(result)
    return answer


def summarize(campaign, task):
    frozen = load_inputs(campaign)
    home = Path(frozen['tasks'][task]['home'])
    candidates = [v for v in ('v0', 'v1') if (home / 'candidates' / v).exists()]
    totals, rows, hashes = {}, {}, {}
    for version in candidates:
        hashes[version] = verify_candidate(home / 'candidates' / version, task)['candidate_sha256']
        rows[version] = {s: results(campaign, task, version, s) for s in ('DEV', 'dev3', 'VAL')}
        totals[version] = {s: sum(r.get('success') is True for r in rr) for s, rr in rows[version].items()}
    complete = {v: all(r['status'] != 'missing' for rr in rows[v].values() for r in rr) for v in candidates}
    choice, reason = 'v0', 'initial candidate retained'
    if 'v1' in candidates and all(complete.values()):
        no_regression = all(totals['v1'][s] >= totals['v0'][s] for s in totals['v0'])
        improvement = sum(totals['v1'].values()) > sum(totals['v0'].values())
        new_invalid = any(b.get('numerical_invalid', False) is not False and a.get('numerical_invalid', False) is False
                          for s in totals['v0'] for a, b in zip(rows['v0'][s], rows['v1'][s]))
        if no_regression and improvement and not new_invalid:
            choice, reason = 'v1', 'more full-task successes, no split regression, no new numerical invalidity'
    answer = dict(task=task, selected=choice, reason=reason, totals=totals, complete=complete,
                  candidate_hashes=hashes, inputs_sha256=frozen['inputs_sha256'], rows=rows)
    if (home / 'SELECTION.json').is_file():
        selection = selected(campaign, task)
        answer['selected'] = selection['selected']
        initial = results(campaign, task, 'v0', 'TEST')
        final = initial if selection['selected'] == 'v0' else results(campaign, task, selection['selected'], 'TEST')
        answer['paired_test'] = {'initial': initial, 'selected': final,
                                 'reuse_identical_v0': selection['selected'] == 'v0'}
    return answer


def finalize(campaign, task):
    with campaign_lock(campaign):
        summary = summarize(campaign, task)
        require(all(summary['complete'].values()), 'selection episodes incomplete; missing results are reported, not discarded')
        home = Path(campaign) / 'tasks' / task
        write_new(home / 'SELECTION.json', summary)
        write_new(home / 'SELECTION_SEAL.json', {'selection_sha256': sha256(home / 'SELECTION.json')})
        (home / 'SELECTION.json').chmod(0o444)
        (home / 'SELECTION_SEAL.json').chmod(0o444)
    return summary


def add_candidate(campaign, task, source, evidence):
    frozen = load_inputs(campaign)
    home = Path(frozen['tasks'][task]['home'])
    with campaign_lock(campaign):
        require(not (home / 'SELECTION.json').exists(), 'candidate revisions sealed after selection')
        rows = attempts(campaign)
        require(not any(x['task'] == task and x.get('split') == 'TEST' for x in rows),
                'candidate revisions sealed after TEST, including across output directories')
        require(any(x['task'] == task and x['version'] == 'v0'
                    and (Path(x['output']) / 'result.json').is_file() for x in rows),
                'v1 requires completed v0 episode evidence')
        evidence_path = Path(evidence).resolve(strict=True)
        require(bool(evidence_path.read_text().strip()), 'v1 evidence explanation is empty')
        record = {'path': str(evidence_path), 'sha256': sha256(evidence_path)}
        return freeze_candidate(source, home / 'candidates/v1', task, 'v1', record)


def execute_split(campaign, task, version, split):
    from .runtime import run_episode
    frozen = load_inputs(campaign)
    home = Path(frozen['tasks'][task]['home'])
    manifest = read_json(frozen['tasks'][task]['layouts'])
    outcomes = []
    for layout in manifest['layouts']:
        if layout['split'] != split:
            continue
        # Reserved attempts are never rerun, even if their process was killed.
        if any(x['task'] == task and x['version'] == version and x['layout_id'] == layout['id']
               for x in attempts(campaign)):
            continue
        outcomes.append(run_episode(task, home / 'candidates' / version, layout,
                                    home / 'episodes' / version / layout['id'], frozen))
    return {'executed': outcomes, 'results': results(campaign, task, version, split)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('freeze'); p.add_argument('--campaign', type=Path, required=True); p.add_argument('--approval', type=Path, required=True)
    p = sub.add_parser('candidate'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    for command in ('run', 'summarize', 'finalize'):
        p = sub.add_parser(command); p.add_argument('--campaign', type=Path, required=True)
        p.add_argument('--task', choices=TASKS, required=True)
        if command == 'run':
            p.add_argument('--candidate', choices=('v0', 'v1'), required=True)
            p.add_argument('--split', choices=tuple(SPLITS), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.output)
        elif args.command == 'freeze':
            result = freeze(args.campaign, args.approval)
        elif args.command == 'candidate':
            result = add_candidate(args.campaign, args.task, args.source, args.evidence)
        elif args.command == 'run':
            result = execute_split(args.campaign, args.task, args.candidate, args.split)
        elif args.command == 'summarize':
            result = summarize(args.campaign, args.task)
        else:
            result = finalize(args.campaign, args.task)
        print(json_bytes(result).decode(), end='')
    except (RuntimeError, OSError, ValueError, ImportError, KeyError) as exc:
        print(json.dumps({'status': 'blocked', 'reason': f'{type(exc).__name__}: {exc}'}), file=sys.stderr)
        return 2
    return 0
