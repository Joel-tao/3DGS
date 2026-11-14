"""Wait for a running training process, then render/evaluate once it succeeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--train-pid', type=int, required=True)
parser.add_argument('--root', type=Path, default=Path('/media/joe/Data/gaussiangraph'))
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
status_path = args.root / 'logs/ramen-status.json'

def status(phase, **extra):
    record = {'phase': phase, 'time_unix': time.time(), **extra}
    temp = status_path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(status_path)
    print(json.dumps(record), flush=True)

def identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rindex(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return fields[19]  # process start time, avoids waiting on reused PIDs
    except FileNotFoundError:
        return None

try:
    started = identity(args.train_pid)
    status('training', train_pid=args.train_pid)
    while started is not None and identity(args.train_pid) == started:
        time.sleep(10)
    log = (args.root / 'logs/ramen-baseline-train.log').read_text(errors='replace')
    if 'Training complete.' not in log:
        raise RuntimeError('Training exited before success; inspect ramen-baseline-train.log')
    env = dict(os.environ, OPENGS_WORK_ROOT=str(args.root))
    for phase in ('render', 'eval'):
        status(phase)
        with (args.root / f'logs/ramen-baseline-{phase}.log').open('w') as handle:
            subprocess.run(['bash', 'scripts/run_ramen.sh', phase], cwd=repo,
                           env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    result = json.loads((args.root / 'outputs/ramen_baseline/metrics.json').read_text())
    status('complete', metrics=result)
except Exception as exc:
    status('failed', error=str(exc))
    raise
