"""Save a labeled time window from the local JSONL logs for sharing with Git.

Example: python3 scripts/save_run.py right-left-test --minutes 3
The files in logs/ are not modified. Run from the project directory.
"""
import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone

FILES = ('model.jsonl', 'adapter.jsonl', 'robot_state.jsonl')
OPTIONAL_FILES = ('mock_directions.jsonl',)
DEFAULT_CONFIG = Path('catchbox/hardware/configuration.yaml')
DEFAULT_ACTOR = Path('artifacts/steady-motion-model/actor.npz')
LABEL = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')


def save_run(label, minutes=5.0, log_dir=Path('logs'), output_dir=Path('shared-runs'), now=None, config_path=DEFAULT_CONFIG, actor_path=DEFAULT_ACTOR):
    if not LABEL.fullmatch(label):
        raise ValueError('Label must use letters, numbers, hyphens or underscores')
    if not math.isfinite(minutes) or not 0 < minutes <= 1440:
        raise ValueError('Minutes must be greater than zero and at most 1440')
    log_dir=Path(log_dir);output_dir=Path(output_dir)
    sources=[log_dir/name for name in FILES]
    missing=[str(p) for p in sources if not p.is_file()]
    if missing:raise FileNotFoundError('Missing log files: '+', '.join(missing))
    sources += [log_dir/name for name in OPTIONAL_FILES if (log_dir/name).is_file()]
    end=float(time.time() if now is None else now); start=end-minutes*60
    output_dir.mkdir(parents=True,exist_ok=True)
    destination=output_dir/label
    if destination.exists():raise FileExistsError('Run already exists: '+str(destination))
    counts={}; summaries={}
    with tempfile.TemporaryDirectory(prefix='.save-run-',dir=str(output_dir)) as temp:
        stage=Path(temp)
        for source in sources:
            count=0;last_row=None
            with source.open('rb') as src, (stage/source.name).open('wb') as dst:
                # Snapshot a fixed byte range so a live writer cannot extend the job.
                remaining=os.fstat(src.fileno()).st_size
                while remaining>0:
                    line=src.readline(remaining);remaining-=len(line)
                    if not line:break
                    if not line.endswith(b'\n'):
                        # A writer may be partway through its last record.
                        break
                    try:
                        row=json.loads(line)
                        stamp=float(row['wall_time_s'])
                    except (ValueError,TypeError,KeyError) as exc:
                        raise ValueError('Invalid log record in '+str(source)) from exc
                    if math.isfinite(stamp) and start<=stamp<=end:
                        dst.write(line);count+=1;last_row=row
            counts[source.name]=count
            summaries[source.name]=last_row.get('dropped') if last_row and last_row.get('event')=='log_summary' else None
        try:
            revision=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
            dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],stderr=subprocess.DEVNULL,text=True).strip())
        except (OSError,subprocess.CalledProcessError):revision=None;dirty=None
        def iso(timestamp):return datetime.fromtimestamp(timestamp,timezone.utc).isoformat()
        config_path=Path(config_path);actor_path=Path(actor_path)
        config_sha=None;actor_sha=None
        if config_path.is_file():
            config_bytes=config_path.read_bytes();(stage/'configuration.yaml').write_bytes(config_bytes)
            config_sha=hashlib.sha256(config_bytes).hexdigest()
        if actor_path.is_file():actor_sha=hashlib.sha256(actor_path.read_bytes()).hexdigest()
        metadata={'label':label,'window_start_utc':iso(start),'window_end_utc':iso(end),
                  'minutes':minutes,'git_commit':revision,'tracked_files_modified':dirty,'records':counts,'dropped_events_by_file':summaries,
                  'config_source':str(config_path),'config_sha256':config_sha,'actor_source':str(actor_path),'actor_sha256':actor_sha,
                  'notes':'Filtered by each record wall_time_s. Clocks must be synchronized across machines; last incomplete line is omitted.'}
        (stage/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        # os.rename refuses to replace an existing nonempty run directory.
        os.rename(str(stage),str(destination))
    return destination,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label',help='Short name such as right-left-test')
    parser.add_argument('--minutes',type=float,default=5.,help='Save the last N minutes (default 5)')
    parser.add_argument('--log-dir',type=Path,default=Path('logs'))
    parser.add_argument('--output-dir',type=Path,default=Path('shared-runs'))
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG,help='Configuration used by the nodes')
    parser.add_argument('--actor',type=Path,default=DEFAULT_ACTOR,help='Actor used by the model node')
    args=parser.parse_args()
    try:destination,counts=save_run(args.label,args.minutes,args.log_dir,args.output_dir,config_path=args.config,actor_path=args.actor)
    except (OSError,ValueError) as exc:parser.error(str(exc))
    print('Saved:',destination)
    for name,count in counts.items():print('  %s: %d records'%(name,count))
    if not all(counts.values()):print('Warning: at least one log has no records in this time window.')
    if not (destination/'configuration.yaml').exists():print('Warning: configuration not found; use --config to provide it.')
    if not args.actor.is_file():print('Warning: actor file not found; use --actor to identify it.')
    if 'mock_directions.jsonl' not in counts:print('Note: no mock-direction phase log found; older runs may not have one.')
    metadata=json.loads((destination/'metadata.json').read_text())
    for name,dropped in metadata['dropped_events_by_file'].items():
        if dropped is None:print('Warning: %s has no final log summary in this window; stop the node to flush and report dropped records.'%name)
        elif dropped:print('Warning: %s reported %d dropped records.'%(name,dropped))
    print('To share: git add %s'%destination)

if __name__=='__main__':main()
