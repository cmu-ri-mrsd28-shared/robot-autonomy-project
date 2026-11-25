import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('save_run',Path(__file__).parents[1]/'scripts'/'save_run.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_selects_time_window_and_preserves_originals(tmp_path):
    logs=tmp_path/'logs';logs.mkdir()
    for name in module.FILES:
        (logs/name).write_bytes(b'{"wall_time_s": 10}\n{"wall_time_s": 90}\n{"wall_time_s": 100}\n{"wall_time_s":')
    path,counts=module.save_run('throw-left',minutes=1,log_dir=logs,output_dir=tmp_path/'share',now=100)
    assert counts==dict.fromkeys(module.FILES,2)
    for name in module.FILES:
        assert [json.loads(x)['wall_time_s'] for x in (path/name).read_text().splitlines()]==[90,100]
        assert (logs/name).read_bytes().endswith(b'{"wall_time_s":')
    assert json.loads((path/'metadata.json').read_text())['label']=='throw-left'
    with pytest.raises(FileExistsError):module.save_run('throw-left',log_dir=logs,output_dir=tmp_path/'share',now=100)


def test_validates_label_and_missing_logs(tmp_path):
    with pytest.raises(ValueError):module.save_run('../escape',log_dir=tmp_path)
    with pytest.raises(FileNotFoundError):module.save_run('valid',log_dir=tmp_path)


def test_captures_mock_events_and_configuration(tmp_path):
    logs=tmp_path/'logs';logs.mkdir()
    for name in module.FILES+module.OPTIONAL_FILES:
        (logs/name).write_text('{"wall_time_s": 90, "event": "test"}\n')
    config=tmp_path/'station.yaml';config.write_text('value: 1\n')
    actor=tmp_path/'actor.npz';actor.write_bytes(b'model')
    path,counts=module.save_run('run',minutes=1,log_dir=logs,output_dir=tmp_path/'share',now=100,config_path=config,actor_path=actor)
    assert counts['mock_directions.jsonl']==1
    assert (path/'configuration.yaml').read_text()=='value: 1\n'
    meta=json.loads((path/'metadata.json').read_text())
    assert meta['config_sha256'] and meta['actor_sha256']
