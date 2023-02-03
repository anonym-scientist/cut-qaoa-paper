import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from json_utils import QiskitResultEncoder
from provider_handler import get_backend_and_provider
from circuit_cutting.execute import ExecutorResults

PATHS = [
    'config.json',
    'param_map/executor_results.json',
    'qaoa_execution/[0-9]*/*.json'

]


def get_creation_time(path: Path) -> datetime:
    birth_time = path.stat().st_birthtime
    return datetime.fromtimestamp(birth_time, tz=timezone.utc)  # server internal time was in utc


def get_all_timestamps(path: Path):
    timestamps = set()
    for p in PATHS:
        for file in path.glob(p):
            t = get_creation_time(file)
            timestamps.add(t)
    return timestamps


def date_dict_to_date_time(date_dict):
    if not date_dict['__datetime.datetime__']:
        raise TypeError
    return datetime.fromisoformat(date_dict['iso'])


def get_backend_from_dir(path: Path) -> str:
    config_path = path / 'config.json'
    with open(config_path.resolve()) as f:
        config = json.load(f)
    return config['backend']


def get_backend_and_timestamps_from_dir(exp_path: Path, executor_results=False):
    timestamps = set()
    for run_dir in exp_path.iterdir():
        if not run_dir.is_dir() or run_dir.name.find('simulator') > -1:
            continue
        backend = get_backend_from_dir(run_dir)
        timestamps.update(get_all_timestamps(run_dir))
        if executor_results:
            exec_results_path = run_dir / 'param_map/executor_results.json'
            exec_results = ExecutorResults.load_results(exec_results_path.resolve())
            for res in exec_results.get_all_results():
                timestamps.add(date_dict_to_date_time(res.date))
        return backend, timestamps


def get_backends_and_timestamps(path=None, min_exp=None, max_exp=None, executor_results=False):
    if path is None:
        path = Path('experiment_complete')
    timestamps_dict = defaultdict(set)
    for exp_dir in path.iterdir():
        if not exp_dir.is_dir() or not re.match('^\d+$', exp_dir.name):
            continue
        if min_exp is not None and int(exp_dir.name) < min_exp:
            continue
        if max_exp is not None and int(exp_dir.name) > max_exp:
            continue
        print(exp_dir.name)
        backend, timestamps = get_backend_and_timestamps_from_dir(exp_dir, executor_results)
        timestamps_dict[backend].update(timestamps)
    return timestamps_dict


def fetch_calibration_data(backend_name, timestamps, path=None):
    backend, provider = get_backend_and_provider(backend_name)
    calibrations = {}
    if path is not None:
        backend_path = path / backend_name
        backend_path.mkdir(parents=True, exist_ok=True)
    n = len(timestamps)
    for i, t in enumerate(timestamps):
        print(f'{i} / {n}')
        props = backend.properties(datetime=t)
        update_time = props.last_update_date
        if update_time not in calibrations:
            calibrations[update_time] = props
            if path is not None:
                file_path = backend_path / f'{backend_name}_{update_time.isoformat()}.json'
                with open(file_path.resolve(), "w") as outfile:
                    json.dump(props.to_dict(), outfile, indent=4, cls=QiskitResultEncoder)
    return calibrations


def get_calibration_data(timestamps_dict, path=None):
    calibration_data = {}
    for backend, timestamps in timestamps_dict.items():
        print(f'Fetch {len(timestamps)} timestamps for {backend}')
        calibration_data[backend] = fetch_calibration_data(backend, timestamps, path)
    return calibration_data


def get_all_calibration_data(data_path=None, min_exp=None, max_exp=None, executor_results=False, write_path=None):
    timestamps_dict = get_backends_and_timestamps(path=data_path, min_exp=min_exp, max_exp=max_exp,
                                                  executor_results=executor_results)
    if write_path is None:
        write_path = Path('calibration_data')
    return get_calibration_data(timestamps_dict, write_path)
