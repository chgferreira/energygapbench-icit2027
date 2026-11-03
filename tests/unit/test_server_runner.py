"""Exercise the supported shell entry point without CUDA or package installation."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def server(tmp_path):
    project = tmp_path / 'project'
    (project / 'scripts').mkdir(parents=True)
    for name in ['server.sh', 'server_dependencies.py', 'validate_release.py', 'restart_server.sh',
                 'run_server_full.sh', 'setup_server.sh']:
        shutil.copyfile(ROOT / 'scripts' / name, project / 'scripts' / name)
    for name in ['configs/experiments/server_full.yaml', 'configs/experiments/server_focused.yaml', 'src/energygap/cli.py',
                 'raw_data/H1.csv', 'raw_data/H2.csv', 'raw_data/I1.csv',
                 'runs/old-study/keep.txt', '.venv-moment/keep.txt']:
        destination = project / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith('configs/'):
            shutil.copyfile(ROOT / name, destination)
        else:
            destination.write_text('preserved\n')
    prefix = tmp_path / 'cuda_env'
    (prefix / 'conda-meta').mkdir(parents=True)
    (prefix / 'conda-meta/history').write_text('existing Conda environment\n')
    executables = tmp_path / 'bin'
    executables.mkdir()
    proc = tmp_path / 'proc'
    proc.mkdir()
    event_path = tmp_path / 'events.jsonl'
    tool_source = f'''#!{sys.executable}
import json
import os
from pathlib import Path
import runpy
import signal
import sys

name = Path(sys.argv[0]).name
with open(os.environ['TEST_EVENTS'], 'a') as stream:
    stream.write(json.dumps({{'tool': name, 'args': sys.argv[1:],
        'threads': os.environ.get('OMP_NUM_THREADS'),
        'offline': os.environ.get('HF_HUB_OFFLINE'),
        'unbuffered': os.environ.get('PYTHONUNBUFFERED')}}) + '\\n')
if name == 'uname':
    print('Linux')
elif name == 'timeout':
    assert sys.argv[1] == '--kill-after=5s'
    os.execvp(sys.argv[3], sys.argv[3:])
elif name == 'flock':
    raise SystemExit(int(os.environ.get('TEST_LOCK_RESULT', '0')))
elif name == 'nvidia-smi':
    print('0, Tesla V100-PCIE-16GB, 550.0, 0 MiB, 16384 MiB')
    raise SystemExit(int(os.environ.get('TEST_DRIVER_RESULT', '0')))
elif name == 'python':
    args = sys.argv[1:]
    if 'scripts/server_dependencies.py' in args:
        if '--environment-only' in args:
            original_iterdir = Path.iterdir
            def test_iterdir(path):
                if str(path) == '/proc':
                    return original_iterdir(Path(os.environ['TEST_PROC']))
                return original_iterdir(path)
            Path.iterdir = test_iterdir
            sys.prefix = os.environ['TEST_ACTUAL_PREFIX']
            sys.argv = args[1:]
            runpy.run_path('scripts/server_dependencies.py', run_name='__main__')
        elif '--install' in args:
            raise SystemExit(int(os.environ.get('TEST_INSTALL_RESULT', '0')))
        else:
            assert '--check' in args
            raise SystemExit(int(os.environ.get('TEST_REQUIREMENTS_RESULT', '0')))
    elif 'scripts/verify_server_models.py' in args:
        print('seven native models verified')
        raise SystemExit(int(os.environ.get('TEST_NATIVE_RESULT', '0')))
    elif 'scripts/validate_release.py' in args:
        if '--required-models' in args or '--required-sites' in args:
            import yaml
            config = yaml.safe_load(Path(args[args.index('--config') + 1]).read_text())
            methods = ('BRITS', 'SAITS', 'CSDI', 'ImputeFormer', 'T1', 'MOMENT-1-base', 'TS-ICL')
            values = [name for name in config['methods'] if name in methods] if '--required-models' in args else config['sites']
            print('\\n'.join(values))
            raise SystemExit(0)
        print('optional validation selected')
        raise SystemExit(int(os.environ.get('TEST_VALIDATION_RESULT', '0')))
    elif any(arg.startswith('scripts/fetch_') for arg in args):
        print('verified pinned model')
    else:
        command = args[args.index('energygap') + 1]
        print(json.dumps({{'command': command, 'startup_trace': os.environ.get('ENERGYGAP_STARTUP_TRACE_SECONDS')}}), flush=True)
        if command == 'doctor':
            raise SystemExit(int(os.environ.get('TEST_DOCTOR_RESULT', '0')))
        if command == 'gpu-smoke':
            raise SystemExit(int(os.environ.get('TEST_SMOKE_RESULT', '0')))
        identity = args[args.index('--experiment-id') + 1]
        if command == 'run':
            Path('runs', identity).mkdir()
            if os.environ.get('TEST_WAIT_FOR_SIGNAL') == '1':
                def safe_stop(number, frame):
                    print('committed-signal-stop', flush=True)
                    raise SystemExit(0)
                signal.signal(signal.SIGINT, safe_stop)
                signal.signal(signal.SIGTERM, safe_stop)
                print('coordinator-ready', flush=True)
                Path('runs', identity, 'ready').write_text('ready')
                while True:
                    signal.pause()
        raise SystemExit(int(os.environ.get('TEST_RUN_RESULT', '0')))
'''
    for tool in ['uname', 'timeout', 'flock', 'nvidia-smi', 'python']:
        executable = executables / tool
        executable.write_text(tool_source)
        executable.chmod(0o755)
    environment = dict(os.environ, PATH=str(executables) + os.pathsep + os.environ['PATH'],
                       CONDA_DEFAULT_ENV='cuda_env', CONDA_PREFIX=str(prefix),
                       TEST_ACTUAL_PREFIX=str(prefix), TEST_EVENTS=str(event_path),
                       TEST_PROC=str(proc))
    return project, proc, event_path, environment


def invoke(server, *arguments, environment=None, script='server.sh'):
    return subprocess.run(['bash', str(server[0] / 'scripts' / script), *arguments],
                          env=environment or server[3], capture_output=True, text=True)


def events(server):
    return [json.loads(line) for line in server[2].read_text().splitlines()]


def launches(server, command):
    return [row for row in events(server) if row['tool'] == 'python'
            and row['args'][:4] == ['-u', '-m', 'energygap', command]]


def test_audited_resume_passes_cpu_scheduling_without_scientific_config_override(server):
    result=invoke(server,'resume','old-study','--energy-cpu-workers','16',
                  '--energy-cpu-max-pending','32','--allow-runtime-patch')
    assert result.returncode==0,result.stdout+result.stderr
    launch,=launches(server,'resume')
    args=launch['args']
    assert args[args.index('--energy-cpu-workers')+1]=='16'
    assert args[args.index('--energy-cpu-max-pending')+1]=='32'
    assert '--allow-runtime-patch' in args and '--config' not in args
    assert '--budget-total-hours' not in args


def test_report_execution_is_forwarded_only_for_run_or_resume(server):
    result=invoke(server,'resume','old-study','--report-workers','16',
                  '--report-chunk-size','20000','--report-bootstrap-chunk-size','1000')
    assert result.returncode==0,result.stdout+result.stderr
    launch,=launches(server,'resume')
    args=launch['args']
    assert args[args.index('--report-workers')+1]=='16'
    assert args[args.index('--report-chunk-size')+1]=='20000'
    assert args[args.index('--report-bootstrap-chunk-size')+1]=='1000'
    assert '--config' not in args
    rejected=invoke(server,'setup','--report-workers','2')
    assert rejected.returncode!=0


def test_default_run_starts_focused_24h_study_with_log_and_preserves_files(server):
    result = invoke(server, 'run')
    assert result.returncode == 0, result.stdout + result.stderr
    launch, = launches(server, 'run')
    args = launch['args']
    identity = args[args.index('--experiment-id') + 1]
    assert identity.startswith('conference-')
    for option, value in [('--profile', 'conference'), ('--config', 'configs/experiments/server_focused.yaml'),
                          ('--resume', 'never'), ('--device', 'cuda:0'), ('--threads', '32'),
                          ('--budget-total-hours', '24'), ('--progress', 'tqdm')]:
        assert args[args.index(option) + 1] == value
    assert launch['threads'] == '32' and launch['offline'] == '1' and launch['unbuffered'] == '1'
    log = (server[0] / 'artifacts/analyses/logs' / f'{identity}.log').read_text()
    assert log.startswith('[1/5]') and '[5/5]' in log and '"startup_trace": "60"' in log
    for name in ['runs/old-study/keep.txt', '.venv-moment/keep.txt', 'raw_data/H1.csv']:
        assert (server[0] / name).read_text() == 'preserved\n'
    assert [row['args'][1] for row in events(server) if row['tool'] == 'timeout'] == ['15s', '15s', '120s']
    again = invoke(server, 'run')
    assert again.returncode == 0, again.stdout + again.stderr
    assert len(list((server[0] / 'runs').glob('conference-*'))) == 2


@pytest.mark.parametrize('preset,downloads,models', [
    ('full', ['scripts/fetch_t1.py', 'scripts/fetch_moment.py', 'scripts/fetch_tsicl.py'], None),
    ('focused', ['scripts/fetch_moment.py'], ['SAITS', 'MOMENT-1-base']),
])
def test_setup_checks_and_prepares_only_selected_models(server, preset, downloads, models):
    environment = dict(server[3], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = invoke(server, 'setup', '--preset', preset, '--threads', '16', environment=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [row for row in events(server) if row['tool'] == 'python']
    installers = [row for row in calls if '--install' in row['args']]
    assert len(installers) == 1 and '--audit-dir' in installers[0]['args']
    assert installers[0]['offline'] is None
    assert [row['args'][1] for row in calls if row['args'][1].startswith('scripts/fetch_')] == downloads
    assert not launches(server, 'run') and not launches(server, 'resume')
    assert len(launches(server, 'doctor')) == len(launches(server, 'gpu-smoke')) == 1
    native_checks = [row for row in calls if 'scripts/verify_server_models.py' in row['args']]
    assert len(native_checks) == 1
    native_args = native_checks[0]['args']
    assert native_args[2:7] == ['--device', 'cuda:0', '--threads', '16', '--output']
    assert native_args[7].endswith('/native_models.json')
    assert native_args[8:] == (['--methods', *models] if models else [])
    assert all(row['threads'] == '16' for row in calls)
    assert not (server[0] / '.venv-tsicl').exists()
    assert not any('pytest' in row['args'] for row in calls)


def test_optional_validation_does_not_install_download_or_launch_experiment(server):
    result = invoke(server, 'validate', 'native', '--threads', '32', '--device', 'cpu')
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [row for row in events(server) if row['tool'] == 'python']
    assert len(calls) == 1
    assert calls[0]['args'] == ['-u', 'scripts/validate_release.py', '--level', 'native',
                               '--threads', '32', '--device', 'cpu', '--config',
                               'configs/experiments/server_focused.yaml']
    assert not launches(server, 'run') and not launches(server, 'resume')


def test_optional_validation_failure_returns_nonzero(server):
    result = invoke(server, 'validate', 'core', environment=dict(server[3], TEST_VALIDATION_RESULT='9'))
    assert result.returncode == 9


@pytest.mark.parametrize(('variable', 'code', 'message'), [
    ('TEST_DOCTOR_RESULT', '7', 'Teste CUDA falhou'),
    ('TEST_DOCTOR_RESULT', '124', 'timeout de 120 s'),
    ('TEST_DRIVER_RESULT', '9', 'Consulta NVIDIA falhou'),
    ('TEST_RUN_RESULT', '6', 'Execução'),
])
def test_failure_status_survives_tee(server, variable, code, message):
    result = invoke(server, 'run', '--experiment-id', 'new-study', environment=dict(server[3], **{variable: code}))
    assert result.returncode == int(code)
    assert message in result.stdout
    assert message in (server[0] / 'artifacts/analyses/logs/new-study.log').read_text()
    if variable != 'TEST_RUN_RESULT':
        assert not launches(server, 'run')


def test_failed_extra_install_never_downloads_or_starts_training(server):
    result = invoke(server, 'setup', environment=dict(server[3], TEST_INSTALL_RESULT='8'))
    assert result.returncode == 8
    assert not any('fetch_' in ' '.join(row['args']) for row in events(server))
    assert not launches(server, 'run')


def test_failed_native_check_never_reports_success_or_starts_full_matrix(server):
    result = invoke(server, 'setup', environment=dict(server[3], TEST_NATIVE_RESULT='12'))
    assert result.returncode == 12
    assert 'Setup concluído' not in result.stdout
    assert not launches(server, 'run')


def test_terminal_interrupt_keeps_log_pipe_open_for_safe_coordinator_stop(server):
    environment = dict(server[3], TEST_WAIT_FOR_SIGNAL='1')
    process = subprocess.Popen(['bash', str(server[0] / 'scripts/server.sh'), 'run',
                                '--experiment-id', 'signal-study'], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    try:
        ready = server[0] / 'runs/signal-study/ready'
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), 'coordinator did not reach its signal boundary'
        os.killpg(process.pid, signal.SIGINT)
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 0, output
        assert 'committed-signal-stop' in output
        assert 'BrokenPipe' not in output
        log = (server[0] / 'artifacts/analyses/logs/signal-study.log').read_text()
        assert 'committed-signal-stop' in log
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


@pytest.mark.parametrize('failure', ['data', 'identity', 'environment', 'prefix', 'metadata'])
def test_preconditions_fail_before_driver_or_training(server, failure):
    environment = dict(server[3])
    if failure == 'data':
        (server[0] / 'raw_data/H2.csv').unlink()
    elif failure == 'identity':
        (server[0] / 'runs/new-study').mkdir()
    elif failure == 'environment':
        environment['CONDA_DEFAULT_ENV'] = 'base'
    elif failure == 'prefix':
        environment['CONDA_PREFIX'] = '/different/cuda_env'
    else:
        (Path(environment['CONDA_PREFIX']) / 'conda-meta/history').unlink()
    result = invoke(server, 'run', '--experiment-id', 'new-study', environment=environment)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not any(row['tool'] == 'nvidia-smi' for row in events(server))
    assert not launches(server, 'run')


def test_activation_by_full_cuda_env_prefix_is_accepted(server):
    environment = dict(server[3], CONDA_DEFAULT_ENV=server[3]['CONDA_PREFIX'])
    result = invoke(server, 'run', environment=environment)
    assert result.returncode == 0, result.stdout + result.stderr


def test_process_guard_rejects_only_this_projects_coordinator_without_killing(server):
    project, proc, _, _ = server
    for pid, cwd, args in [
        ('12345', project, ['python', '-u', '-m', 'energygap', 'resume', '--experiment-id', 'old-study']),
        ('23456', project.parent, ['python', '-m', 'energygap', 'run']),
        ('34567', project, ['python', '-m', 'energygap.methods.frozen_worker']),
    ]:
        entry = proc / pid
        entry.mkdir()
        (entry / 'cwd').symlink_to(cwd, target_is_directory=True)
        (entry / 'cmdline').write_bytes(('\0'.join(args) + '\0').encode())
    result = invoke(server, 'run', '--experiment-id', 'new-study')
    assert result.returncode == 2
    assert '12345' in result.stdout and 'Nenhum processo foi interrompido' in result.stdout
    assert '23456' not in result.stdout and '34567' not in result.stdout
    assert not any(row['tool'] == 'nvidia-smi' for row in events(server))
    assert (project / 'runs/old-study/keep.txt').exists()


def test_project_lock_rejects_second_canonical_launcher(server):
    result = invoke(server, 'run', environment=dict(server[3], TEST_LOCK_RESULT='1'))
    assert result.returncode == 2
    assert not any(row['tool'] == 'python' for row in events(server))


@pytest.mark.parametrize('budget', [None, '240'])
def test_resume_uses_saved_config_and_budget_unless_explicitly_extended(server, budget):
    options = ['resume', 'old-study', '--threads', '16']
    if budget is not None:
        options += ['--budget-hours', budget]
    result = invoke(server, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    launch, = launches(server, 'resume')
    args = launch['args']
    assert args[args.index('--experiment-id') + 1] == 'old-study'
    assert args[args.index('--threads') + 1] == '16'
    assert args[args.index('--progress') + 1] == 'tqdm'
    assert '--config' not in args
    assert ('--budget-total-hours' in args) == (budget is not None)
    if budget:
        assert args[args.index('--budget-total-hours') + 1] == budget


@pytest.mark.parametrize(('script', 'options', 'expected'), [
    ('setup_server.sh', [], 'gpu-smoke'),
    ('restart_server.sh', ['--experiment-id', 'new-study'], 'run'),
    ('run_server_full.sh', ['--experiment-id', 'new-study'], 'run'),
    ('run_server_full.sh', [], 'run'),
    ('run_server_full.sh', ['--resume', 'old-study'], 'resume'),
    ('run_server_full.sh', ['--resume', 'old-study', '--threads', '16'], 'resume'),
])
def test_old_scripts_only_forward_to_supported_entrypoint(server, script, options, expected):
    result = invoke(server, *options, script=script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(launches(server, expected)) == 1
    if script == 'run_server_full.sh' and expected == 'run':
        args = launches(server, 'run')[0]['args']
        assert args[args.index('--config') + 1] == 'configs/experiments/server_full.yaml'
        assert args[args.index('--budget-total-hours') + 1] == '168'


def test_explicit_full_preset_keeps_original_configuration_and_default_budget(server):
    result = invoke(server, 'run', '--preset', 'full')
    assert result.returncode == 0, result.stdout + result.stderr
    args = launches(server, 'run')[0]['args']
    assert args[args.index('--config') + 1] == 'configs/experiments/server_full.yaml'
    assert args[args.index('--budget-total-hours') + 1] == '168'


def test_custom_configuration_path_and_budget_are_passed_without_hidden_preset_overrides(server):
    project = server[0]
    config = yaml.safe_load((project / 'configs/experiments/server_focused.yaml').read_text())
    config['sites'] = ['H1']
    destination = project / 'configs/custom profile.yaml'
    destination.write_text(yaml.safe_dump(config))
    (project / 'raw_data/H2.csv').unlink()
    result = invoke(server, 'run', '--config', str(destination), '--budget-hours', '12')
    assert result.returncode == 0, result.stdout + result.stderr
    args = launches(server, 'run')[0]['args']
    assert args[args.index('--config') + 1] == str(destination)
    assert args[args.index('--budget-total-hours') + 1] == '12'


@pytest.mark.parametrize('options', [
    ['resume', 'old-study', '--preset', 'focused'],
    ['resume', 'old-study', '--preset', 'full'],
    ['resume', 'old-study', '--config', 'configs/experiments/server_focused.yaml'],
    ['run', '--preset', 'full', '--config', 'another.yaml'],
    ['run', '--preset', 'unknown'],
])
def test_resume_and_configuration_conflicts_fail_before_any_process_launch(server, options):
    result = invoke(server, *options)
    assert result.returncode == 2
    assert not server[2].exists()


def test_baseline_only_custom_setup_does_not_fetch_or_check_unselected_native_models(server):
    project = server[0]
    config = yaml.safe_load((project / 'configs/experiments/server_focused.yaml').read_text())
    config['methods'] = ['median', 'linear']
    config['require_available'] = []
    destination = project / 'configs/baselines.yaml'
    destination.write_text(yaml.safe_dump(config))
    result = invoke(server, 'setup', '--config', str(destination))
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [row for row in events(server) if row['tool'] == 'python']
    assert not any('scripts/verify_server_models.py' in row['args'] or
                   any(arg.startswith('scripts/fetch_') for arg in row['args']) for row in calls)
    assert len(launches(server, 'doctor')) == 1
    assert not launches(server, 'gpu-smoke')
