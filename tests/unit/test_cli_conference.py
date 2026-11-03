import json

from energygap.cli import main
from energygap.runtime.state import StateStore


def test_conference_dispatches_full_configuration_and_threads(tmp_path,monkeypatch,capsys):
    import energygap.conference as runner
    calls=[]
    monkeypatch.setattr(runner,'run_conference',lambda **kw:calls.append(kw) or {'status':'completed'})
    monkeypatch.setattr('energygap.runtime.threads.configure_cpu_threads',lambda n:{'requested':n})
    assert main(['--root',str(tmp_path),'run','--profile','conference','--config','full.yaml',
                 '--device','cuda:0','--threads','32','--budget-total-hours','168'])==0
    assert calls[0]['threads']==32 and calls[0]['device']=='cuda:0'
    assert calls[0]['budget_total_hours']==168
    assert str(calls[0]['config_path'])=='full.yaml'
    assert json.loads(capsys.readouterr().out)['status']=='completed'


def test_resume_restores_scientific_settings_device_threads_and_existing_budget(tmp_path,monkeypatch):
    import energygap.conference as runner
    settings={'scientific':'immutable'}
    config={'profile':'conference','experiment_settings':settings,'max_episodes':None,
            'execution':{'threads':32,'device':'cuda:0'}}
    with StateStore(tmp_path/'runs/full',config,budget_seconds=168*3600):pass
    calls=[]
    monkeypatch.setattr(runner,'run_conference',lambda **kw:calls.append(kw) or {'status':'completed'})
    monkeypatch.setattr('energygap.runtime.threads.configure_cpu_threads',lambda n:{'requested':n})
    assert main(['--root',str(tmp_path),'resume','--experiment-id','full'])==0
    assert calls[0]['threads']==32 and calls[0]['device']=='cuda:0'
    assert calls[0]['settings']==settings
    assert calls[0]['budget_total_hours']==168
    assert calls[0]['resume']=='require'


def test_preflight_block_is_nonzero_exit_and_does_not_launch_pilot(tmp_path,monkeypatch):
    import energygap.conference as runner
    monkeypatch.setattr(runner,'run_conference',lambda **kw:{'status':'blocked_preflight','blocked':['missing checkpoint']})
    monkeypatch.setattr('energygap.runtime.threads.configure_cpu_threads',lambda n:{'requested':n})
    assert main(['--root',str(tmp_path),'run','--profile','conference','--device','cpu'])==2


def test_resume_passes_only_explicit_runtime_options(tmp_path,monkeypatch):
    import energygap.conference as runner
    config=dict(profile='conference',experiment_settings={'scientific':'frozen'},max_episodes=None,
                execution=dict(device='cpu',threads=1))
    with StateStore(tmp_path/'runs/kept',config,budget_seconds=86400):pass
    calls=[]
    monkeypatch.setattr(runner,'run_conference',lambda **kw:calls.append(kw) or {'status':'completed'})
    assert main(['--root',str(tmp_path),'resume','--experiment-id','kept',
                 '--energy-cpu-workers','2','--energy-cpu-max-pending','4','--allow-runtime-patch'])==0
    assert calls[0]['energy_cpu_workers']==2 and calls[0]['energy_cpu_max_pending']==4
    assert calls[0]['allow_runtime_patch'] is True and calls[0]['settings']==config['experiment_settings']


def test_runtime_overrides_cannot_be_silently_ignored_by_a_diagnostic_profile(tmp_path):
    assert main(['--root',str(tmp_path),'run','--profile','smoke','--energy-cpu-workers','2'])==2


def test_report_execution_options_preserve_saved_scientific_settings(tmp_path,monkeypatch):
    import energygap.conference as runner
    config=dict(profile='conference',experiment_settings={'scientific':'immutable'},max_episodes=None,
                execution=dict(device='cpu',threads=2))
    path=tmp_path/'runs/reporting'
    with StateStore(path,config,budget_seconds=86400):pass
    calls=[]
    monkeypatch.setattr(runner,'run_conference',lambda **kw:calls.append(kw) or {'status':'completed'})
    assert main(['--root',str(tmp_path),'resume','--experiment-id','reporting',
                 '--report-workers','2','--report-chunk-size','20000',
                 '--report-bootstrap-chunk-size','500','--allow-runtime-patch'])==0
    assert calls[0]['report_workers']==2
    assert calls[0]['report_chunk_size']==20000 and calls[0]['report_bootstrap_chunk_size']==500
    assert calls[0]['settings']==config['experiment_settings']
    with StateStore(path,resume='require',reconcile_on_open=False) as store:
        assert store.get_meta('config')==config
    assert main(['--root',str(tmp_path),'run','--profile','smoke','--report-workers','2'])==2
