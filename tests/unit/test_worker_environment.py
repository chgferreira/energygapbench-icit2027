import json
import os
from pathlib import Path
import subprocess
import sys

from energygap.methods.worker_environment import isolated_worker_environment


def test_worker_cannot_import_foreign_pythonpath_or_user_site(tmp_path,monkeypatch):
    foreign=tmp_path/'foreign';foreign.mkdir()
    (foreign/'energygap_foreign_marker.py').write_text('foreign = True\n')
    monkeypatch.setenv('PYTHONPATH',str(foreign))
    monkeypatch.setenv('PYTHONHOME','invalid-foreign-interpreter-prefix')
    monkeypatch.setenv('PYTHONUSERBASE',str(tmp_path/'foreign-user'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setenv('ENERGYGAP_NUM_THREADS','32')
    original=dict(os.environ)
    root=Path(__file__).resolve().parents[2]
    env=isolated_worker_environment(root)
    child=subprocess.run([sys.executable,'-c',
        'import importlib.util,json,site; print(json.dumps({"foreign":importlib.util.find_spec("energygap_foreign_marker") is not None,"user_site":site.ENABLE_USER_SITE}))'],
        env=env,cwd=tmp_path,capture_output=True,text=True,check=True)
    assert json.loads(child.stdout)=={'foreign':False,'user_site':False}
    assert env['CUDA_VISIBLE_DEVICES']=='0'
    assert env['ENERGYGAP_NUM_THREADS']=='32'
    assert env['PYTHONPATH']==str(root/'src')
    assert 'PYTHONHOME' not in env and 'PYTHONUSERBASE' not in env
    assert dict(os.environ)==original


