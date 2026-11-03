from pathlib import Path
import yaml
from energygap.methods.registry import load_registry,REQUIRED


def test_full_eleven_method_catalogue_has_truthful_status():
    root=Path(__file__).resolve().parents[2]
    data=load_registry(root)
    assert REQUIRED<={m['name'] for m in data['methods']}
    assert data['conference_complete'] is False
    for model in data['methods']:
        assert model['implementation_status'] in ('integrated','unavailable','unsupported','not_reproduced')
        assert model['validation_status']
        if model['implementation_status']!='integrated':
            assert model['availability_reason']
            assert not model['conference_eligible']
        assert (root/'configs/methods'/f'{model["name"].lower().replace("-","_")}.yaml').exists()


def test_release_allowlist_excludes_private_data(tmp_path):
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('release',root/'scripts/package_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'README.md').write_text('Synthetic package test')
    (tmp_path/'raw_data').mkdir();(tmp_path/'raw_data/private.csv').write_text('secret')
    result=module.build(tmp_path)
    import tarfile
    with tarfile.open(tmp_path/result['archive']) as archive:
        assert not any('private' in p for p in archive.getnames())
        assert 'energygapbench/MANIFEST.sha256' in archive.getnames()
