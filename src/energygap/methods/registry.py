"""Explicit capability catalogue. Installation does not imply validation."""
from pathlib import Path
import yaml

REQUIRED={'BRITS','SAITS','CSDI','ImputeFormer','T1','MOMENT-1-base','FIM','TabPFN-TS','MoTM','TS-ICL','CBiT'}


def load_registry(root=Path('.')):
    data=yaml.safe_load((Path(root)/'model_registry.yaml').read_text())
    names={m['name'] for m in data['methods']}
    if not REQUIRED <= names:raise ValueError('Missing required methods: '+str(REQUIRED-names))
    return data


def status_catalogue(root=Path('.')):
    return [dict(name=m['name'],implementation_status=m['implementation_status'],validation_status=m['validation_status'],availability_reason=m['availability_reason'],supports_resume=m['supports_resume']) for m in load_registry(root)['methods']]
