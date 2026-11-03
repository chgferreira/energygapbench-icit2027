"""Produce human-readable integration status from the explicit registry."""
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def build():
    registry=yaml.safe_load((ROOT/'model_registry.yaml').read_text())
    lines=['# Model integration status','',
        'Catalogue completeness is distinct from implementation, native validation and scientific completion. No unavailable method is represented by a renamed baseline. See machine-readable model_registry.yaml and artifacts/reports/*audit.json for sources, exact revisions and validation scopes.','',
        '| Method | Implementation | Validation | Remaining blocker |',
        '|---|---|---|---|']
    for m in registry['methods']:
        lines.append(f"| {m['name']} | {m['implementation_status']} | {m['validation_status']} | {m.get('conference_blocker') or m.get('availability_reason') or 'Full-matrix cost/coverage pending'} |")
    lines.extend(['','Native PyPOTS fits preserve upstream losses and have completed-fit/inference checkpoint support only. The separately named saits_protocol_training_loop uses the real upstream SAITS architecture and ORT+MIT gradient loss with explicit balanced point/block masks; CPU optimizer-step recovery, actual SIGTERM/SIGKILL and full-state comparisons are validated separately. Its schedule, validation policy and tested scope do not confer recovery on every adapter.',
        '', 'T1 uses untouched pinned official MIT source, with a bounded constant-LR protocol training loop. D=1/L=512 and tiny overfit/native parity were checked; multivariate dependencies cannot be evaluated with fabricated channels. Native T1 in-progress training recovery remains unsupported.',
        '', 'MOMENT uses frozen official base weights at L=512, patches8 and masked reconstruction. Upstream partially missing patches mask all8 positions; hidden values are zeroed before masked normalization. Pretraining overlap is unknown. Frozen load/inference cost is recorded, local training is not_applicable.',
        '', 'FIM uses the appropriate temporal block extension in its contract, not local FIM-l. Model-card license terms for temporal/pointwise checkpoints remain unresolved despite MIT code. Its safe batch adapter has contract tests, not a reproduced native result.',
        '', 'R24 TabPFN-TS/MoTM sources were inspected, including Fourier features, context-only fit and expected INR weight/config bundles. Missing licensed artifacts remain unavailable. Generic regressors or forecasting calls are not substitutes. TS-ICL native impute runs in its own noncommercial environment with a verified checkpoint and median primary output; marginal quantiles are not trajectories. CBiT has verified paper metadata but no usable author-linked implementation established in this audit.',
        '', 'CPI energy donor adaptation is executable in its own cumulative-energy/interval-mean track and clearly differs from the official daily/Prophet workflow. It does not enter instantaneous P rankings. SADI/TG-MSFM/DANLI remain related-work extensions, not required-eleven integrations.',
        '', 'The complete conference profile is incomplete; code, checkpoint, license, mask/HPO schedule, recovery and cost eligibility must all hold before a method contributes a conference result. A successful two-epoch integrity pilot is not convergence or scientific superiority.',
        ''])
    (ROOT/'docs/model_integration_status.md').write_text('\n'.join(lines))


if __name__=='__main__':build()
