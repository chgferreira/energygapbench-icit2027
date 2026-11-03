"""Record actual local delivery evidence, keeping scientific incompleteness explicit."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import json
import re
import sqlite3
import yaml

ROOT=Path(__file__).resolve().parents[1]


def read_json(path):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else {}


def run_status(identifier):
    path=ROOT/'runs'/identifier/'state.sqlite'
    if not path.exists():return {'experiment_id':identifier,'status':'not_present'}
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
    progress=db.execute('SELECT SUM(completed_units),SUM(total_units),COUNT(*) FROM jobs').fetchone()
    count=db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    checkpoint=db.execute('SELECT path,step FROM checkpoints ORDER BY created_at DESC LIMIT 1').fetchone()
    db.close()
    return {'experiment_id':identifier,'status':meta['status'],'fingerprint':meta['fingerprint'],
            'source_sha256':meta.get('config',{}).get('code_hash'),
            'confirmed_units':progress[0],'total_units':progress[1],'jobs':progress[2],'result_rows':count,
            'recorded_active_seconds':meta.get('active_seconds'),'budget_seconds':meta.get('budget_seconds'),
            'last_checkpoint':{'path':checkpoint[0],'step':checkpoint[1]} if checkpoint else None,
            'state':str(path.relative_to(ROOT))}


def build(test_log,clean_log):
    import sys
    from energygap.pipeline import code_hash
    registry=yaml.safe_load((ROOT/'model_registry.yaml').read_text())
    plan=read_json('artifacts/reports/conference_plan.json')
    paired=read_json('artifacts/reports/paired_integrity_pilot.json')
    tsicl=read_json('artifacts/reports/tsicl_real_pilot.json')
    explore=read_json('artifacts/reports/exploratory_energy_summary.json')
    smoke=next((p.parent.name for p in sorted((ROOT/'runs').glob('smoke-*/state.sqlite'),key=lambda p:p.stat().st_mtime,reverse=True)
                if run_status(p.parent.name).get('source_sha256')==code_hash()),None)
    ids=[smoke,'pilot-083764a78be2ffec','real-models-pilot-recovery',tsicl.get('experiment_id'),paired.get('experiment_id'),explore.get('experiment_id')]
    def test_result(path):
        p=Path(path);lines=p.read_text().splitlines() if p.exists() else []
        return {'command_log':str(path),'result':next((s for s in reversed(lines) if re.search(r'\d+ passed',s)),'not_recorded')}
    data={'generated_at':datetime.now(timezone.utc).isoformat(),'source_sha256':code_hash(),
          'implementation_complete_against_entire_request':False,'conference_experiments_complete':False,
          'physical_confirmatory_complete':False,'tests':[test_result(test_log),test_result(clean_log)],
          'integrated_catalogue_methods':[m['name'] for m in registry['methods'] if m['implementation_status']=='integrated'],
          'unavailable_catalogue_methods':[{'method':m['name'],'reason':m['availability_reason']} for m in registry['methods'] if m['implementation_status']!='integrated'],
          'runs':[run_status(i) for i in ids if i],
          'paired_pilot':{k:paired.get(k) for k in ['experiment_id','n_methods','n_queries','n_reconstruction_rows','n_forecast_rows','same_support_across_methods','confirmatory']},
          'conference_plan':{k:plan.get(k) for k in ['total_jobs','counts','estimated_seconds','estimated_finish','can_start_full_matrix','training_counts']},
          'publication':{'published':False,'pushed':False,'DOI':None,'real_data_license':'not_granted'},
          'important_limits':['Physical measurement semantics unresolved; hypotheses exploratory only.',
                             'Full conference executor/HPO/heavy training recovery and all requested controls incomplete.',
                             'Local CPU/POSIX only; no universal GPU/distributed guarantee.',
                             'Recorded active job time does not fully cover all preparation outside the ledger.',
                             'Legacy interrupted smoke-b44ba1339a100d1d is superseded by a new immutable identity; not resumed across code changes.']}
    (ROOT/'artifacts/reports/delivery.json').write_text(json.dumps(data,indent=2))
    lines=['# Entrega local do EnergyGapBench','',
           'O projeto é executável e os pilotos abaixo foram realizados. **A implementação integral da especificação e o estudo conference ainda não estão concluídos.** Os limites não são substituídos por resultados sintéticos ou métodos renomeados.','',
           '## Verificações executadas','']
    lines += [f"- {t['result']} — `{t['command_log']}`." for t in data['tests']]
    lines += ['','A reprodução mínima usa o pacote fonte extraído sem raw_data, runs, dados processados, pesos ou ambientes opcionais. Os hashes dos três CSV originais foram reconferidos; nenhum original foi modificado.','',
              '## Execuções e estado','',
              '| Experimento | Estado | Unidades confirmadas | Registros | Trabalho ativo registrado |',
              '|---|---|---:|---:|---:|']
    for r in data['runs']:
        lines.append(f"| {r['experiment_id']} | {r['status']} | {r.get('confirmed_units')}/{r.get('total_units')} | {r.get('result_rows')} | {r.get('recorded_active_seconds',0):.3f} s |")
    lines += ['','O piloto básico real tem 648 linhas de reconstrução e 324 previsões válidas. O suplemento pareado usa 13 métodos nas mesmas 18 consultas: 234 reconstruções e 117 previsões de uma hora, com os mesmos pesos ridge por site. São análises exploratórias de integridade/custo, com orçamento pequeno; não validam convergência, ranking geral ou novidade científica. Resultados negativos foram preservados.','',
              'A retomada real com SIGTERM concluiu 108 consultas, sem duplicações e com hashes anteriores preservados. Testes adicionais com SIGKILL/SIGTERM verificaram o estado completo do SAITS upstream no loop explicitamente nomeado. Isso não valida a retomada de treino de todos os outros modelos.','',
              '## Código e métodos','',
              'Criados src/energygap/{data,protocols,masking,methods,forecasting,evaluation,statistics,reporting,runtime}, CLI, testes, configurações, locks, Makefile, CI, documentação, metadados de depósito e pacote fonte local. O catálogo inclui os onze candidatos. Integrações disponíveis: '+', '.join(data['integrated_catalogue_methods'])+'. Também há seis baselines, correção comum, adaptação CPI em energia e adaptador UCI opcional.','']
    lines += [f"- **{m['method']} indisponível:** {m['reason']}" for m in data['unavailable_catalogue_methods']]
    lines += ['','MOMENT e TS-ICL possuem ambientes isolados compatíveis e checkpoints oficiais verificados. O piloto pareado usa somente a saída MOMENT sob pins oficiais; o diagnóstico anterior com overrides permanece identificado. TS-ICL conserva termos não comerciais e quantis marginais.','',
              '## Comandos exatos','',
              '```sh',
              '.venv/bin/python -m pytest -q',
              '.venv/bin/energygap run --profile smoke --max-episodes 1 --progress log --budget-total-hours .1',
              f'.venv/bin/energygap status --experiment-id {smoke} --json',
              f'.venv/bin/energygap resume --experiment-id {smoke} --progress tqdm',
              '.venv/bin/energygap run --profile pilot --progress log --budget-total-hours .1',
              '.venv/bin/energygap plan --profile conference --dry-run --budget-total-hours .1',
              '.venv/bin/python scripts/exploratory_energy.py',
              f'.venv/bin/python scripts/paired_integrity_pilot.py --experiment-id {paired.get("experiment_id")} --resume require --progress none',
              '.venv/bin/python scripts/package_release.py','```','',
              'O estado fica em runs/<id>/state.sqlite. Consulte delivery.json e status para caminhos do último checkpoint e progresso. O smoke atual está concluído; sua retomada foi testada sem novo fit/inferência. Não há ETA honesta da matriz completa: cardinalidades/custos e requisitos restantes impedem iniciá-la. O plano é somente planejamento; não representa milhares de jobs executados.','',
              '## Pendências concretas','',
              'Confirmar potência instantânea/média e posição/semântica do timestamp; verificar metrologia/atualização dos contadores; resolver artefatos/licenças/código dos quatro modelos indisponíveis; completar executor conference, HPO, recuperação de todos os adaptadores, contabilidade de preparação e controles operacionais; ampliar suporte temporal antes da inferência confirmatória. Extensões C/O/A/tempo irregular, todos os horizontes e análise estatística completa permanecem separadas.','',
              'O pacote local exclui dados privados e pesos; nenhuma publicação, push, mensagem externa, depósito ou DOI foi realizado. Autoria e permissões dos dados exigem confirmação antes de um release público. Veja limitations.md, model_integration_status.md, resume_and_recovery.md e release_checklist.md.','']
    (ROOT/'docs/entrega.md').write_text('\n'.join(lines))
    print(json.dumps({'report':'docs/entrega.md','tests':data['tests'],'source_sha256':data['source_sha256'],'conference_complete':False},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--test-log',required=True);parser.add_argument('--clean-log',required=True);args=parser.parse_args();build(args.test_log,args.clean_log)
