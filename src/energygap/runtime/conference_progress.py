"""Durable conference counts and an explicitly partial, empirical serial ETA.

A fit and an inference query are different workloads.  They never share a rate
or a progress-bar denominator here.  Estimates use measured work in this run;
an unseen workload remains uncalibrated rather than borrowing another model's
speed.  Observed ranges describe timing variation, not confidence intervals or
guarantees about the remaining run.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from time import perf_counter

from .state import TERMINAL


LOG = logging.getLogger(__name__)
MAX_OBSERVATIONS = 20
_GROUPED_WORK = """
SELECT status,
       COALESCE(json_extract(metadata,'$.phase'),''),
       COALESCE(json_extract(metadata,'$.signature'),''),
       COALESCE(json_extract(metadata,'$.timing_signature'),''),
       COALESCE(json_extract(metadata,'$.method'),''),
       COALESCE(json_extract(metadata,'$.partition'),''),
       COALESCE(json_extract(metadata,'$.condition'),''),
       SUM(total_units), SUM(completed_units), COUNT(*),
       SUM(CASE WHEN total_units>0 THEN 1 ELSE 0 END),
       SUM(CASE WHEN total_units>0 AND completed_units>=1 THEN 1 ELSE 0 END)
FROM jobs
GROUP BY status,2,3,4,5,6,7
"""


def _family(phase, kind=None):
    """Normalize explicit caller names without pooling unlike model queries."""
    phase = str(phase or kind or 'uncalibrated')
    if kind == 'fit' and phase.startswith('hpo:'):
        return 'fit:' + phase.split(':', 1)[1]
    if kind == 'hpo_score' and phase.startswith('hpo:'):
        return 'hpo_score:' + phase.split(':', 1)[1]
    if kind == 'bootstrap' or phase.startswith(('conference-bootstrap:', 'bootstrap:')):
        return 'bootstrap'
    if phase == 'correction_trial' or kind == 'correction_trial':
        return 'correction_selection'
    return phase


def _hours(seconds):
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}min'
    return f'{seconds / 3600:.2f}h'


def _short_label(value, limit=48):
    label = str(value)
    if len(label) <= limit:
        return label
    before = (limit - 1) // 2
    return label[:before] + '…' + label[-(limit - before - 1):]


class ConferenceProgress:
    """One run-level display with bounded refresh and persistence cadences.

    ``job_id`` identifies the current stage only; measuring an observation does
    not fetch its job or scan the ledger.  Callers can provide an exact
    architecture cost key as ``phase`` and the same ``timing_signature`` in job
    metadata.  Otherwise a fit is estimated only from completed fits of its own
    model, with its empirical variation shown explicitly.
    """

    def __init__(self, store, mode, *, refresh_seconds=1., persist_seconds=10.):
        if any(not math.isfinite(value) or value <= 0 for value in (refresh_seconds, persist_seconds)):
            raise ValueError('progress intervals must be positive')
        self.store, self.mode = store, mode
        self.refresh_seconds, self.persist_seconds = float(refresh_seconds), float(persist_seconds)
        saved = store.get_meta('conference_cost_history', None)
        if saved is None:
            saved = store.get_meta('conference_timings', {})
        self.timings = {}
        for name, row in saved.items():
            self.timings[name] = dict(units=row.get('units', 0), seconds=row.get('seconds', 0.),
                                     observations=list(row.get('observations', []))[-MAX_OBSERVATIONS:])
        self.started = perf_counter()
        self.last_refresh = None
        self.last_persist = None
        self._ledger_revision = None
        self._grouped_rows = None
        self._timings_revision = self._estimated_timings_revision = 0
        self._persisted_timings_revision = -1
        self.last_log = -math.inf
        self.last_eta = None
        self.current_phase = None
        self.current_job = None
        self.current_training = store.get_meta('current_training') or {}
        self.current_report = store.get_meta('report_progress') or {}
        self._report_clock = None
        self._report_dirty = False
        self._context_job = None
        self._job_context = {}
        self._settings = (store.get_meta('config') or {}).get('experiment_settings', {})
        self._run_status = store.get_meta('status')
        self._training_dirty = False
        self.bar = self.query_bar = self.activity_bar = None
        if mode == 'tqdm':
            from tqdm.auto import tqdm
            formatting = '{l_bar}{bar}| {n_fmt}/{total_fmt} elapsed={elapsed} {postfix}'
            self.bar = tqdm(total=0, desc='TRABALHOS confirmados', unit='job', position=0,
                            bar_format=formatting, mininterval=self.refresh_seconds)
            self.query_bar = tqdm(total=0, desc='CONSULTAS confirmadas', unit='query', position=1,
                                  bar_format=formatting, mininterval=self.refresh_seconds)
            self.activity_bar = tqdm(total=0, desc='Preparando execução', unit='passo', position=2,
                                     bar_format=formatting, mininterval=self.refresh_seconds)

    def _context(self):
        """Read metadata once per job, never once per validation batch."""
        if self.current_job != self._context_job:
            self._context_job = self.current_job
            try:
                self._job_context = dict(self.store.job(self.current_job)['metadata']) if self.current_job else {}
            except KeyError:
                self._job_context = {}
            parts = str(self.current_job or '').split(':')
            if parts[0] == 'hpo' and len(parts) >= 4:
                for key, value in [('site', parts[1]), ('method', parts[2]), ('trial', parts[3])]:
                    self._job_context.setdefault(key, value)
                self._job_context.setdefault('fold', min(self._settings.get('folds', [1])))
                if self._settings.get('train_seeds'):
                    # The executor fits every HPO trial with the first declared
                    # training seed; historical job metadata omits this field.
                    self._job_context.setdefault('train_seed', self._settings['train_seeds'][0])
            elif len(parts) >= 3 and parts[1].startswith('fold'):
                self._job_context.setdefault('site', parts[0])
                self._job_context.setdefault('fold', parts[1][4:])
                if parts[2] != 'prepare':
                    self._job_context.setdefault('method', parts[2])
                for part in parts[3:]:
                    if part.startswith('seed'):
                        self._job_context.setdefault('train_seed', part[4:])
        result = dict(self._job_context)
        for key in ('site', 'method', 'fold', 'train_seed', 'trial', 'partition', 'condition'):
            if self.current_training.get(key) is not None:
                result[key] = self.current_training[key]
        return result

    def _activity_label(self):
        if self._run_status in ('completed', 'completed_with_failures'):
            return 'Execução concluída' + (' com falhas registradas' if self._run_status == 'completed_with_failures' else '')
        if self._run_status in ('interrupted', 'paused_budget'):
            return ('Execução interrompida; trabalho persistido preservado' if self._run_status == 'interrupted' else
                    'Execução pausada no teto cumulativo; trabalho persistido preservado')
        if self.current_report:
            info = self.current_report
            label = 'Relatório: ' + str(info.get('label') or info.get('stage') or 'inicializando')
            details = []
            for key, prefix in (('site', 'dataset '), ('hypothesis', ''), ('contrast', ''),
                                ('endpoint', ''), ('block_days', 'bloco=')):
                if info.get(key) is not None:
                    details.append(prefix + str(info[key]) + ('d' if key == 'block_days' else ''))
            return label + (' | ' + ' | '.join(details) if details else '')
        context = self._context()
        stage = self.current_training.get('stage') or self.current_phase or context.get('phase', '')
        if stage in ('training', 'epoch_start'):
            action = 'Treinando'
        elif stage == 'validation':
            action = 'Validando'
        elif stage == 'epoch_complete':
            action = 'Época concluída de'
        elif str(stage).startswith('hpo_score'):
            action = 'Selecionando HPO de'
        elif stage == 'inference' or str(stage).startswith('infer:'):
            action = 'Reconstruindo com'
        elif str(stage).startswith(('correction', 'energy')):
            action = 'Aplicando correção energética de'
        elif str(stage).startswith(('report', 'conference-bootstrap', 'bootstrap')):
            action = 'Gerando relatórios e estatísticas'
        elif str(stage).startswith('prepare'):
            action = 'Preparando dados e modelos'
        else:
            action = 'Executando' if self.current_job else 'Preparando execução'
        message = action
        if context.get('method'):
            message += ' ' + _short_label(context['method'], 36)
        if context.get('site'):
            message += ' no dataset ' + str(context['site'])
        details = []
        if context.get('fold') is not None:
            details.append('fold ' + str(context['fold']))
        seed = context.get('train_seed')
        if seed is not None and str(seed) != 'None':
            details.append('seed ' + str(seed))
        if context.get('trial') is not None:
            try:
                native_trial = int(context['trial'])
                indices = self._settings.get('hpo_trial_indices')
                trial = str(indices.index(native_trial) + 1 if indices and native_trial in indices else native_trial + 1)
            except (TypeError, ValueError):
                trial = str(context['trial'])
            total = self._settings.get('hpo_trials')
            details.append('HPO ' + trial + ('/' + str(total) if total else ''))
        if context.get('partition'):
            details.append({'validation': 'validação', 'test': 'teste'}.get(context['partition'], context['partition']))
        if context.get('condition'):
            details.append(str(context['condition']))
        if stage == 'validation' and self.current_training.get('validation_scope') == 'fixed_panel':
            full_count = self.current_training.get('validation_full_queries_total')
            details.append('painel fixo' + (f' de {full_count} consultas disponíveis' if full_count is not None else ''))
        return message + (' | ' + ' | '.join(details) if details else '')

    def _activity_counts(self):
        if self.current_report:
            return (self.current_report.get('done', 0), self.current_report.get('total'),
                    self.current_report.get('unit') or 'registro')
        current = self.current_training
        if current.get('validation_queries_total') is not None:
            return current.get('validation_queries_done', 0), current['validation_queries_total'], 'consulta de validação'
        if current.get('batches_total') is not None:
            return current.get('batches_done', 0), current['batches_total'], 'batch de treino'
        if current.get('shard_queries_total') is not None:
            return current.get('shard_queries_done', 0), current['shard_queries_total'], 'consulta do shard'
        return 0, 0, 'etapa'

    def _observe(self, phase, units, seconds, kind):
        if not phase or units <= 0 or seconds < 0 or not math.isfinite(seconds):
            return
        signature = _family(phase, kind)
        row = self.timings.setdefault(signature, dict(units=0, seconds=0., observations=[]))
        row['units'] += units
        row['seconds'] += seconds
        observations = row['observations']
        observations.append(dict(units=units, seconds=seconds))
        del observations[:-MAX_OBSERVATIONS]
        self._timings_revision += 1

    def training(self, info):
        """Report an epoch or validation boundary without counting it as a query.

        Validation performed inside training is displayed separately from the
        durable HPO-score and inference queries.  It is already included in the
        eventual measured fit duration and must not be charged a second time.
        """
        previous_stage = self.current_training.get('stage')
        previous_job = self.current_training.get('job_id')
        if info is None:
            self.current_training = {}
        else:
            if self.current_report:
                self.current_report = {}
                self._report_clock = None
                self._report_dirty = True
            incoming = dict(info)
            if incoming.get('job_id') != self.current_training.get('job_id'):
                self.current_training = {}
            self.current_training.update(incoming)
            if incoming.get('stage') in {'training', 'epoch_start'}:
                self.current_training.pop('validation_queries_done', None)
                self.current_training.pop('validation_queries_total', None)
                self.current_training.pop('validation_scope', None)
                self.current_training.pop('validation_full_queries_total', None)
                for key in ('inference_batch_index', 'inference_batches_total', 'inference_batch_size'):
                    self.current_training.pop(key, None)
            if incoming.get('stage') == 'validation':
                self.current_training.pop('batches_done', None)
                self.current_training.pop('batches_total', None)
            if incoming.get('stage') == 'inference':
                for key in ('epoch', 'max_epochs', 'patience', 'bad_epochs', 'batches_done', 'batches_total',
                            'validation_queries_done', 'validation_queries_total', 'epoch_timing_observations',
                            'fit_remaining_heuristic_range_seconds', 'fit_remaining_epoch_scenarios',
                            'fit_estimated_after_completed_epoch', 'fit_range_rule'):
                    self.current_training.pop(key, None)
            if incoming.get('stage') == 'epoch_complete':
                self._epoch_estimate(incoming)
            self.current_job = incoming.get('job_id', self.current_job)
        self._training_dirty = True
        boundary = (info is None or self.current_training.get('stage') != previous_stage or
                    self.current_training.get('job_id') != previous_job or
                    self.current_training.get('stage') == 'epoch_complete')
        return self.update(_training_only=True, _persist_boundary=boundary)

    def _epoch_estimate(self, incoming):
        """Time only this active fit; early-stopping scenarios are not ledger units."""
        current = self.current_training
        seconds = incoming.get('epoch_seconds')
        epoch = incoming.get('epoch')
        if seconds is None or epoch is None or seconds <= 0 or not math.isfinite(seconds):
            return
        observations = list(current.get('epoch_timing_observations', []))
        # Repeated presentation of the same durable epoch must not add samples.
        if not observations or epoch > observations[-1]['epoch']:
            observations.append(dict(epoch=epoch, seconds=float(seconds)))
            observations = observations[-MAX_OBSERVATIONS:]
        current['epoch_timing_observations'] = observations
        maximum, patience, bad = (current.get(key) for key in ('max_epochs', 'patience', 'bad_epochs'))
        if maximum is None or patience is None or bad is None:
            return
        maximum_remaining = max(0, maximum - epoch)
        earliest_stopping = min(maximum_remaining, max(0, patience - bad))
        rates = [item['seconds'] for item in observations]
        current['fit_remaining_heuristic_range_seconds'] = [earliest_stopping * min(rates),
                                                            maximum_remaining * max(rates)]
        current['fit_remaining_epoch_scenarios'] = [earliest_stopping, maximum_remaining]
        current['fit_estimated_after_completed_epoch'] = epoch
        current['fit_range_rule'] = ('remaining patience versus maximum epochs, multiplied by observed '
                                     'epoch min/max duration; not a confidence interval or a bound')

    def stage(self, phase, *, job_id=None, **info):
        self.current_phase = phase
        self._switch_job(job_id)
        if phase == 'report' and not info:
            return self.update(report_info=dict(stage='starting', label='inicializando',
                                                done=0, total=None, unit='etapa', status='running'))
        if not str(phase).startswith(('report', 'bootstrap', 'conference-bootstrap')) and self.current_report:
            self.current_report = {}
            self._report_clock = None
            self._report_dirty = True
        if info:
            return self.training(dict(info, job_id=job_id or self.current_job, stage=phase))
        return self.update()

    def _report_update(self, info, phase, job_id):
        """Observe same-stage wall deltas; never time offline periods or mix units.

        Callers publish cumulative counters after their safe export/commit
        boundary.  The clock includes computation, filesystem publication and
        ledger commit overhead between ticks, unlike compute-only service time.
        No result records are read by this presentation path.
        """
        incoming = dict(info)
        stage = str(incoming.get('stage') or 'report')
        family = str(incoming.get('cost_family') or stage)
        if stage == 'bootstrap' and not incoming.get('cost_family'):
            family += ':days' + str(incoming.get('block_days', 'unknown'))
        incoming['cost_family'] = family
        incoming['stage'] = stage
        incoming.setdefault('done', 0)
        if incoming['done'] is None:
            incoming['done'] = 0
        incoming.setdefault('total', None)
        incoming.setdefault('unit', 'registro')
        key = incoming.get('progress_key') or (
            stage, incoming.get('contrast_id'), incoming.get('block_days'),
            incoming.get('source'), incoming.get('site'), job_id,
        )
        incoming['progress_key'] = key if isinstance(key, str) else json.dumps(key)
        now = perf_counter()
        previous = self._report_clock
        if previous and previous[0] == key and previous[1] == family:
            delta = incoming['done'] - previous[2]
            elapsed = now - previous[3]
            if delta > 0 and elapsed > 0:
                self._observe('report_stage:' + family, delta, elapsed, None)
        if (not previous or previous[0] != key or previous[1] != family or
                incoming['done'] != previous[2]):
            self._report_clock = (key, family, incoming['done'], now)
        self.current_report = incoming
        self._report_dirty = True
        if self.current_training:
            self.current_training = {}
            self._training_dirty = True

    def _report_estimate(self):
        if not self.current_report:
            return None
        info = dict(self.current_report)
        family = info['cost_family'] if 'cost_family' in info else info.get('stage', 'report')
        row = self.timings.get('report_stage:' + family, {})
        observations = [item for item in row.get('observations', [])
                        if item.get('units', 0) > 0 and item.get('seconds', 0) > 0]
        calibrated = len(observations) >= 2 and row.get('units', 0) > 0
        total, done = info.get('total'), info.get('done', 0)
        remaining = max(0, total - done) if total is not None else None
        estimate = None
        interval = None
        if remaining == 0:
            estimate, interval = 0., [0., 0.]
        elif remaining is not None and calibrated:
            estimate = remaining * row['seconds'] / row['units']
            rates = [item['seconds'] / item['units'] for item in observations]
            interval = [remaining * min(rates), remaining * max(rates)]
        info.update(current_stage_remaining_seconds=estimate,
                    current_stage_heuristic_range_seconds=interval,
                    eta_calibration=('complete' if remaining == 0 else
                                     'empirical_wall_rate' if estimate is not None else
                                     'unknown_total' if total is None else 'calibrating'),
                    timing_observations=len(observations),
                    report_remaining_seconds=None,
                    report_eta_calibration='remaining_plan_unknown',
                    bootstrap_remaining_seconds=None,
                    timing_rule='same-family wall time between published counters; includes commit overhead; '
                                'observed ranges are not confidence intervals or guarantees',
                    updated_at=datetime.now(timezone.utc).isoformat())
        plan = info.get('remaining_plan')
        def estimate_plan(items):
            known = 0.
            unknown = []
            for item in items:
                name, units = item.get('family'), item.get('remaining_units')
                timing = self.timings.get('report_stage:' + str(name), {})
                samples = [sample for sample in timing.get('observations', [])
                           if sample.get('units', 0) > 0 and sample.get('seconds', 0) > 0]
                if units == 0:
                    continue
                if units is None or units < 0 or len(samples) < 2 or not timing.get('units'):
                    unknown.append(name)
                    continue
                known += units * timing['seconds'] / timing['units']
            return known, unknown
        bootstrap_plan = info.get('bootstrap_remaining_plan')
        if info.get('bootstrap_plan_complete') and isinstance(bootstrap_plan, list):
            known, unknown = estimate_plan(bootstrap_plan)
            info.update(known_bootstrap_remaining_seconds=known,
                        uncalibrated_bootstrap_families=unknown,
                        bootstrap_remaining_seconds=known if not unknown else None)
        if info.get('report_complete'):
            info.update(report_remaining_seconds=0., report_eta_calibration='complete')
        elif info.get('plan_complete') and isinstance(plan, list):
            known, unknown = estimate_plan(plan)
            info.update(known_report_remaining_seconds=known,
                        uncalibrated_report_families=unknown,
                        report_remaining_seconds=known if not unknown else None,
                        report_eta_calibration='empirical_wall_rates' if not unknown else 'partial')
        return info

    def _switch_job(self, job_id):
        if job_id is None:
            return
        if self.current_training and self.current_training.get('job_id') != job_id:
            self.current_training = {}
            self._training_dirty = True
        self.current_job = job_id

    @staticmethod
    def _workloads(row):
        status, phase, signature, timing_signature, model, partition, condition, total, done, jobs, fits, fitted = row
        if status in TERMINAL:
            return []
        if phase == 'hpo':
            result = []
            if fits > fitted:
                result.append((timing_signature or 'fit:' + model, fits - fitted, 'fit', fits - fitted))
            remaining = max(0, total - fits - max(0, done - fitted))
            if remaining:
                result.append(('hpo_score:' + model, remaining, 'query', jobs))
            return result
        remaining = max(0, total - done)
        if not remaining:
            return []
        if phase == 'fit':
            family = timing_signature or 'fit:' + model
            kind = 'fit'
        elif phase in {'inference', 'hpo_score'}:
            family = timing_signature or signature or (
                f'infer:{model}:{partition}:{condition}' if phase == 'inference' else 'hpo_score:' + model)
            kind = 'query'
        else:
            family = timing_signature or signature or phase or 'uncalibrated'
            kind = 'other'
        return [(_family(family, phase), remaining, kind, jobs)]

    def _estimate(self):
        rows = self._grouped_rows
        total_jobs = resolved_jobs = query_total = query_done = 0
        statuses = {}
        known = low = high = 0.
        calibrated = uncalibrated = 0
        unknown = {}
        unknown_fits = unknown_queries = unknown_other = 0
        for row in rows:
            status, phase, _, _, _, _, _, total, done, jobs, fits, fitted = row
            total_jobs += jobs
            statuses[status] = statuses.get(status, 0) + jobs
            resolved_jobs += jobs if status in TERMINAL else 0
            if phase in {'inference', 'hpo_score'}:
                query_total += total
                query_done += done
            elif phase == 'hpo':
                query_total += max(0, total - fits)
                query_done += max(0, done - fitted)
            for family, remaining, kind, count in self._workloads(row):
                timing = self.timings.get(family)
                if not timing or not timing['units']:
                    previous = unknown.setdefault(family, dict(units=0, workloads=0, kind=kind))
                    previous['units'] += remaining
                    previous['workloads'] += count
                    uncalibrated += count
                    if kind == 'fit':
                        unknown_fits += remaining
                    elif kind == 'query':
                        unknown_queries += remaining
                    else:
                        unknown_other += count
                    continue
                average = timing['seconds'] / timing['units']
                observed = [item['seconds'] / item['units'] for item in timing['observations'] if item['units']]
                known += remaining * average
                low += remaining * min(observed, default=average)
                high += remaining * max(observed, default=average)
                calibrated += count
        now = datetime.now(timezone.utc)
        complete = not unknown
        return dict(
            remaining_seconds=known if complete else None,
            known_remaining_seconds=known,
            estimated_finish=(now + timedelta(seconds=known)).isoformat() if complete else None,
            heuristic_range_seconds=[low, high] if complete else None,
            known_heuristic_range_seconds=[low, high],
            range_rule='observed min/max seconds per workload unit; not a confidence interval or a bound',
            calibration='empirical_serial_phase_means' if complete else 'partial',
            uncalibrated_signatures=sorted(unknown),
            uncalibrated_workloads=unknown,
            uncalibrated_fit_count=unknown_fits,
            uncalibrated_query_count=unknown_queries,
            uncalibrated_other_workloads=unknown_other,
            calibrated_workloads=calibrated,
            uncalibrated_workload_count=uncalibrated,
            resolved_jobs=resolved_jobs, total_jobs=total_jobs,
            job_status_counts=statuses,
            confirmed_queries=query_done, total_queries=query_total,
            current_phase=self.current_phase, current_job=self.current_job,
            current_training=dict(self.current_training) if self.current_training else None,
            current_fit_heuristic_range_seconds=self.current_training.get('fit_remaining_heuristic_range_seconds'),
            elapsed_session_seconds=max(0., perf_counter() - self.started),
            updated_at=now.isoformat(),
            execution_rule='serial measured work; no division by CPU thread count',
        )

    def _activity_postfix(self):
        # Put exact counters before long ETA/job text.  Narrow terminals must
        # not turn an apparently truncated 1625-query denominator into 162.
        if self.current_report:
            info = self._report_estimate()
            total = info.get('total')
            counters = [f'{info.get("done",0)}/{total if total is not None else "?"} {info.get("unit","registros")}']
            if info.get('jobs_total') is not None:
                counters.append(f'tarefas bootstrap={info.get("jobs_done",0)}/{info["jobs_total"]}')
            if info.get('attempts_total') is not None:
                counters.append(f'réplicas confirmadas={info.get("attempts_confirmed",0)}/{info["attempts_total"]}')
            seconds = info.get('current_stage_remaining_seconds')
            counters.append('ETA etapa~' + _hours(seconds) if seconds is not None else
                            'ETA etapa calibrando' if total is not None else 'total da etapa desconhecido')
            if info.get('bootstrap_remaining_seconds') is not None:
                counters.append('ETA bootstrap~' + _hours(info['bootstrap_remaining_seconds']))
            report_seconds = info.get('report_remaining_seconds')
            counters.append('ETA relatório~' + _hours(report_seconds) if report_seconds is not None else
                            'ETA relatório total não estimado')
            return ' '.join(counters)
        current = self.current_training
        counters = []
        if current.get('validation_queries_total') is not None:
            counters.append(f'val={current.get("validation_queries_done",0)}/{current["validation_queries_total"]}')
        if current.get('epoch') is not None:
            epoch = f'epoch={current["epoch"]}'
            if current.get('max_epochs'):
                epoch += f'/{current["max_epochs"]}'
            counters.append(epoch)
        if current.get('batches_total') is not None:
            counters.append(f'batch={current.get("batches_done",0)}/{current["batches_total"]}')
        if current.get('inference_batch_index') is not None:
            batch = f'lote={current["inference_batch_index"]}'
            if current.get('inference_batches_total') is not None:
                batch += f'/{current["inference_batches_total"]}'
            if current.get('inference_batch_size') is not None:
                batch += f' n={current["inference_batch_size"]}'
            counters.append(batch)
        if current.get('shard_queries_total') is not None:
            counters.append(f'shard={current.get("shard_queries_done",0)}/{current["shard_queries_total"]}')
        if current.get('fit_remaining_heuristic_range_seconds') is not None:
            lo, hi = current['fit_remaining_heuristic_range_seconds']
            counters.append(f'treino~{_hours(lo)}–{_hours(hi)} (fit atual; cenários de early stopping)')
        return ' '.join(counters)

    def _postfix(self, eta):
        report = eta.get('current_report')
        if report and self._run_status not in ('completed', 'completed_with_failures'):
            remaining = report.get('report_remaining_seconds')
            stage_remaining = report.get('current_stage_remaining_seconds')
            if remaining is not None:
                timing = 'ETA relatório~' + _hours(remaining)
            elif stage_remaining is not None:
                timing = 'ETA etapa~' + _hours(stage_remaining) + '; relatório total não estimado'
            else:
                timing = 'ETA relatório calibrando; total não estimado'
        elif eta['uncalibrated_signatures']:
            if eta['calibrated_workloads']:
                timing = 'ETA parcial=' + _hours(eta['known_remaining_seconds']) + '; total não estimado'
            else:
                timing = 'ETA calibrando; total não estimado'
            timing += f'; sem medida={len(eta["uncalibrated_signatures"])} fases'
            if eta['uncalibrated_fit_count']:
                timing += f'/{eta["uncalibrated_fit_count"]} treinos'
        else:
            timing = 'ETA global~' + _hours(eta['remaining_seconds'])
            lo, hi = eta['heuristic_range_seconds']
            if hi > lo:
                timing += f' (observado {_hours(lo)}–{_hours(hi)})'
        unavailable = sum(eta['job_status_counts'].get(status, 0) for status in ('unavailable', 'unsupported', 'blocked'))
        if unavailable:
            timing += f' indisponíveis={unavailable}'
        failed = eta['job_status_counts'].get('failed_terminal', 0)
        if failed:
            timing += f' falhas={failed}'
        return timing

    def _live_eta(self, eta):
        """Overlay cheap, current stage counters on the cached workload estimate."""
        result = dict(eta)
        result.update(current_phase=self.current_phase, current_job=self.current_job,
                      current_training=dict(self.current_training) if self.current_training else None,
                      current_fit_heuristic_range_seconds=self.current_training.get('fit_remaining_heuristic_range_seconds'),
                      elapsed_session_seconds=max(0., perf_counter() - self.started),
                      updated_at=datetime.now(timezone.utc).isoformat())
        result['activity'] = self._activity_label()
        result['current_report'] = self._report_estimate()
        return result

    def update(self, phase=None, units=0, seconds=0, *, force=False, job_id=None, kind=None,
               report_info=None, _training_only=False, _persist_boundary=False):
        if report_info is not None:
            previous = (self.current_report.get('stage'), self.current_report.get('progress_key'),
                        self.current_report.get('status'))
            self._report_update(report_info, phase, job_id)
            current = (self.current_report.get('stage'), self.current_report.get('progress_key'),
                       self.current_report.get('status'))
            _persist_boundary = _persist_boundary or previous != current
        else:
            self._observe(phase, units, seconds, kind)
        if phase:
            self.current_phase = phase
        self._switch_job(job_id)
        now = perf_counter()
        revision = self.store.progress_revision
        work_changed = self._ledger_revision != revision
        persist_due = self.last_persist is None or now - self.last_persist >= self.persist_seconds
        if (not force and not work_changed and not _persist_boundary and not persist_due and self.last_refresh is not None
                and now - self.last_refresh < self.refresh_seconds):
            self.last_eta = self._live_eta(self.last_eta)
            return self.last_eta
        self.last_refresh = now
        if not self._training_dirty and not _training_only and not self.current_report:
            # Support callers that still publish epoch metadata directly.
            self.current_training = self.store.get_meta('current_training') or {}
        if force or work_changed or self._grouped_rows is None:
            self._grouped_rows = list(self.store.db.execute(_GROUPED_WORK))
            self._ledger_revision = revision
            self._run_status = self.store.get_meta('status')
            if self._run_status in ('completed', 'completed_with_failures', 'interrupted', 'paused_budget'):
                self.current_training = {}
                self._training_dirty = True
                if self.current_report:
                    self.current_report['execution_status'] = self._run_status
                    if self._run_status in ('completed', 'completed_with_failures'):
                        self.current_report.update(report_complete=True, status='complete')
                    self._report_dirty = True
        costs_changed = self._timings_revision != self._estimated_timings_revision
        if force or work_changed or costs_changed or self.last_eta is None:
            eta = self._estimate()
            self._estimated_timings_revision = self._timings_revision
        else:
            eta = self._live_eta(self.last_eta)
        eta['activity'] = self._activity_label()
        eta['current_report'] = self._report_estimate()
        persist = force or work_changed or _persist_boundary or persist_due
        if persist:
            # Scientific commits and boundaries persist immediately. Batch-only
            # presentation updates stay in memory between compact metadata saves.
            transaction = nullcontext() if self.store.db.in_transaction else self.store.transaction()
            with transaction:
                if self._training_dirty:
                    self.store.set_meta('current_training', self.current_training or None)
                if self._report_dirty:
                    self.store.set_meta('report_progress', eta['current_report'])
                if force or self._timings_revision != self._persisted_timings_revision:
                    self.store.set_meta('conference_cost_history', self.timings)
                    self.store.set_meta('conference_timings', self.timings)
                self.store.set_meta('conference_eta', eta)
                self.store.set_meta('eta', eta)
            self._persisted_timings_revision = self._timings_revision
            self._training_dirty = False
            self._report_dirty = False
            self.last_persist = now
        self.last_eta = eta
        if self.bar is not None:
            self.bar.total = eta['total_jobs']
            self.bar.n = eta['resolved_jobs']
            self.bar.set_postfix_str(self._postfix(eta), refresh=False)
            self.bar.refresh()
            self.query_bar.total = eta['total_queries']
            self.query_bar.n = eta['confirmed_queries']
            current = self.current_training
            pending = ('consultas concluídas; réplicas bootstrap e exportações na linha abaixo' if self.current_report else
                       'treino/validação em andamento; contabilizados após commit'
                       if current.get('stage') in ('training', 'epoch_start', 'validation', 'epoch_complete') else
                       'HPO + inferências persistidas; validação por época na linha abaixo')
            self.query_bar.set_postfix_str(pending, refresh=False)
            self.query_bar.refresh()
            self.activity_bar.n, self.activity_bar.total, self.activity_bar.unit = self._activity_counts()
            self.activity_bar.set_description_str(self._activity_label(), refresh=False)
            self.activity_bar.set_postfix_str(self._activity_postfix(), refresh=False)
            self.activity_bar.refresh()
        elif self.mode == 'log' and (force or now - self.last_log >= 2.):
            LOG.info(json.dumps(dict(event='conference_progress', **eta)))
            self.last_log = now
        return eta

    def close(self):
        self.update(force=True)
        if self.query_bar is not None:
            self.query_bar.close()
        if self.activity_bar is not None:
            self.activity_bar.close()
        if self.bar is not None:
            self.bar.close()
