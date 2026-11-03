#!/usr/bin/env python3
"""Check selected native models using synthetic inputs in one process.

This is an installation check, not an experiment or evidence for the paper.
CUDA is the default; CPU is available for local validation without GPU claims.
All pretrained weights must already exist locally and pass their adapter checks.
Without --methods, all seven supported integrations are checked.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('BRITS', 'SAITS', 'CSDI', 'ImputeFormer', 'T1', 'MOMENT-1-base', 'TS-ICL')
DETERMINISTIC_BATCH_METHODS = ('BRITS', 'SAITS', 'ImputeFormer', 'T1', 'MOMENT-1-base')


def architecture(name):
    if name == 'BRITS':
        return dict(rnn_hidden_size=8)
    if name == 'SAITS':
        return dict(n_layers=1, d_model=8, n_heads=2, d_k=4, d_v=4, d_ffn=8)
    if name == 'CSDI':
        return dict(n_layers=1, n_heads=2, n_channels=8, d_time_embedding=8,
                    d_feature_embedding=4, d_diffusion_embedding=8, n_diffusion_steps=2)
    if name == 'ImputeFormer':
        return dict(n_layers=1, d_input_embed=8, d_learnable_embed=8, d_proj=4,
                    d_ffn=8, n_temporal_heads=2)
    if name == 'T1':
        return dict(n_heads=8, n_blocks=[1, 1], kernel_size_large=[7, 3])
    return {}


def publish(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, path)


def parameter_devices(model, name, requested):
    import torch
    if name in ('BRITS', 'SAITS', 'CSDI', 'ImputeFormer'):
        modules = {'model': model.adapter.model.model}
    elif name == 'T1':
        modules = {'model': model.adapter.model}
    elif name == 'MOMENT-1-base':
        modules = {'model': model.adapter.native.model}
    else:
        modules = {'imputer': model.adapter.native.imputer, 'forecaster': model.adapter.native.forecaster}
    result = {}
    for component, module in modules.items():
        parameters = list(module.parameters())
        if not parameters or any(parameter.device != torch.device(requested) for parameter in parameters):
            raise RuntimeError(f'{name}/{component}: parameters are not on {requested}')
        result[component] = dict(parameters=sum(parameter.numel() for parameter in parameters),
                                 devices=sorted({str(parameter.device) for parameter in parameters}))
    return result


def check_brits_training_step(model, query):
    """Compare native/accelerated FP32 loss, gradients and one Adam update."""
    import numpy as np
    import torch
    from energygap.methods.brits_fast import forward
    from energygap.methods.prepared_inference import BRITS_INPUT_CACHE
    safe = np.where(query.available_to_method & ~query.padding_mask, query.values, np.nan)
    normalized = ((safe - model.adapter.mean) / model.adapter.scale).reshape(1, model.context, 1)
    tensors, _ = BRITS_INPUT_CACHE.prepare(normalized)
    native = model.adapter.model
    inputs = native._assemble_input_for_testing(tensors)
    left, right = deepcopy(native.model), deepcopy(native.model)
    left.train(); right.train()
    opt_left = torch.optim.Adam(left.parameters(), lr=.001)
    opt_right = torch.optim.Adam(right.parameters(), lr=.001)
    opt_left.zero_grad(set_to_none=True); opt_right.zero_grad(set_to_none=True)
    output_left = left(inputs, calc_criterion=True)
    output_right = forward(right, inputs, calc_criterion=True)
    atol = rtol = 1e-5
    if not torch.allclose(output_left['loss'], output_right['loss'], atol=atol, rtol=rtol):
        raise RuntimeError('BRITS: native/vectorized FP32 training losses differ')
    output_left['loss'].backward(); output_right['loss'].backward()
    gradient_difference = 0.
    for (name, parameter), (other_name, other) in zip(left.named_parameters(), right.named_parameters()):
        if name != other_name or (parameter.grad is None) != (other.grad is None):
            raise RuntimeError('BRITS: native/vectorized gradient support differs')
        if parameter.grad is not None:
            if (not torch.isfinite(parameter.grad).all() or not torch.isfinite(other.grad).all()
                    or not torch.allclose(parameter.grad, other.grad, atol=atol, rtol=rtol)):
                raise RuntimeError('BRITS: native/vectorized FP32 gradients differ: ' + name)
            gradient_difference = max(gradient_difference,
                float(torch.max(torch.abs(parameter.grad - other.grad)).detach().cpu()))
    opt_left.step(); opt_right.step()
    parameter_difference = 0.
    for parameter, other in zip(left.parameters(), right.parameters()):
        if not torch.allclose(parameter, other, atol=atol, rtol=rtol):
            raise RuntimeError('BRITS: native/vectorized one-step Adam parameters differ')
        parameter_difference = max(parameter_difference,
            float(torch.max(torch.abs(parameter - other)).detach().cpu()))
    return dict(passed=True, device=model.device, cuda_tested=str(model.device).startswith('cuda:'),
        precision='float32', atol=atol, rtol=rtol,
        gradient_max_abs_error=gradient_difference, updated_parameter_max_abs_error=parameter_difference,
        loss_native=float(output_left['loss'].detach().cpu()),
        loss_vectorized=float(output_right['loss'].detach().cpu()), optimizer='Adam_one_step')


def check_model(name, root, device, threads):
    import numpy as np
    import torch
    from energygap.methods.conference_models import LOCAL_METHODS, create_adapter
    context = 16 if name in LOCAL_METHODS else 512
    sample_options = {'csdi_sample_batch_size': 4} if name == 'CSDI' else {}
    model = create_adapter(name, root=root, device=device, threads=threads, seed=11,
        params={'architecture': architecture(name)}, epochs=1, patience=1, batch_size=2,
        context=context, inference_batch_size=256, **sample_options)
    if name in LOCAL_METHODS:
        # The validation support follows the training support chronologically.
        times = np.arange(9 * context, dtype=np.float32).reshape(9, context, 1)
        windows = np.sin(times / 5) + times / 100
        train, validation = windows[:6].copy(), windows[6:].copy()
        train[0, 2], validation[0, 4] = np.nan, np.nan
        model.fit_windows(train, validation)
        if model.completed_epochs != 1:
            raise RuntimeError(f'{name}: one native training epoch was not completed')
    values = np.sin(np.arange(context) / 5) + np.arange(context) / 100
    padding = np.zeros(context, bool)
    observed = np.ones(context, bool)
    observed[context // 2:context // 2 + max(3, context // 16)] = False
    if context == 512:
        padding[:16] = True
        observed[padding] = False
    values[~observed] = np.nan
    times = np.arange(context, dtype=float) * 300
    query = SimpleNamespace(values=values, available_to_method=observed, padding_mask=padding,
                            times=times, issuance_time=float(times[-1]), task='R')
    requested_samples = 5 if name == 'CSDI' else 2
    with torch.inference_mode():
        result = model.infer_query(query, samples=requested_samples, sample_seed=23)
    point = np.asarray(result['point_estimate'])
    if point.shape != (context,) or not np.isfinite(point[~padding]).all():
        raise RuntimeError(f'{name}: output must be finite on every nonpadding position')
    if not np.array_equal(point[observed], values[observed]):
        raise RuntimeError(f'{name}: admitted observations were changed')
    batch_check = None
    if name in DETERMINISTIC_BATCH_METHODS:
        with torch.inference_mode():
            outputs = model.infer_queries([query] * 4, samples=2,
                                          sample_seeds=[23] * 4, batch_size=4)
        if len(outputs) != 4:
            raise RuntimeError(f'{name}: native batch returned a different query count')
        maximum_difference = 0.
        hidden = ~observed & ~padding
        for output in outputs:
            batched = np.asarray(output['point_estimate'])
            if batched.shape != point.shape or not np.isfinite(batched[~padding]).all():
                raise RuntimeError(f'{name}: native batch returned invalid shape/nonfinite output')
            if not np.array_equal(batched[observed], values[observed]):
                raise RuntimeError(f'{name}: native batch changed admitted observations')
            if not np.allclose(batched[hidden], point[hidden], rtol=1e-5, atol=1e-5):
                raise RuntimeError(f'{name}: native batch disagrees with scalar reconstruction')
            timing = output.get('timing', {})
            if timing.get('batch_size') != 4 or timing.get('timing_mode') != 'batch_amortized':
                raise RuntimeError(f'{name}: adapter did not confirm a real native batch of four')
            maximum_difference = max(maximum_difference, float(np.max(np.abs(batched[~padding] - point[~padding]))))
        batch_check = dict(passed=True, actual_batch_size=4, scalar_parity_max_abs_error=maximum_difference,
                           rtol=1e-5, atol=1e-5, observed_preserved=True,
                           timing_mode='batch_amortized')
        if name == 'BRITS':
            batch_check['native_vectorized_training_step'] = check_brits_training_step(model, query)
            original = model.brits_execution
            try:
                model.set_brits_execution('native')
                with torch.inference_mode():
                    native_outputs = model.infer_queries([query] * 4, batch_size=4)
                differences = [float(np.max(np.abs(np.asarray(native['point_estimate']) -
                                                    np.asarray(fast['point_estimate']))))
                               for native, fast in zip(native_outputs, outputs)]
                if len(native_outputs) != 4 or any(not np.allclose(native['point_estimate'],
                        fast['point_estimate'], atol=1e-5, rtol=1e-5)
                        for native, fast in zip(native_outputs, outputs)):
                    raise RuntimeError('BRITS: native versus vectorized FP32 batch parity failed')
                batch_check['native_vectorized_fp32_parity'] = dict(passed=True,
                    max_abs_error=max(differences), execution_modes=['native', original],
                    same_weights=True, atol=1e-5, rtol=1e-5)
            finally:
                model.set_brits_execution(original)
    sample_batch_check = None
    if name == 'CSDI':
        samples = np.asarray(result.get('samples'))
        if samples.shape != (5, context) or not np.isfinite(samples).all():
            raise RuntimeError('CSDI: five finite joint trajectories with the original context are required')
        if not np.array_equal(samples[:, observed], np.repeat(values[None, observed], 5, axis=0)):
            raise RuntimeError('CSDI: sampled trajectories changed admitted observations')
        timing, metadata = result.get('timing', {}), result.get('model_metadata', {})
        if (timing.get('sample_batch_sizes') != [4, 1]
                or timing.get('timing_mode') != 'query_wall_mc_batched'
                or metadata.get('csdi_sample_batch_size') != 4
                or metadata.get('sampling_policy') != 'native_replicated_condition_rows_v1'):
            raise RuntimeError('CSDI: adapter did not confirm native sample batches of four plus one')
        with torch.inference_mode():
            repeated = model.infer_query(query, samples=5, sample_seed=23)
        if not np.array_equal(samples, repeated.get('samples')):
            raise RuntimeError('CSDI: repeated joint sampling differs for the same configuration and seed')
        sample_batch_check = dict(passed=True, samples=5, sample_shape=list(samples.shape),
            sample_batch_sizes=[4, 1], observed_preserved=True, finite=True,
            reproducible_same_configuration_and_seed=True,
            sampling_policy=metadata['sampling_policy'], timing_mode=timing['timing_mode'])
    quantiles = result.get('quantiles')
    if quantiles is not None:
        if not np.isfinite(quantiles[~padding]).all():
            raise RuntimeError(f'{name}: nonfinite marginal quantiles')
        if not np.array_equal(quantiles[observed], np.repeat(values[observed, None], quantiles.shape[-1], axis=1)):
            raise RuntimeError(f'{name}: admitted observations changed in marginal quantiles')
    devices = parameter_devices(model, name, device)
    return dict(passed=True, context=context, completed_epochs=model.completed_epochs,
                observed_preserved=True, finite_nonpadding=True, parameter_devices=devices,
                configured_inference_batch_size=256,
                deterministic_batch_check=batch_check,
                joint_sample_batch_check=sample_batch_check,
                quantile_shape=list(quantiles.shape) if quantiles is not None else None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', choices=('cuda:0', 'cpu'))
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS),
                        help='Native integrations selected for this preset; omitted flag checks all seven')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/analyses/server_environment/native_models.json')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if len(set(args.methods)) != len(args.methods):
        parser.error('--methods must not contain duplicates')
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f'Verificando {len(args.methods)} modelos nativos selecionados com entradas sintéticas; importando bibliotecas...', flush=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from energygap.runtime.threads import configure_cpu_threads, configure_torch_threads
    configure_cpu_threads(args.threads)
    import torch
    configure_torch_threads(torch, args.threads)
    from energygap.runtime.device import cuda_probe, resolve_device, torch_device_metadata
    device = resolve_device(args.device, allow_missing_torch=False)
    diagnostic = cuda_probe(device) if device.startswith('cuda:') else torch_device_metadata(device)
    report = dict(status='checking', scope='synthetic_installation_check_not_scientific_results',
                  timestamp=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                  device=device, cuda_tested=device.startswith('cuda:'), threads=args.threads,
                  selected_methods=args.methods,
                  runtime=diagnostic,
                  hardware_compatibility=dict(v100_sm70_checked=(device.startswith('cuda:') and
                      diagnostic.get('compute_capability') == [7, 0]),
                      actual_compute_capability=diagnostic.get('compute_capability'),
                      real_cuda_kernel_probe=diagnostic.get('cuda_kernel_test'),
                      requires_bf16=False, requires_tf32=False, requires_triton=False,
                      precision='float32_native_no_autocast'), methods={})
    publish(args.output, report)
    from energygap.methods.conference_models import model_availability
    available = model_availability(ROOT, device=device, threads=args.threads)
    missing = {name: available[name]['reason'] for name in args.methods if not available[name]['available']}
    if missing:
        report.update(status='failed', missing_prerequisites=missing)
        publish(args.output, report)
        raise RuntimeError(f'Modelos obrigatórios indisponíveis: {missing}')
    for index, name in enumerate(args.methods, 1):
        print(f'[{index}/{len(args.methods)}] {name}: início ({device}).', flush=True)
        started = perf_counter()
        try:
            result = check_model(name, ROOT, device, args.threads)
        except Exception as exc:
            report['methods'][name] = dict(passed=False, error=str(exc), seconds=perf_counter() - started)
            report['status'] = 'failed'
            publish(args.output, report)
            raise
        result['seconds'] = perf_counter() - started
        report['methods'][name] = result
        publish(args.output, report)
        print(f'[{index}/{len(args.methods)}] {name}: aprovado em {result["seconds"]:.1f}s.', flush=True)
        gc.collect()
        if device.startswith('cuda:'):
            torch.cuda.empty_cache()
    report['status'] = 'passed'
    publish(args.output, report)
    print(f'{len(args.methods)} modelos selecionados aprovados. Auditoria: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
