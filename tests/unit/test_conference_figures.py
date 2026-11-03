import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energygap.reporting.conference_figures import contrast_figure, generate_conference_figures


def test_unavailable_contrast_has_text_and_no_zero_estimate_marker():
    fig=contrast_figure([dict(site='H1',hypothesis='instantaneous',contrast='contrast_energy_gain',
        endpoint='nmae',estimate=None,ci95=None,reason='missing temporal support')], 'instantaneous')
    ax=fig.axes[0]
    assert not ax.collections
    assert any('NA:' in text.get_text() for text in ax.texts)
    assert any('Exploratory conditional physics' in text.get_text() for text in fig.texts)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_figures_export_measured_compact_metrics_and_streamed_scalar_reductions(tmp_path):
    reports=tmp_path/'run/reports';reports.mkdir(parents=True)
    main=[];core=[];episodes=[];forecasts=[];energy=[]
    for method in ('linear','CSDI','TS-ICL'):
        core.append(dict(site='H1',method=method,variant='P',nmae=.2,forecast_nmae=.3))
        for condition,duration in (('C3',15.),('C4',60.),('C5',180.)):
            main.append(dict(site='H1',fold=1,condition=condition,duration=duration,method=method,variant='P',
                nmae=.2,forecast_nmae=.3,latency_p50_seconds=.01))
            for seed in ([11,23] if method=='CSDI' else [None]):
                intervals={str(c):dict(coverage=c-.05) for c in (.5,.8,.9,.95)}
                if method=='TS-ICL':
                    intervals['0.95']={'coverage':None,'reason':'missing native quantiles'}
                probability={'intervals':intervals,'crps':.1} if method=='CSDI' else {'intervals':intervals,'crps_truncated_quantile_approximation':.2} if method=='TS-ICL' else None
                episodes.append(dict(site='H1',fold=1,partition='test',condition=condition,duration=duration,
                    method=method,variant='P',gap_id=condition,mask_seed=2001,train_seed=seed,
                    probabilistic=json.dumps(probability) if probability else None))
                for horizon in (1800,3600):
                    forecasts.append(dict(site='H1',fold=1,partition='test',condition=condition,duration=duration,
                        method=method,variant='P',gap_id=condition,mask_seed=2001,train_seed=seed,
                        horizon_seconds=horizon,nmae=1. if seed!=23 else 3.))
            episodes.append(dict(site='H1',fold=1,partition='test',condition=condition,duration=duration,
                method='linear+energy_consistent',variant='E:interval_mean_start',gap_id=condition,mask_seed=2001,
                net_power_integral_error_kwh=-.2,net_input_counter_residual_kwh=0.))
            energy.append(dict(site='H1',fold=1,partition='test',condition=condition,duration=duration,
                method='cpi_energy_donor_adapted',physical_hypothesis='interval_mean_start',gap_id=condition,mask_seed=2001,
                net_power_integral_error_kwh=.1,net_input_counter_residual_kwh=0.))
    # Duplicate C6 geometry must not enter primary C3-C5 curves.
    main.append(dict(site='H1',fold=1,condition='C6',duration=15.,method='linear',variant='P',
        nmae=1000.,forecast_nmae=1000.,latency_p50_seconds=1000.))
    for name,rows in (('main_table',main),('core_aggregate',core),('episodes',episodes),('forecasts_long',forecasts),('energy_track',energy)):
        pd.DataFrame(rows).to_csv(reports/(name+'.csv'),index=False)
    (reports/'contrasts.json').write_text(json.dumps([dict(site='H1',hypothesis='interval_mean_start',
        contrast='contrast_energy_gain',endpoint='nmae',estimate=None,ci95=None,reason='missing temporal support')]))
    (reports/'summary.json').write_text(json.dumps({'blocked_methods':{'FIM':{'reason':'unresolved weights license'}}}))
    paths=generate_conference_figures(reports.parent)
    assert len(paths)>=18
    assert {Path(path).suffix for path in paths}=={'.png','.svg','.pdf'}
    assert all(Path(path).is_file() and Path(path).stat().st_size>100 for path in paths)
    assert all(Path(path).parent==reports/'figures' for path in paths)
    # Strict JSON parse checks that no absent measurement became NaN/Infinity.
    def invalid_constant(value):
        raise AssertionError(value)
    manifest=json.loads((reports/'figures/manifest.json').read_text(),parse_constant=invalid_constant)
    assert manifest['physical_confirmatory'] is False
    assert 'FIM' in manifest['blocked_methods']
    reconstruction=next(row for row in manifest['figures'] if row['name']=='reconstruction_duration_baselines')
    assert {point['nmae'] for point in reconstruction['points']}=={.2}
    horizon=next(row for row in manifest['figures'] if row['name']=='forecast_horizons_learned')
    assert {point['value'] for point in horizon['points']}=={2.}
    uncertainty=next(row for row in manifest['figures'] if row['name']=='uncertainty_coverage_P')
    unavailable=[point for point in uncertainty['points'] if point['method']=='TS-ICL' and point['metric']=='coverage_0.95']
    assert unavailable and unavailable[0]['value'] is None
    assert not list(reports.glob('.figure-reduction-*'))
