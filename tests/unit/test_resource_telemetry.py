"""Linux proc parsing handles names/PID reuse and reports cores, not RAM occupancy."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('resource_telemetry',Path(__file__).resolve().parents[2]/'scripts/resource_telemetry.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_stat_tree_and_cpu_core_rates_do_not_count_reused_pid():
    # Fields 3..24 after a parenthesized name with spaces/parentheses.
    fields=['S','1']+['0']*20
    fields[11]='120';fields[12]='80';fields[17]='4';fields[19]='99';fields[21]='10'
    row=module.parse_stat('42 (python (worker)) '+' '.join(fields),4096)
    assert row['pid']==42 and row['name']=='python (worker)'
    assert row['cpu_ticks']==200 and row['threads']==4 and row['rss_bytes']==40960
    child={**row,'pid':43,'ppid':42,'cpu_ticks':100}
    stranger={**row,'pid':44,'ppid':999}
    rows=module.tree_rows({42:row,43:child,44:stranger},42)
    assert [r['pid'] for r in rows]==[42,43]
    old={(42,99):{**row,'cpu_ticks':100},(43,98):child}
    assert module.rates(rows,old,1.,100)==1.
    assert rows[0]['cpu_percent_one_core']==100.
    assert rows[1]['cpu_percent_one_core'] is None
