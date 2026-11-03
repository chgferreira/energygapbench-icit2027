import logging

from energygap.runtime import startup
from energygap.runtime.state import StateStore
import energygap.conference as runner


def test_watchdog_is_optional_and_cancels_only_its_own_timer(monkeypatch):
    calls=[]
    monkeypatch.setattr(startup,'_armed',False)
    monkeypatch.setattr(startup.faulthandler,'dump_traceback_later',lambda seconds,**kw:calls.append(('begin',seconds,kw)))
    monkeypatch.setattr(startup.faulthandler,'cancel_dump_traceback_later',lambda:calls.append(('cancel',)))
    monkeypatch.delenv('ENERGYGAP_STARTUP_TRACE_SECONDS',raising=False)
    startup.begin_startup_trace();startup.finish_startup_trace()
    assert calls==[]
    monkeypatch.setenv('ENERGYGAP_STARTUP_TRACE_SECONDS','60')
    startup.begin_startup_trace();startup.finish_startup_trace();startup.finish_startup_trace()
    assert calls==[('begin',60.,{'repeat':True}),('cancel',)]


def test_startup_message_precedes_scientific_import_failure(tmp_path,monkeypatch,capsys,caplog):
    import builtins
    from energygap.cli import main
    real_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='energygap.pipeline':raise RuntimeError('simulated scientific import failure')
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    caplog.set_level(logging.INFO)
    assert main(['--root',str(tmp_path),'run','--profile','conference','--device','cpu'])==2
    assert 'EnergyGap: comando=run' in caplog.text
    assert 'importando executor' in caplog.text
    assert 'simulated scientific import failure' in capsys.readouterr().out


def test_global_progress_batches_refresh_without_losing_timings_or_final_counts(tmp_path,monkeypatch):
    clock=[0.]
    from energygap.runtime import conference_progress
    monkeypatch.setattr(conference_progress,'perf_counter',lambda:clock[0])
    with StateStore(tmp_path/'run',{'test':'progress'}) as store:
        with store.defer_snapshots(),store.transaction():
            store.add_job('a',10000,metadata={'signature':'query'})
            store.add_job('b',0,metadata={'signature':'catalogue'})
            store.mark('b','unavailable','test catalogue entry')
        display=runner.GlobalProgress(store,'none')
        queries=[];store.db.set_trace_callback(queries.append)
        first=display.update('query',1,.1)
        before=len(queries)
        for _ in range(9999):display.update('query',1,.1)
        assert len(queries)==before  # no SQL scan or fsync on every fast unit
        store.db.execute("UPDATE jobs SET completed_units=10000,status='succeeded' WHERE job_id='a'")
        display.close()
        eta=store.get_meta('eta');timing=store.get_meta('conference_timings')['query']
        assert timing['units']==10000
        assert abs(timing['seconds']-1000.)<1e-8
        assert eta['remaining_seconds']==0 and eta['resolved_jobs']==2
        assert first['remaining_seconds'] is not None
