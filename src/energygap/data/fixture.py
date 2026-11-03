"""Labeled deterministic synthetic fixture; never represented as residential data."""
from datetime import datetime, timedelta, timezone
from math import sin, pi, nan
import random
from .schema import CanonicalRecord, SeriesData


def _segment(a,b,dt):
    if a>=0 and b>=0: return .5*(a+b)*dt,0.
    if a<=0 and b<=0: return 0.,-.5*(a+b)*dt
    fraction=abs(a)/(abs(a)+abs(b))
    if a>0: return .5*a*dt*fraction,.5*(-b)*dt*(1-fraction)
    return .5*b*dt*(1-fraction),.5*(-a)*dt*fraction


def make_fixture(days=35, step_minutes=5, seed=7, site_id='SYNTH', natural_missing=True):
    rng=random.Random(seed)
    start=datetime(2024,1,1,tzinfo=timezone.utc)
    n=int(days*24*60/step_minutes)
    values=[]
    for i in range(n):
        hour=(i*step_minutes/60)%24
        # Includes import/export, abrupt changes, zeros, calendar structure, noise.
        p=1.8+sin(2*pi*hour/24)-3.6*max(0,sin(pi*(hour-6)/12))+.08*rng.gauss(0,1)
        p += .5 if (i//int(24*60/step_minutes))%7>=5 else 0
        if i%2003 in (400,401): p=0.
        values.append(p)
    imp=exp=0.; rows=[]
    resolution=.001
    for i,p in enumerate(values):
        if i:
            a,b=_segment(values[i-1],p,step_minutes/60)
            imp+=a; exp+=b
        t=start+timedelta(minutes=i*step_minutes)
        missing=natural_missing and i%1777 in (1000,1001,1002)
        rows.append(CanonicalRecord(site_id,'synthetic-v1',str(i),t,t,t,nan if missing else p,nan if missing else p,observed_original=not missing,natural_missing=missing,available_to_method=not missing,auxiliaries={'temperature':20+4*sin(2*pi*(i*step_minutes/60)/24)},import_counter=None if missing else round(imp/resolution)*resolution,export_counter=None if missing else round(exp/resolution)*resolution,counter_availability_time=t))
    return SeriesData(rows,site_id,step_minutes*60,{'synthetic':True,'power_unit':'kW','power_semantics':'instantaneous','import_sign':'positive','counter_unit':'kWh','counter_resolution':resolution,'counter_model':'simulated cumulative readout quantized at 0.001 kWh; underlying integral is fixture-defined','availability_assumption':'synthetic synchronous observations'})
