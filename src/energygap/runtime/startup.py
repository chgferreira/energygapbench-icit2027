"""Optional startup traceback watchdog; no models or numerical settings change."""
from __future__ import annotations

import faulthandler
import logging
import math
import os

_armed=False


def begin_startup_trace():
    global _armed
    seconds=float(os.environ.get('ENERGYGAP_STARTUP_TRACE_SECONDS','0'))
    if not math.isfinite(seconds) or seconds<0:
        raise ValueError('ENERGYGAP_STARTUP_TRACE_SECONDS must be finite and nonnegative')
    if seconds:
        faulthandler.dump_traceback_later(seconds,repeat=True)
        _armed=True
        logging.getLogger('energygap.startup').info(
            'Diagnóstico ativo: se a inicialização demorar, a pilha Python será gravada a cada %.0f s.',seconds)


def finish_startup_trace():
    global _armed
    if _armed:
        faulthandler.cancel_dump_traceback_later()
        _armed=False
