"""Presentation reconstructed from durable state; no model imports."""
from __future__ import annotations
import json
import logging
from typing import TextIO

class Progress:
    def __init__(self, total: int, initial: int = 0, mode: str = 'none', desc: str = '', stream: TextIO | None = None):
        if not 0 <= initial <= total: raise ValueError('invalid durable progress')
        if mode not in {'tqdm','log','none'}: raise ValueError(mode)
        self.total, self.initial, self.confirmed, self.mode = total, initial, initial, mode
        self.bar = None
        if mode == 'tqdm':
            from tqdm.auto import tqdm
            self.bar = tqdm(total=total, initial=initial, desc=desc, file=stream,
                            unit='units', mininterval=1, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
            self.bar.refresh()
        elif mode == 'log': logging.getLogger('energygap.progress').info(json.dumps({'total':total,'initial':initial,'desc':desc}))

    def update(self, confirmed: int, eta: dict | None = None):
        if not self.confirmed <= confirmed <= self.total: raise ValueError('progress must be monotone in a session')
        if self.bar:
            self.bar.update(confirmed-self.confirmed)
            if eta:
                self.bar.set_postfix_str('ETA=' + str(eta.get('remaining_seconds')) + 's finish=' + str(eta.get('estimated_finish')), refresh=False)
        elif self.mode == 'log': logging.getLogger('energygap.progress').info(json.dumps({'confirmed':confirmed,'total':self.total,'eta':eta}))
        self.confirmed = confirmed

    def close(self):
        if self.bar: self.bar.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
