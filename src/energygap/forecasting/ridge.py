"""One forecaster per fold; no targets fabricated by interpolation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve


def causal_fill(values, initial: float = 0.0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Causal fill requires a one-dimensional series")
    initial = float(initial)
    finite = np.isfinite(values)
    if finite.all():
        return values.copy()
    # Index zero is the causal initial value. Nonfinite entries inherit the
    # latest finite source index; zeros and their sign remain observations.
    latest = np.where(finite, np.arange(1, len(values) + 1), 0)
    np.maximum.accumulate(latest, out=latest)
    return np.r_[initial, values][latest]


def _causal_fill_rows(values, initial):
    """Independent causal filling of a dense [histories, history] matrix."""
    finite = np.isfinite(values)
    if finite.all():
        return values
    latest = np.where(finite, np.arange(1, values.shape[1] + 1), 0)
    np.maximum.accumulate(latest, axis=1, out=latest)
    sources = np.concatenate((np.full((len(values), 1), float(initial)), values), axis=1)
    return np.take_along_axis(sources, latest, axis=1)


def _calendar_features(issuances):
    phase = np.asarray(issuances, float) / 86400 * 2 * np.pi
    return np.column_stack((np.sin(phase), np.cos(phase), np.sin(phase / 7), np.cos(phase / 7)))


def match_target(times, observed, issuance: float, horizon_seconds: float, tolerance: float = 90):
    """Return nearest ORIGINAL target index and signed deviation, deterministic ties."""
    t = np.asarray(times, dtype=float)
    target = issuance + horizon_seconds
    eligible = np.flatnonzero(np.asarray(observed, bool) & (t > issuance) & (np.abs(t-target) <= tolerance))
    if not len(eligible):
        return None, None
    index = min(eligible, key=lambda i: (abs(t[i]-target), t[i], int(i)))
    return int(index), float(t[index]-target)


@dataclass
class DirectRidge:
    history: int = 288
    horizon_steps: int = 12
    alpha: float = 1.0
    device: str = "cpu"
    backend: str = "auto"

    def _configure_execution(self):
        from energygap.runtime.device import resolve_device

        if self.backend not in {"auto", "scipy", "torch"}:
            raise ValueError("Ridge backend must be auto, scipy, or torch")
        self.execution_device = resolve_device(self.device)
        self.execution_backend = (
            "torch" if self.execution_device != "cpu" else "scipy"
        ) if self.backend == "auto" else self.backend
        if self.execution_backend == "scipy" and self.execution_device != "cpu":
            raise ValueError("The scipy ridge backend requires device='cpu'")
        self.__dict__.pop("_torch_state", None)

    def __getstate__(self):
        # Checkpoints store portable arrays, never CUDA tensors or CUDA contexts.
        state = self.__dict__.copy()
        state.pop("_torch_state", None)
        return state

    def _torch_parameters(self):
        if "_torch_state" not in self.__dict__:
            import torch
            from energygap.runtime.device import resolve_device

            device = resolve_device(self.execution_device)
            self._torch_state = {
                name: torch.as_tensor(getattr(self, name), dtype=torch.float64, device=device)
                for name in ("mean", "scale", "coef")
            }
        return self._torch_state

    def features(self, history, issuance):
        values = np.asarray(history, float)
        if len(values) < self.history:
            values = np.pad(values, (self.history-len(values), 0), constant_values=self.initial)
        values = causal_fill(values[-self.history:], self.initial)
        phase = float(issuance) / 86400 * 2*np.pi
        return np.r_[values, np.sin(phase), np.cos(phase), np.sin(phase/7), np.cos(phase/7)]

    def features_many(self, histories, issuances):
        """Vectorized scalar-equivalent features for dense or ragged histories.

        Truncate before causal filling, exactly as in ``features``. A value
        preceding the retained history cannot fill a leading gap in that tail.
        """
        if not isinstance(histories, np.ndarray):
            histories = list(histories)
        times = np.asarray(issuances if isinstance(issuances, np.ndarray) else list(issuances), float)
        if times.ndim != 1 or len(histories) != len(times):
            raise ValueError("One issuance per history is required")
        if not len(histories):
            return np.empty((0, self.history + 4), float)
        try:
            dense = np.asarray(histories, float)
        except ValueError:
            dense = None
        if dense is not None and dense.ndim == 2:
            if dense.shape[1] < self.history:
                values = np.pad(dense, ((0, 0), (self.history - dense.shape[1], 0)),
                                constant_values=self.initial)
            else:
                values = dense[:, -self.history:]
        else:
            values = np.full((len(histories), self.history), float(self.initial))
            # Ragged row assembly is O(batch), with no Python scan over time.
            for row, history in enumerate(histories):
                history = np.asarray(history, float)
                if history.ndim != 1:
                    raise ValueError("Each history must be one-dimensional")
                count = min(len(history), self.history)
                if count:
                    values[row, -count:] = history[-count:]
        filled = _causal_fill_rows(values, self.initial)
        return np.concatenate((filled, _calendar_features(times)), axis=1)

    def fit(self, values, times, *, split="train", target_bounds=None):
        if split != "train":
            raise ValueError("Forecaster fit requires training data only")
        self._configure_execution()
        y = np.asarray(values, float)
        t = np.asarray(times, float)
        finite = y[np.isfinite(y)]
        if not len(finite):
            raise ValueError("No observed training targets")
        # An initial constant is explicitly causal; it does not use later training rows.
        self.initial = 0.0
        filled = causal_fill(y, self.initial)
        issuances = np.arange(self.history - 1, len(y) - self.horizon_steps)
        targets = issuances + self.horizon_steps
        # Labels remain ORIGINAL finite targets. Filling only supplies histories.
        eligible = np.isfinite(y[targets])
        if target_bounds:
            eligible &= (target_bounds[0] <= t[targets]) & (t[targets] < target_bounds[1])
        # The inclusive 90-second physical-horizon rule retains record order.
        eligible &= ~(np.abs(t[targets] - t[issuances] - self.horizon_steps * 300) > 90)
        issuances, targets = issuances[eligible], targets[eligible]
        if len(issuances) < 2:
            raise ValueError("Insufficient original training targets for direct ridge")
        windows = np.lib.stride_tricks.sliding_window_view(filled, self.history)
        rows = windows[issuances - self.history + 1]
        x = np.concatenate((rows, _calendar_features(t[issuances])), axis=1)
        labels = y[targets]
        if self.execution_backend == "torch":
            import torch
            from energygap.runtime.device import synchronize

            # Preserve the CPU forecaster's float64 arithmetic and population std.
            with torch.no_grad():
                tensor = torch.as_tensor(x, dtype=torch.float64, device=self.execution_device)
                targets = torch.as_tensor(labels, dtype=torch.float64, device=self.execution_device)
                mean = tensor.mean(dim=0)
                scale = tensor.std(dim=0, correction=0)
                scale = torch.where(scale < 1e-10, torch.ones_like(scale), scale)
                z = (tensor-mean)/scale
                intercept = targets.mean()
                gram = z.T@z+self.alpha*torch.eye(z.shape[1], dtype=torch.float64, device=tensor.device)
                coef = torch.linalg.solve(gram, z.T@(targets-intercept))
                synchronize(self.execution_device)
                self.mean = mean.cpu().numpy()
                self.scale = scale.cpu().numpy()
                self.intercept = float(intercept.cpu())
                self.coef = coef.cpu().numpy()
                self._torch_state = {"mean": mean, "scale": scale, "coef": coef}
        else:
            self.mean = x.mean(axis=0)
            self.scale = x.std(axis=0)
            self.scale[self.scale < 1e-10] = 1
            z = (x-self.mean)/self.scale
            self.intercept = float(np.mean(labels))
            self.coef = solve(z.T@z+self.alpha*np.eye(z.shape[1]), z.T@(np.asarray(labels)-self.intercept), assume_a="pos")
        self.n_training_targets = len(labels)
        return self

    def predict(self, history, issuance):
        x = self.features(history, issuance)
        if getattr(self, "execution_backend", "scipy") == "torch":
            import torch
            from energygap.runtime.device import synchronize

            state = self._torch_parameters()
            with torch.no_grad():
                tensor = torch.as_tensor(x, dtype=torch.float64, device=self.execution_device)
                result = (tensor-state["mean"])/state["scale"]@state["coef"]+self.intercept
                synchronize(self.execution_device)
                return float(result.cpu())
        return float((x-self.mean)/self.scale@self.coef+self.intercept)

    def predict_many(self, histories, issuances):
        """Batch transfers/linear algebra, preserving the scalar feature contract."""
        x=self.features_many(histories, issuances)
        if not len(x):return np.empty(0)
        if getattr(self,'execution_backend','scipy')=='torch':
            import torch
            from energygap.runtime.device import synchronize
            state=self._torch_parameters()
            with torch.no_grad():
                tensor=torch.as_tensor(x,dtype=torch.float64,device=self.execution_device)
                result=(tensor-state['mean'])/state['scale']@state['coef']+self.intercept
                synchronize(self.execution_device)
                return result.cpu().numpy()
        return (x-self.mean)/self.scale@self.coef+self.intercept


def select_alpha(train_values, train_times, validation_values, validation_times, candidates=(0.1,1.,10.), *, device="cpu", backend="auto",horizon_steps=12):
    """Validation uses original histories with one common causal natural-gap filler."""
    scores = []
    for alpha in candidates:
        model = DirectRidge(alpha=alpha, device=device, backend=backend,horizon_steps=horizon_steps).fit(train_values, train_times)
        combined = np.r_[train_values, validation_values]
        times = np.r_[train_times, validation_times]
        targets=[];histories=[];issuances=[]
        start = len(train_values)
        for j in range(start, len(combined)):
            i = j-model.horizon_steps
            if i < 0 or not np.isfinite(combined[j]) or abs(times[j]-times[i]-horizon_steps*300)>90:
                continue
            histories.append(combined[max(0,i-287):i+1]);issuances.append(times[i]);targets.append(combined[j])
        errors=np.abs(model.predict_many(histories,issuances)-np.asarray(targets))
        scores.append((float(np.mean(errors)) if len(errors) else float("inf"), alpha))
    loss, selected = min(scores)
    if not np.isfinite(loss):
        raise ValueError("No admissible validation forecasting targets")
    return selected, scores
