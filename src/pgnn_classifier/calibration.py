from __future__ import annotations

import math

import torch
from torch.nn import functional as F


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iterations: int = 100) -> float:
        logits = logits.detach().float().cpu()
        labels = labels.detach().float().cpu()
        valid = (labels >= 0) & torch.isfinite(logits)
        logits, labels = logits[valid], labels[valid]
        if logits.numel() == 0 or labels.unique().numel() < 2:
            self.temperature = 1.0
            return self.temperature

        log_temperature = torch.tensor(math.log(self.temperature), requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature], lr=0.1, max_iter=max_iterations, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature.clamp(-5.0, 5.0))
            loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature.detach().clamp(-5.0, 5.0)).item())
        return self.temperature

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def state_dict(self) -> dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_state_dict(cls, state: dict[str, float]) -> "TemperatureScaler":
        return cls(float(state["temperature"]))
