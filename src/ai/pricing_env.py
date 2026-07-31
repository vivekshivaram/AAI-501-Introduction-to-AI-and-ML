"""Gym-compatible pricing environment for queue and demand control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import json
import math
import types

import numpy as np

from src.config import DATA_OUTPUTS_DIRECTORY, DEMAND_FORECAST_FILENAME
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - exercised when gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - lightweight local fallback
    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _Box:
        def __init__(self, low, high, shape, dtype) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            return np.zeros(self.shape, dtype=self.dtype)

    class _Env:
        metadata: dict[str, object] = {}

    gym = types.SimpleNamespace(Env=_Env)
    spaces = types.SimpleNamespace(Discrete=_Discrete, Box=_Box)


@dataclass(frozen=True)
class PricingEnvConfig:
    """Configuration for the queue-based pricing environment."""

    queue_bucket_edges: tuple[int, int, int, int] = (0, 5, 10, 20)
    demand_bucket_quantiles: tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)
    action_multipliers: tuple[float, ...] = (1.0, 1.125, 1.25, 1.375, 1.5)
    base_order_value: float = 100.0
    delay_cost_per_order: float = 8.0
    service_capacity: int = 12
    demand_elasticity: float = 0.12
    max_queue_length: int = 60
    episode_horizon: int = 24


def load_demand_forecast(path: Path | None = None) -> list[float]:
    """Load the 24-hour demand forecast or synthesize a safe fallback."""

    path = path or (DATA_OUTPUTS_DIRECTORY / DEMAND_FORECAST_FILENAME)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        forecast = payload.get("hourly_forecast", [])
        values = [max(0.0, float(value)) for value in forecast]
        if values:
            return values

    logger.warning("Demand forecast not found at %s; synthesizing fallback series.", path)
    hours = np.arange(24, dtype=float)
    synthetic = 14.0 + 5.0 * np.sin((hours - 7.0) * math.pi / 12.0)
    synthetic = np.maximum(0.0, synthetic)
    return synthetic.round(2).tolist()


class PricingEnv(gym.Env):
    """A 5x5 pricing environment with 5 surge-multiplier actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        demand_forecast: Iterable[float] | None = None,
        config: PricingEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PricingEnvConfig()
        self.demand_forecast = [float(value) for value in (demand_forecast or load_demand_forecast())]
        if not self.demand_forecast:
            self.demand_forecast = load_demand_forecast()

        self._queue_length = 0
        self._forecast_index = 0
        self._episode_step = 0
        self._demand_thresholds = self._build_demand_thresholds()

        self.action_space = spaces.Discrete(len(self.config.action_multipliers))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, 5),
            dtype=np.float32,
        )

    def _build_demand_thresholds(self) -> tuple[float, float, float, float]:
        forecast = np.asarray(self.demand_forecast, dtype=float)
        if forecast.size == 0:
            return (5.0, 10.0, 15.0, 20.0)

        quantiles = np.quantile(forecast, self.config.demand_bucket_quantiles)
        return tuple(float(value) for value in quantiles)

    def _bucket_from_thresholds(self, value: float, thresholds: tuple[float, ...]) -> int:
        for index, threshold in enumerate(thresholds):
            if value <= threshold:
                return index
        return len(thresholds)

    def queue_bucket(self, queue_length: int | None = None) -> int:
        queue_value = self._queue_length if queue_length is None else queue_length
        return self._bucket_from_thresholds(queue_value, self.config.queue_bucket_edges)

    def demand_bucket(self, demand_value: float | None = None) -> int:
        if demand_value is None:
            demand_value = self.demand_forecast[self._forecast_index % len(self.demand_forecast)]
        return self._bucket_from_thresholds(demand_value, self._demand_thresholds)

    def state_index(self, queue_length: int | None = None, demand_value: float | None = None) -> int:
        queue_bucket = self.queue_bucket(queue_length)
        demand_bucket = self.demand_bucket(demand_value)
        return queue_bucket * 5 + demand_bucket

    def state_matrix(
        self,
        queue_length: int | None = None,
        demand_value: float | None = None,
    ) -> np.ndarray:
        matrix = np.zeros((5, 5), dtype=np.float32)
        matrix[self.queue_bucket(queue_length), self.demand_bucket(demand_value)] = 1.0
        return matrix

    def _demand_value_for_step(self) -> float:
        return self.demand_forecast[self._forecast_index % len(self.demand_forecast)]

    def _arrival_volume(self, demand_value: float, multiplier: float) -> int:
        adjusted = demand_value * max(0.35, 1.0 - self.config.demand_elasticity * (multiplier - 1.0))
        return int(round(max(0.0, adjusted)))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        initial_queue = int(options.get("initial_queue_length", 0))
        initial_forecast_index = int(options.get("forecast_index", 0))
        self._queue_length = max(0, min(self.config.max_queue_length, initial_queue))
        self._forecast_index = initial_forecast_index % len(self.demand_forecast)
        self._episode_step = 0
        observation = self.state_matrix()
        info = {
            "queue_length": self._queue_length,
            "demand_value": self._demand_value_for_step(),
            "state_index": self.state_index(),
        }
        return observation, info

    def step(self, action: int):
        if not 0 <= int(action) < self.action_space.n:
            raise ValueError(f"Action must be in [0, {self.action_space.n - 1}].")

        action_index = int(action)
        multiplier = float(self.config.action_multipliers[action_index])
        demand_value = self._demand_value_for_step()
        arrivals = self._arrival_volume(demand_value, multiplier)

        total_work = self._queue_length + arrivals
        served = min(total_work, self.config.service_capacity)
        next_queue_length = max(0, total_work - self.config.service_capacity)
        next_queue_length = min(self.config.max_queue_length, next_queue_length)

        revenue = served * self.config.base_order_value * multiplier
        delay_penalty = next_queue_length * self.config.delay_cost_per_order
        reward = float(revenue - delay_penalty)

        self._queue_length = next_queue_length
        self._forecast_index = (self._forecast_index + 1) % len(self.demand_forecast)
        self._episode_step += 1

        observation = self.state_matrix()
        terminated = False
        truncated = self._episode_step >= self.config.episode_horizon
        info = {
            "action_index": action_index,
            "surge_multiplier": multiplier,
            "queue_length": self._queue_length,
            "demand_value": demand_value,
            "served": served,
            "arrivals": arrivals,
            "revenue": float(revenue),
            "delay_penalty": float(delay_penalty),
            "state_index": self.state_index(),
        }
        return observation, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - no visual renderer yet
        return {
            "queue_length": self._queue_length,
            "forecast_index": self._forecast_index,
            "demand_value": self._demand_value_for_step(),
        }

    def encode_context(self, queue_length: int, demand_value: float) -> np.ndarray:
        return self.state_matrix(queue_length=queue_length, demand_value=demand_value)

    def describe_state(self, queue_length: int | None = None, demand_value: float | None = None) -> dict[str, int]:
        queue_bucket = self.queue_bucket(queue_length)
        demand_bucket = self.demand_bucket(demand_value)
        return {
            "queue_bucket": queue_bucket,
            "demand_bucket": demand_bucket,
            "state_index": queue_bucket * 5 + demand_bucket,
        }

    def update(self, context) -> np.ndarray:
        """Build the RL state matrix from the live simulation context."""

        queue_length = len(getattr(context, "pending_orders", []))
        demand_value = self.demand_forecast[getattr(context, "tick", 0) % len(self.demand_forecast)]
        observation = self.state_matrix(queue_length=queue_length, demand_value=demand_value)
        self._queue_length = queue_length
        self._forecast_index = getattr(context, "tick", 0) % len(self.demand_forecast)
        return observation


def state_index_from_matrix(state_matrix: np.ndarray) -> int:
    """Recover the tabular state index from a one-hot 5x5 matrix."""

    if state_matrix.shape != (5, 5):
        raise ValueError(f"Expected a 5x5 matrix, received shape {state_matrix.shape}.")
    row, col = np.unravel_index(int(np.argmax(state_matrix)), state_matrix.shape)
    return row * 5 + col