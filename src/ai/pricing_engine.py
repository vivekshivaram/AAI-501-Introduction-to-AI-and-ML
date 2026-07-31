"""Operational pricing engine, CNN training, and tabular Q-learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ai.img_loader import ImageDatasetConfig, build_image_dataloaders
from src.ai.pricing_env import PricingEnv, PricingEnvConfig, load_demand_forecast, state_index_from_matrix
from src.config import ARTIFACTS_DIRECTORY, DATA_OUTPUTS_DIRECTORY, DEMAND_FORECAST_FILENAME
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.simulation.simulation_context import SimulationContext

logger = get_logger(__name__)


@dataclass(frozen=True)
class PackageCNNTrainingConfig:
    """Configuration for the damaged-box ResNet18 trainer."""

    data_root: Path
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    artifact_path: Path = ARTIFACTS_DIRECTORY / "package_cnn.pt"
    seed: int = 42


@dataclass(frozen=True)
class QLearningConfig:
    """Configuration for tabular Q-learning over the 5x5 pricing state space."""

    forecast_path: Path | None = None
    artifact_path: Path = ARTIFACTS_DIRECTORY / "q_table.npy"
    episodes: int = 240
    alpha: float = 0.15
    gamma: float = 0.92
    epsilon: float = 0.25
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.05
    max_queue_length: int = 60


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_tensor_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch.to(device=device, non_blocking=True)


def _load_resnet18(num_classes: int) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        try:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            model = resnet18(weights=None)
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "torchvision is required for ResNet18 training. Install it with `python3 -m pip install torchvision`."
        ) from exc

    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def _evaluate_cnn(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = _to_tensor_device(images, device)
            labels = labels.to(device=device, dtype=torch.float32).unsqueeze(1)
            logits = model(images)
            loss = loss_fn(logits, labels)
            losses.append(float(loss.item()))
            predictions = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.long).squeeze(1)
            correct += int((predictions == labels.squeeze(1).to(dtype=torch.long)).sum().item())
            total += int(labels.size(0))

    accuracy = correct / total if total else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy,
    }


def train_package_cnn(config: PackageCNNTrainingConfig) -> dict[str, float | str]:
    """Fine-tune a ResNet18 binary classifier for damaged-box inspection."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    bundle = build_image_dataloaders(
        ImageDatasetConfig(
            root_dir=config.data_root,
            image_size=config.image_size,
            batch_size=config.batch_size,
            validation_split=0.15,
            test_split=0.15,
            seed=config.seed,
            augment=True,
        )
    )
    if len(bundle.class_names) != 2:
        raise ValueError(
            f"Expected a binary damaged-box dataset, found classes: {bundle.class_names}"
        )

    device = _select_device()
    model = _load_resnet18(num_classes=1).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_validation_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for images, labels in bundle.train_loader:
            images = _to_tensor_device(images, device)
            labels = labels.to(device=device, dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.size(0))
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(sample_count, 1)
        validation_metrics = _evaluate_cnn(model, bundle.validation_loader, device)
        logger.info(
            "CNN epoch %s/%s | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
            epoch + 1,
            config.epochs,
            train_loss,
            validation_metrics["loss"],
            validation_metrics["accuracy"],
        )

        if validation_metrics["loss"] <= best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate_cnn(model, bundle.test_loader, device)
    config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": bundle.class_names,
            "image_size": config.image_size,
            "arch": "resnet18",
        },
        config.artifact_path,
    )

    return {
        "artifact_path": str(config.artifact_path),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
    }


def _load_q_table(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Q-table not found at {path}. Run `train_q_table()` first."
        )
    q_table = np.load(path)
    if q_table.shape != (25, 5):
        raise ValueError(f"Expected a (25, 5) Q-table, received shape {q_table.shape}.")
    return q_table.astype(np.float32)


def train_q_table(config: QLearningConfig) -> dict[str, float | str | np.ndarray]:
    """Train a tabular Q-learning policy using the pricing environment."""

    forecast = load_demand_forecast(config.forecast_path)
    env = PricingEnv(
        demand_forecast=forecast,
        config=PricingEnvConfig(max_queue_length=config.max_queue_length),
    )
    q_table = np.zeros((25, 5), dtype=np.float32)
    epsilon = config.epsilon

    for episode in range(config.episodes):
        initial_queue = episode % max(1, config.max_queue_length // 2)
        state, _ = env.reset(
            options={
                "initial_queue_length": initial_queue,
                "forecast_index": episode % len(forecast),
            }
        )

        terminated = False
        truncated = False
        while not (terminated or truncated):
            state_index = state_index_from_matrix(state)
            if np.random.random() < epsilon:
                action = int(np.random.randint(0, env.action_space.n))
            else:
                action = int(np.argmax(q_table[state_index]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_index = state_index_from_matrix(next_state)
            best_next_value = float(np.max(q_table[next_state_index]))
            td_target = reward + config.gamma * best_next_value
            q_table[state_index, action] += config.alpha * (
                td_target - q_table[state_index, action]
            )
            state = next_state

        epsilon = max(config.min_epsilon, epsilon * config.epsilon_decay)

    config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.artifact_path, q_table)
    logger.info("Q-table saved to %s", config.artifact_path)

    return {
        "artifact_path": str(config.artifact_path),
        "epsilon_final": float(epsilon),
        "q_table": q_table,
    }


@dataclass
class PricingEngine:
    """Operational wrapper that maps live simulation state to a pricing action."""

    q_table_path: Path = ARTIFACTS_DIRECTORY / "q_table.npy"
    forecast_path: Path | None = None
    env_config: PricingEnvConfig = field(default_factory=PricingEnvConfig)

    def __post_init__(self) -> None:
        self._forecast = load_demand_forecast(self.forecast_path)
        self._env = PricingEnv(demand_forecast=self._forecast, config=self.env_config)
        self._q_table = _load_q_table(self.q_table_path)

    @property
    def q_table(self) -> np.ndarray:
        return self._q_table

    @property
    def forecast(self) -> list[float]:
        return self._forecast

    def _demand_value_for_tick(self, tick: int | None = None) -> float:
        if tick is None:
            tick = 0
        return self._forecast[tick % len(self._forecast)]

    def select_action(
        self,
        queue_length: int,
        tick: int | None = None,
        demand_value: float | None = None,
    ) -> dict[str, float | int | np.ndarray]:
        demand_value = self._demand_value_for_tick(tick) if demand_value is None else demand_value
        state_matrix = self._env.encode_context(queue_length=queue_length, demand_value=demand_value)
        state_index = state_index_from_matrix(state_matrix)
        action_index = int(np.argmax(self._q_table[state_index]))
        surge_multiplier = float(self.env_config.action_multipliers[action_index])
        return {
            "action_index": action_index,
            "surge_multiplier": surge_multiplier,
            "state_index": state_index,
            "state_matrix": state_matrix,
            "demand_value": float(demand_value),
        }

    def act(self, state: np.ndarray | int) -> float:
        """Return the surge multiplier for the provided RL state."""

        if isinstance(state, np.ndarray):
            state_index = state_index_from_matrix(state)
        else:
            state_index = int(state)

        action_index = int(np.argmax(self._q_table[state_index]))
        return float(self.env_config.action_multipliers[action_index])

    def apply_to_context(self, context: "SimulationContext", tick: int | None = None) -> dict[str, float | int | np.ndarray]:
        queue_length = len(context.pending_orders)
        action = self.select_action(queue_length=queue_length, tick=tick or context.tick)
        context.surge_multiplier = float(action["surge_multiplier"])
        return action


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-q-table",
        action="store_true",
        help="Train and save the tabular pricing policy.",
    )
    parser.add_argument(
        "--train-cnn",
        type=Path,
        help="Train the damaged-box ResNet18 model from the given dataset root.",
    )
    parser.add_argument(
        "--q-table-path",
        type=Path,
        default=ARTIFACTS_DIRECTORY / "q_table.npy",
    )
    parser.add_argument(
        "--forecast-path",
        type=Path,
        default=DATA_OUTPUTS_DIRECTORY / DEMAND_FORECAST_FILENAME,
    )
    args = parser.parse_args()

    if args.train_cnn is not None:
        train_package_cnn(
            PackageCNNTrainingConfig(
                data_root=args.train_cnn,
                artifact_path=ARTIFACTS_DIRECTORY / "package_cnn.pt",
            )
        )

    if args.train_q_table or args.train_cnn is None:
        train_q_table(
            QLearningConfig(
                forecast_path=args.forecast_path,
                artifact_path=args.q_table_path,
            )
        )


if __name__ == "__main__":
    main()