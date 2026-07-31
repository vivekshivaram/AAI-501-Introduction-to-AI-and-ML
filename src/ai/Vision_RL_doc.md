# Vision & RL Doc

This document explains the code under `src/ai/` step by step and maps it back to the project architecture.

## 1. What This Folder Does

The `src/ai/` package handles the final two intelligent layers in LogiSim-AI:

1. Package inspection with a ResNet18-based binary classifier.
2. Surge pricing with a tabular Q-learning policy.

Together, these modules read the demand forecast, inspect package images, and update the live simulation context with a pricing decision.

## 2. File Overview

### `src.ingestion.download_damaged_box.py`

Downloads the damaged-box Kaggle dataset and copies it into `data/raw/damaged_box`.

Main responsibilities:

1. Use KaggleHub to fetch the dataset.
2. Place the extracted tree under `data/raw/damaged_box`.
3. Skip the download if the folder already exists unless `--force` is used.

### `img_loader.py`

Builds a PyTorch dataset and dataloaders for the damaged-box image classification task.

Expected folder layout:

```text
data/raw/damaged_box/
├── intact/
│   ├── img1.jpg
│   └── ...
└── damaged/
    ├── img2.jpg
    └── ...
```

Main responsibilities:

1. Find class subdirectories under the dataset root.
2. Collect image files from each class folder.
3. Resize every image to the configured square size, default `224 x 224`.
4. Apply lightweight augmentation during training.
5. Split the dataset into train, validation, and test subsets.
6. Return PyTorch dataloaders so training code can iterate batch by batch.

### `pricing_env.py`

Implements the reinforcement-learning environment for dynamic pricing.

Main responsibilities:

1. Load the 24-hour demand forecast from `data/outputs/demand_forecast.json`.
2. Convert queue length and demand level into a 5x5 one-hot state matrix.
3. Expose 5 discrete actions that correspond to surge multipliers from `1.0` to `1.5`.
4. Simulate how a pricing action changes arrivals, served orders, and queue growth.
5. Produce a reward based on revenue minus delay penalty.
6. Provide architecture-compatible hooks:
   - `update(context)` to build the current RL state from the live simulation.
   - `step(action)` for training.

### `pricing_engine.py`

Contains the training and runtime logic for the two learned models.

Main responsibilities:

1. Fine-tune ResNet18 and save the trained package inspector.
2. Train a Q-table over the 25 pricing states and 5 actions.
3. Load the saved Q-table at runtime.
4. Translate a state matrix into a surge multiplier.
5. Apply the selected multiplier back into `SimulationContext`.

## 3. Step-by-Step Approach

## Step 1: Load package images

The dataset loader in `img_loader.py` assumes a binary classification problem.

1. A user points the loader at the dataset root, typically `data/raw/damaged_box`.
2. The loader discovers class directories such as `intact` and `damaged`.
3. Each image file is opened, converted to RGB, and resized.
4. Training samples can receive simple augmentation, such as flips and small rotations.
5. The samples are split into train, validation, and test subsets.
6. PyTorch dataloaders are built from those subsets.

Why this structure matters:

1. It keeps data loading separate from model training.
2. It makes the code reusable for experiments.
3. It avoids hardcoding dataset paths or labels in the model code.

## Step 2: Train the ResNet18 inspector

The CNN training path lives in `train_package_cnn()` inside `pricing_engine.py`.

1. Build the dataloaders with `build_image_dataloaders()`.
2. Verify that the dataset is binary.
3. Load a ResNet18 backbone.
4. Freeze all pretrained backbone parameters.
5. Replace the final classification head with a binary output layer.
6. Train only the final layer at first, which keeps training lightweight.
7. Evaluate on validation data after every epoch.
8. Keep the best validation checkpoint in memory.
9. Evaluate the final model on the test set.
10. Save the trained bundle to `artifacts/package_cnn.pt`.

Architecture fit:

1. The inspector is the “Inspect” stage in the project overview.
2. It is meant to reject damaged packages before dispatch.
3. Its output can be attached to each order as `inspection_passed`.

## Step 3: Build the pricing state

The RL environment in `pricing_env.py` turns the current simulation situation into a compact state.

1. Read queue length from the live simulation context.
2. Read the hourly demand forecast for the current tick.
3. Bucket queue length into 5 bands.
4. Bucket demand into 5 bands using forecast quantiles.
5. Combine both buckets into a `5 x 5` one-hot matrix.

That matrix is the state representation used by the Q-table.

Why a 5x5 matrix:

1. It is small enough for tabular Q-learning.
2. It still captures two important signals: queue pressure and demand pressure.
3. It matches the architecture note in the README and the design diagram.

## Step 4: Define the pricing action space

The environment exposes 5 actions.

1. Action 0 means no surge: `1.0x`.
2. Action 1 means mild surge: `1.125x`.
3. Action 2 means medium surge: `1.25x`.
4. Action 3 means higher surge: `1.375x`.
5. Action 4 means maximum surge: `1.5x`.

These actions represent the pricing lever that the RL agent controls.

## Step 5: Define the reward signal

The training environment rewards actions using a simple business rule:

```text
Reward = Revenue - Delay
```

In code, the reward is computed from:

1. Served orders.
2. Surge multiplier.
3. Queue growth.
4. Delay penalty per waiting order.

Why this works:

1. Revenue encourages the model to raise prices when demand is strong.
2. Delay penalty discourages the queue from growing without bound.
3. The policy learns a tradeoff instead of always choosing the largest multiplier.

## Step 6: Train the Q-table

The Q-learning routine is `train_q_table()` in `pricing_engine.py`.

1. Load the demand forecast.
2. Create a `PricingEnv` instance.
3. Allocate a `25 x 5` Q-table.
4. Run repeated episodes with epsilon-greedy exploration.
5. For each state, choose either a random action or the best known action.
6. Step the environment.
7. Update the Q-value with the Bellman rule.
8. Decay exploration over time.
9. Save the table to `artifacts/q_table.npy`.

Why the table has shape `25 x 5`:

1. There are 25 states because the environment uses 5 queue buckets and 5 demand buckets.
2. There are 5 actions because the pricing policy supports 5 surge multipliers.

## Step 7: Use the pricing engine at runtime

`PricingEngine` is the operational wrapper used by the simulation.

1. It loads the saved Q-table from disk.
2. It loads the demand forecast.
3. It converts the current queue and demand into a state matrix.
4. It finds the best action from the Q-table.
5. It converts that action into a surge multiplier.
6. It can write the selected multiplier into `context.surge_multiplier`.

The runtime hooks are:

1. `select_action(queue_length, tick, demand_value)` for explicit action lookup.
2. `act(state)` for architecture compatibility with the executor.
3. `apply_to_context(context)` for direct simulation integration.

## 4. How The Pieces Connect

The flow is:

1. Orders enter the simulation.
2. Package images are inspected by the CNN.
3. Demand forecast data is loaded from JSON.
4. The environment converts queue pressure and demand pressure into a state matrix.
5. The Q-table chooses a pricing action.
6. The pricing engine writes the chosen multiplier into the simulation context.

This keeps the learning logic isolated from the simulation engine while still letting the simulation consume the result each tick.

## 5. Runtime Contract With The Simulation

The architecture expects the RL layer to behave like this:

1. The environment builds the state from the simulation context.
2. The agent reads that state and returns a surge multiplier.
3. The simulation context stores the multiplier for downstream components.

In this implementation:

1. `PricingEnv.update(context)` returns the state matrix.
2. `PricingEngine.act(state)` returns the surge multiplier.
3. `PricingEngine.apply_to_context(context)` writes the multiplier back into `context.surge_multiplier`.

## 6. Example Commands

Train the image classifier:

```bash
python -m src.ai.pricing_engine --train-cnn data/raw/damaged_box
```

Train the Q-table:

```bash
python -m src.ai.pricing_engine --train-q-table
```

Inspect the dataset summary:

```bash
python -m src.ai.img_loader data/raw/damaged_box
```

## 7. Artifacts Produced

1. `artifacts/package_cnn.pt` stores the trained ResNet18 weights and metadata.
2. `artifacts/q_table.npy` stores the learned pricing policy.
3. `data/outputs/demand_forecast.json` supplies the demand signal used by the RL layer.

## 8. Implementation Notes

1. The code is written to work even if `gymnasium` is not installed, through a small fallback stub.
2. The forecast loader falls back to a synthetic series when the forecast file is missing, which makes local experimentation easier.
3. The package dataset loader expects binary class folders, which keeps the CNN task simple and explicit.
4. The pricing environment uses a compact state space so tabular Q-learning stays practical.

## 9. Recommended Next Step

If you want the RL layer to participate in the full simulation loop automatically, the next integration point is `src/simulation/simulation_executor.py`.