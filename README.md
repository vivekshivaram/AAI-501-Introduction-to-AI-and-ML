# LogiSim-AI

**Course:** AAI-501 — Introduction to Artificial Intelligence and Machine Learning  
**Program:** MS in Applied Artificial Intelligence — University of San Diego

A tick-based last-mile delivery simulation that combines geospatial data engineering, machine learning, and reinforcement learning to model intelligent routing and dynamic pricing decisions.

---

## What It Does

Each simulation tick models one real-world dispatch cycle:

1. **Sample** — Orders are drawn from an Amazon Last-Mile Delivery dataset and snapped to road-network nodes.
2. **Inspect** — A ResNet18 CNN classifies package images; damaged packages are rejected.
3. **Predict** — A Random Forest Regressor injects predicted delay penalties into road-edge weights.
4. **Dispatch** — An A\* router (with ML-fused costs) finds the optimal path for each vehicle.
5. **RL Update** — A Q-Learning agent reads the queue state and sets the surge price multiplier.
6. **Move** — Vehicles advance one edge along their assigned routes.

> See [ARCHITECTURE.md](ARCHITECTURE.md) for full data-flow diagrams, module dependency graphs, and architectural notes.

---

## Project Structure

```
AAI-501-Introduction-to-AI-and-ML/
├── data/
│   ├── raw/            # amazon_delivery.csv · static_city_map.graphml
│   ├── interim/        # mapped_orders.pkl
│   └── outputs/        # demand_forecast.json · diagnostic plots
├── artifacts/          # Trained model bundles (gitignored)
│   ├── delay_forest.pkl        # RandomForest regressor — delay prediction
│   ├── package_cnn.pt          # ResNet18 binary classifier — package inspection  [TODO]
│   └── q_table.npy             # Q-Learning weight matrix — surge pricing         [TODO]
├── logs/               # simulation.log (auto-created)
└── src/
    ├── config.py        # single source of truth for all paths & constants
    ├── ingestion/       # create_city_map.py · snap_nodes.py
    ├── graph/           # graph.py · edge.py · node.py
    ├── models/          # order.py · vehicle.py · states.py
    ├── analytics/       # delay_model.py · demand_forecast.py
    ├── routing/         # dynamic_astar.py
    ├── optimization/    # astar_routing.py · cvrptw_dispatcher.py · cvrptw_dispatcher_milp.py · dispatcher_config.py
    ├── ai/              # img_loader.py · pricing_env.py · pricing_engine.py [TODO]
    ├── simulation/      # executor · context · movement · events · delay_map · route · simulation_clock · simulation_context · vehicle_position
    ├── utils/           # logger · geo_utils
    └── main.py          # master script — runs the full simulation          [TODO]
```

---

## Prerequisites

- Python 3.10 or higher
- Internet access for the **one-time** map download step only

Install dependencies:

```bash
# Data engineering, routing, simulation
pip install osmnx networkx scikit-learn statsmodels pandas numpy joblib

# MILP dispatcher
pip install pulp

# Vision inspection + reinforcement learning
pip install torch torchvision gymnasium
```

---

## How to Run

All commands are run from the **project root** (`AAI-501-Introduction-to-AI-and-ML/`).

### Step 1 — Download the Road Network *(once, needs internet)*

Downloads the Bengaluru road network from OpenStreetMap and saves it locally.  
Every subsequent step runs fully offline.

```bash
python -m src.ingestion.create_city_map
```

To use a different city:

```bash
python -m src.ingestion.create_city_map --city "Mumbai, Maharashtra, India"
```

**Output:** `data/raw/static_city_map.graphml`

---

### Step 2 — Snap Orders to Road Nodes *(offline)*

Reads `amazon_delivery.csv`, filters to the Bengaluru region, and maps each GPS coordinate to the nearest OSM road-network node.

```bash
python -m src.ingestion.snap_nodes
```

**Input:** `data/raw/amazon_delivery.csv` · `data/raw/static_city_map.graphml`  
**Output:** `data/interim/mapped_orders.pkl`

> Place `amazon_delivery.csv` in `data/raw/` before running this step.

---

### Step 3 — Train the Delay Regressor *(offline)*

Trains a Random Forest model to predict delivery delay (minutes) from distance, weather, vehicle type, and traffic level.

```bash
python -m src.analytics.delay_model
```

**Input:** `data/interim/mapped_orders.pkl`  
**Output:** `artifacts/delay_forest.pkl`

---

### Step 4 — Generate Demand Forecast *(offline)*

Fits a Holt-Winters model on order timestamps and produces a 24-hour ahead hourly volume forecast.

```bash
python -m src.analytics.demand_forecast
```

**Input:** `data/interim/mapped_orders.pkl`  
**Output:** `data/outputs/demand_forecast.json`

---

### Step 5 — Train the CNN Inspector *(TODO)*

Fine-tunes ResNet18 on the damaged-box dataset to classify packages as Intact or Damaged.

```bash
python -m src.ai.img_loader      # preprocess package images
python -m src.ai.pricing_env     # verify Gymnasium env skeleton
```

**Dataset:** [Damaged Box — Kaggle](https://www.kaggle.com/datasets/teomingzhe/damaged-box)  
**Output:** `artifacts/package_cnn.pt`

---

### Step 6 — Train the Q-Learning Agent *(TODO)*

Trains the tabular Q-table to set surge multipliers based on queue state and demand forecast.

```bash
python -m src.ai.pricing_engine
```

**Input:** `data/outputs/demand_forecast.json` · reward formula `Reward = Revenue − Delay`  
**Output:** `artifacts/q_table.npy`

---

### Step 7 — Run the Full Simulation *(TODO)*

Once all artifacts are in place, the master script wires all components:

```bash
python src/main.py
```

Or wire manually:

```python
from src.graph.graph import Graph
from src.analytics.delay_model import DelayPredictor
from src.routing.dynamic_astar import DynamicAStar
from src.simulation.simulation_executor import SimulationExecutor
from src.simulation.simulation_context import SimulationContext
from src.optimization.cvrptw_dispatcher_milp import CVRPTWDispatcherMilp
from src.optimization.astart_routing import AStarRouting
from src.routing.heuristic import TravelTimeHeuristic
from src.simulation.delay_map import DelayMap
from src.optimization.dispatcher_config import DispatcherConfig

graph = Graph()
graph.load()

predictor = DelayPredictor()          # delay regression model
router    = DynamicAStar(graph, predictor)

heuristic = TravelTimeHeuristic(60.0)
astar_routing = AStarRouting(graph, heuristic, DelayMap())
config = DispatcherConfig()
dispatcher = CVRPTWDispatcherMilp(astar_routing, config)

executor = SimulationExecutor(
    sampler        = ...,             # samples orders from dataset each tick
    inspector      = ...,             # ResNet18 CNN — Intact / Damaged classification
    predictor      = predictor,       # RandomForest — predicts edge delay penalties
    dispatcher     = dispatcher,      # PuLP CVRPTW — multi-vehicle route assignment
    rl_environment = ...,             # Gymnasium env — queue-state pricing environment
    rl_agent       = ...,             # Q-table agent — surge multiplier decisions
    movement_engine= ...,             # advances vehicles one edge per tick
)

context = SimulationContext(graph=graph, current_time=...)

for tick in range(1000):
    executor.execute_tick(context)
```

---

## Team Responsibilities

| Role | Module | Deliverables | Status |
|---|---|---|---|
| Infrastructure & Optimization | `src/graph/` `src/simulation/` `src/optimization/` | `static_city_map.graphml` · `astar_routing.py` · `cvrptw_dispatcher.py` · `main.py` | Partial |
| Data Engineering & ML | `src/ingestion/` `src/analytics/` `src/routing/` | `mapped_orders.pkl` · `delay_forest.pkl` · `demand_forecast.json` · `dynamic_astar.py` | **Complete** |
| Vision & RL | `src/ai/` | `package_cnn.pt` · `q_table.npy` · `pricing_env.py` · `pricing_engine.py` | TODO |

The `SimulationContext` object is the shared handoff between all three engineers each tick.

---

## Remaining Work

### Infrastructure & Optimization
- [ ] `src/main.py` — master script wiring all components into a single simulation run
- [ ] Reward formula `Reward = Revenue − Delay` shared with the pricing team

### Vision & RL
- [ ] `src/ai/img_loader.py` — PyTorch DataLoader for damaged-box package images
- [ ] `src/ai/pricing_env.py` — `gymnasium.Env` subclass; 5-action discrete space (+0 % to +50 %); 5×5 state matrix (queue length × demand bucket)
- [ ] Fine-tune ResNet18 (freeze backbone, replace final layer with binary head) → `artifacts/package_cnn.pt`
- [ ] Train tabular Q-table using `Reward = Revenue − Delay`; inject `demand_forecast.json` into state → `artifacts/q_table.npy`
- [ ] `src/ai/pricing_engine.py` — operational wrapper connecting Q-table to live simulation context

---

## Key Configuration

All paths and constants are in [`src/config.py`](src/config.py).  
No hardcoded paths anywhere else in the codebase.

| Constant | Default | Purpose |
|---|---|---|
| `DELAY_ALPHA` | `50.0` | Minutes of delay → metres of extra path cost |
| `DEFAULT_SPEED_KMH` | `40` | Fallback vehicle speed |
| `MAX_SIMULATION_STEPS` | `1000` | Tick limit per run |

