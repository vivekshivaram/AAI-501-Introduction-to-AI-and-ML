# LogiSim-AI — Architecture & Design Reference

## 1. Repository Layout

```
AAI-501-Introduction-to-AI-and-ML/
│
├── data/                          # All file-based data (gitignore large files)
│   ├── raw/
│   │   ├── amazon_delivery.csv    # Source delivery dataset (~43 k orders, India)
│   │   └── static_city_map.graphml# OSM road network — generated once, reused offline
│   ├── interim/
│   │   └── mapped_orders.pkl      # GPS coords snapped to OSM node IDs
│   └── outputs/
│       ├── demand_forecast.json   # 24-element hourly volume forecast
│       └── *.png                  # Optional diagnostic plots
│
├── artifacts/                     # Trained ML model bundles (binary, gitignored)
│   └── delay_forest.pkl           # RandomForest + encoders + metadata
│
├── logs/                          # Runtime log files (auto-created)
│   └── simulation.log
│
└── src/                           # All source code
    ├── config.py                  # Single source of truth for all paths & constants
    │
    ├── ingestion/                 # Data preparation (run before simulation)
    │   ├── create_city_map.py     # Downloads OSM road network → data/raw/
    │   └── snap_nodes.py          # Snaps GPS coords to node IDs → data/interim/
    │
    ├── graph/                     # Road-network primitives
    │   ├── graph.py               # Graph class — loads .graphml, nearest-node, shortest-path
    │   ├── edge.py                # Edge dataclass (source, dest, length, travel_time, speed)
    │   └── node.py                # Node dataclass (id, latitude, longitude)
    │
    ├── models/                    # Business-domain entity dataclasses
    │   ├── order.py               # Order (pickup/delivery nodes, deadline, predicted_delay)
    │   ├── vehicle.py             # Vehicle (capacity, speed, route, load)
    │   └── states.py              # VehicleState & SimulationState enums
    │
    ├── analytics/                 # ML training & inference modules
    │   ├── delay_model.py         # Train RF regressor; DelayPredictor inference class
    │   └── demand_forecast.py     # Holt-Winters 24-h demand forecast
    │
    ├── routing/
    │   └── dynamic_astar.py       # DynamicAStar — A* with ML delay penalties on edges
    │
    ├── simulation/                # Tick-based simulation engine
    │   ├── simulation_executor.py # Orchestrates all 7 steps per tick
    │   ├── simulation_context.py  # Shared mutable state for one simulation run
    │   ├── movement_engine.py     # Moves vehicles along routes one edge per tick
    │   ├── delay_map.py           # Holds edge → penalty dict; updated each tick
    │   ├── events.py              # Event dataclasses (VehicleMoved, Delivered, …)
    │   ├── route.py               # Route dataclass (node list, distances, arrival times)
    │   ├── statistics.py          # Aggregate metrics (deliveries, distance, delay)
    │   ├── vehicle_position.py    # Tracks current node & route progress per vehicle
    │   ├── movement_result.py     # Result of a single vehicle movement step
    │   └── simulation_clock.py    # Simulation time management
    │
    └── utils/
        ├── geo_utils.py           # Haversine distance, travel-time estimation
        └── logger.py              # Unified file + console logger
```

---

## 2. Data Flow

```mermaid
flowchart TD
    subgraph INTERNET ["☁️  Internet — one-time only"]
        OSM[(OpenStreetMap API)]
    end

    subgraph INGESTION ["src/ingestion — run once"]
        CMY[create_city_map.py]
        SN[snap_nodes.py]
    end

    subgraph RAW ["data/raw"]
        CSV[amazon_delivery.csv]
        GML[static_city_map.graphml]
    end

    subgraph INTERIM ["data/interim"]
        PKL[mapped_orders.pkl\nOrder_ID · Store_Node · Drop_Node\nDistance_KM · Weather · Vehicle\nTraffic · Delay_Minutes · Timestamp]
    end

    subgraph ANALYTICS ["src/analytics — run once"]
        DM[delay_model.py\nRandomForestRegressor]
        DF[demand_forecast.py\nHolt-Winters]
    end

    subgraph ARTIFACTS ["artifacts/"]
        PKL2[delay_forest.pkl\nmodel · encoders · metadata]
    end

    subgraph OUTPUTS ["data/outputs"]
        JSON[demand_forecast.json\nhourly_forecast: float×24]
    end

    subgraph SIM ["Simulation Runtime"]
        CTX[SimulationContext]
        EXEC[SimulationExecutor]
        PRED[DelayPredictor]
        DMAP[DelayMap]
        ROUTER[DynamicAStar]
        ME[MovementEngine]
        G[Graph]
    end

    OSM -->|download| CMY
    CMY -->|save| GML
    CSV --> SN
    GML --> SN
    SN -->|pickle| PKL
    PKL --> DM
    PKL --> DF
    DM -->|joblib.dump| PKL2
    DF -->|json| JSON

    GML -->|ox.load_graphml| G
    PKL2 -->|joblib.load| PRED
    G --> CTX
    PRED --> CTX

    CTX --> EXEC
    EXEC -->|predict| PRED
    PRED -->|update| DMAP
    DMAP --> ROUTER
    ROUTER -->|find_path| EXEC
    EXEC -->|move| ME
    ME --> CTX
```

---

## 3. Per-Tick Simulation Loop

```mermaid
sequenceDiagram
    participant E as SimulationExecutor
    participant S as Sampler
    participant I as Inspector
    participant P as DelayPredictor
    participant D as Dispatcher
    participant RL as RL Agent
    participant M as MovementEngine
    participant CTX as SimulationContext

    loop Every Tick T
        E->>S: sample(context)
        S-->>CTX: append pending_orders

        E->>I: inspect(orders)
        I-->>CTX: set order.inspection_passed<br/>move failed → rejected_orders

        E->>P: predict(context)
        P->>CTX: read graph node coords
        P->>P: predict_delay(distance, weather, vehicle, traffic)
        P-->>CTX: write order.predicted_delay<br/>write delay_map.edge_penalties

        E->>D: dispatch(context)
        D->>D: astar_path(weight = Length + Delay×α)
        D-->>CTX: assign vehicles → dispatched_orders

        E->>RL: update(context) → state
        RL-->>CTX: surge_multiplier = Q-table action

        E->>M: move(context)
        M-->>CTX: advance vehicle positions<br/>fire DeliveryCompletedEvent
        Note over CTX: tick += 1
    end
```

---

## 4. Module Dependency Graph

```mermaid
graph LR
    CFG[config.py]

    subgraph graph ["src/graph"]
        GR[graph.py]
        ED[edge.py]
        ND[node.py]
    end

    subgraph models ["src/models"]
        OR[order.py]
        VH[vehicle.py]
        ST[states.py]
    end

    subgraph ingestion ["src/ingestion"]
        CM[create_city_map.py]
        SN2[snap_nodes.py]
    end

    subgraph analytics ["src/analytics"]
        DLY[delay_model.py]
        DMD[demand_forecast.py]
    end

    subgraph routing ["src/routing"]
        DA[dynamic_astar.py]
    end

    subgraph simulation ["src/simulation"]
        CTX2[simulation_context.py]
        EX[simulation_executor.py]
        MVE[movement_engine.py]
        DMP[delay_map.py]
    end

    subgraph utils ["src/utils"]
        LOG[logger.py]
        GEO[geo_utils.py]
    end

    CFG --> GR
    CFG --> SN2
    CFG --> CM
    CFG --> DLY
    CFG --> DMD
    CFG --> LOG

    GR --> ED
    GR --> ND

    CTX2 --> GR
    CTX2 --> OR
    CTX2 --> VH
    CTX2 --> DMP

    MVE --> GR
    MVE --> VH

    DA --> GR
    DA --> DLY

    EX --> CTX2
    EX --> DLY
    EX --> DA

    DLY --> GEO
    SN2 --> CFG
    SN2 --> LOG
    DLY --> LOG
    DMD --> LOG
    GR --> LOG
```

---

## 5. ML Model — Delay Regressor

```mermaid
flowchart LR
    subgraph Features
        F1[Distance_KM\ncontinuous]
        F2[Weather_Conditions\nLabelEncoded]
        F3[Vehicle_Type\nLabelEncoded]
        F4[Traffic_Level\nLabelEncoded]
    end

    subgraph Model ["RandomForestRegressor\n100 trees · depth 15"]
        RF((RF))
    end

    subgraph Bundle ["artifacts/delay_forest.pkl"]
        B1[model]
        B2[encoders dict]
        B3[feature_cols]
        B4[metadata\nr2 · mae · n_samples]
    end

    F1 & F2 & F3 & F4 --> RF
    RF --> B1
    B1 & B2 & B3 & B4 --> Bundle

    Bundle -->|joblib.load| DP[DelayPredictor\n.predict_delay\n.predict context]
    DP -->|float ≥ 0| OUT[Delay minutes\nper edge context]
```

---

## 6. Dynamic A* Edge Weight

```mermaid
flowchart LR
    L["Length_ij\n(metres)"]
    D["Predicted_Delay\n(minutes)"]
    A["α = 50.0\n(m per minute)"]
    W["Weight_ij\n= Length + Delay × α"]

    L --> W
    D --> MUL["×"]
    A --> MUL
    MUL --> W

    W -->|"nx.astar_path\n(weight=weight_fn)"| PATH[Optimal path\nunder current conditions]
```

> **Tuning α**: The dispatcher can pass a custom `alpha` to `DynamicAStar(graph, predictor, alpha=...)`.  
> `α = 0` → pure physical distance routing.  
> `α = 100` → delay avoidance dominates.

---


---

## 8. Run Order

```mermaid
flowchart TD
    S0["① python -m src.ingestion.create_city_map\n☁️ Needs internet — run once"]
    S1["② python -m src.ingestion.snap_nodes\n🔒 Fully offline after step ①"]
    S2["③ python -m src.analytics.delay_model\n🔒 Fully offline"]
    S3["④ python -m src.analytics.demand_forecast\n🔒 Fully offline"]
    S4["⑤ Simulation runtime\nSimulationExecutor.execute_tick loop\n🔒 Fully offline"]

    S0 -->|"data/raw/static_city_map.graphml"| S1
    S1 -->|"data/interim/mapped_orders.pkl"| S2
    S1 -->|"data/interim/mapped_orders.pkl"| S3
    S2 -->|"artifacts/delay_forest.pkl"| S4
    S3 -->|"data/outputs/demand_forecast.json"| S4
    S0 -->|"data/raw/static_city_map.graphml"| S4
```

---

## 9. Shared Data Contract (cross-engineer schema)

| Field | Type | Set by | Consumed by |
|---|---|---|---|
| `Store_Node` | `int` (OSM node ID) | `snap_nodes.py` | Dispatcher, DynamicAStar |
| `Drop_Node` | `int` (OSM node ID) | `snap_nodes.py` | Dispatcher, DynamicAStar |
| `order.predicted_delay` | `float` (minutes) | `DelayPredictor.predict()` | Dispatcher, Statistics |
| `delay_map.edge_penalties` | `dict[(int,int), float]` | `DelayPredictor.predict()` | DynamicAStar weight function |
| `demand_forecast.json` | `{"hourly_forecast": float[24]}` | `demand_forecast.py` | RL pricing environment state matrix |
| `order.inspection_passed` | `bool \| None` | CNN Inspector (`pricing_env.py`) | Dispatcher (filters rejections) |
| `context.surge_multiplier` | `float` | Q-table RL agent | Pricing layer |
| `static_city_map.graphml` | GraphML file | `create_city_map.py` | `Graph.load()`, `snap_nodes.py` |
