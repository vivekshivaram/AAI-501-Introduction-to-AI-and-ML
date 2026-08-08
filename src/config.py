"""
Configuration file for LogiSim-AI simulation.

This module centralizes all configurable parameters for the simulation,
including file paths, simulation parameters, vehicle settings, and ML hyperparameters.

Configuration Sections:
- Directory layout: Data, artifacts, logs
- File names: Map, models, datasets
- Simulation parameters: Steps, fleet size, time increments
- Order parameters: Weight, volume, delivery windows
- Package inspection: Damage rejection rates
- Dynamic routing: Speed, delay penalties
- Pricing (RL): Surge multipliers, Q-learning hyperparameters
- Vehicle types: Capacities and speeds
- Random seed: For reproducibility
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Directory layout ---
DATA_DIRECTORY = ROOT_DIR / "data"
DATA_RAW_DIRECTORY = DATA_DIRECTORY / "raw"
DATA_INTERIM_DIRECTORY = DATA_DIRECTORY / "interim"
DATA_OUTPUTS_DIRECTORY = DATA_DIRECTORY / "outputs"
LOG_DIRECTORY = ROOT_DIR / "logs"
ARTIFACTS_DIRECTORY = ROOT_DIR / "artifacts"

# Road network map lives alongside other raw input files
MAP_DIRECTORY = DATA_RAW_DIRECTORY

# --- File names ---
MAP_FILENAME = "static_city_map.graphml"
DELAY_MODEL_FILENAME = "delay_forest.pkl"
DEMAND_FORECAST_FILENAME = "demand_forecast.json"
MAPPED_ORDERS_FILENAME = "mapped_orders.pkl"
AMAZON_DELIVERY_FILENAME = "amazon_delivery.csv"

# --- Simulation parameters ---
DEFAULT_SPEED_KMH = 80
SIMULATION_STEP_SECONDS = 60
SIMULATION_STEP_MINUTES = 1
MAX_SIMULATION_STEPS = 20
DEFAULT_VEHICLE_CAPACITY = 50
FLEET_SIZE = 20
ORDERS_PER_TICK = 5
LOG_LEVEL = "INFO"

# --- Order parameters ---
ORDER_MIN_WEIGHT_KG = 0.5
ORDER_MAX_WEIGHT_KG = 15.0
ORDER_MIN_VOLUME_M3 = 0.01
ORDER_MAX_VOLUME_M3 = 0.20
ORDER_DELIVERY_WINDOW_MINUTES = 90
ORDER_DEADLINE_HOURS = 4

# --- Package inspection ---
PACKAGE_DAMAGE_REJECTION_RATE = 0.05  # 5% of packages rejected as damaged

# --- Dynamic routing ---
# Penalty scale: 1 minute of predicted delay equals this many metres of extra path cost
DELAY_ALPHA = 50.0
ROUTING_AVERAGE_SPEED_KMH = 60.0

# --- Pricing (RL) ---
SURGE_MULTIPLIERS = [1.0, 1.125, 1.25, 1.375, 1.5]
DEFAULT_SURGE_MULTIPLIER = 1.0

# --- Q-Learning parameters ---
Q_LEARNING_ALPHA = 0.15
Q_LEARNING_EPSILON = 0.25
Q_LEARNING_EPSILON_DECAY = 0.99
Q_LEARNING_MIN_EPSILON = 0.05

# --- Random seed for reproducibility ---
RANDOM_SEED = 42

# --- Vehicle types ---
VEHICLE_TYPES = {
    "bicycle": {"capacity": 10.0, "speed_kmh": 15.0},
    "scooter": {"capacity": 20.0, "speed_kmh": 30.0},
    "motorcycle": {"capacity": 30.0, "speed_kmh": 45.0},
    "van": {"capacity": 200.0, "speed_kmh": 35.0},
}
