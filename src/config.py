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
DEFAULT_SPEED_KMH = 40
SIMULATION_STEP_SECONDS = 60
MAX_SIMULATION_STEPS = 100
DEFAULT_VEHICLE_CAPACITY = 50
LOG_LEVEL = "INFO"

# --- Dynamic routing ---
# Penalty scale: 1 minute of predicted delay equals this many metres of extra path cost
DELAY_ALPHA = 50.0
