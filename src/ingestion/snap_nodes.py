"""
snap_nodes.py
Feature 1: Geospatial Ingestion & Node Snapping

Converts raw latitude/longitude pairs from the Amazon Last-Mile Delivery dataset
into offline OSM Node IDs using the pre-built static_city_map.graphml road network
produced by the create_city_map.py utility.

Deliverable: data/interim/mapped_orders.pkl
  Columns: Order_ID, Store_Node, Drop_Node, Distance_KM,
           Weather_Conditions, Vehicle_Type, Traffic_Level,
           Delay_Minutes, Order_Timestamp

Usage (from project root):
    python -m src.ingestion.snap_nodes

Import:
    from src.ingestion.snap_nodes import snap_nodes
    df = snap_nodes()
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox

warnings.filterwarnings("ignore")

from src.config import (
    MAP_DIRECTORY,
    MAP_FILENAME,
    DATA_RAW_DIRECTORY,
    DATA_INTERIM_DIRECTORY,
    AMAZON_DELIVERY_FILENAME,
    MAPPED_ORDERS_FILENAME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)




def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine formula returning distances in kilometres."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def snap_nodes(
    csv_path: Path | None = None,
    graphml_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load, clean, spatially snap, and export delivery orders to OSM node IDs.

    Parameters
    ----------
    csv_path : Path, optional
        Path to amazon_delivery.csv.
        Defaults to ``data/raw/amazon_delivery.csv``.
    graphml_path : Path, optional
        Path to static_city_map.graphml (output from create_city_map.py).
        Defaults to ``maps/static_city_map.graphml``.
    output_path : Path, optional
        Destination pickle path.
        Defaults to ``data/interim/mapped_orders.pkl``.

    Returns
    -------
    pd.DataFrame
        Node-snapped orders ready for Feature 2 and Feature 3.

    Raises
    ------
    FileNotFoundError
        If the CSV dataset or the GraphML map is missing.
    """
    csv_path = csv_path or (DATA_RAW_DIRECTORY / AMAZON_DELIVERY_FILENAME)
    graphml_path = graphml_path or (MAP_DIRECTORY / MAP_FILENAME)
    output_path = output_path or (DATA_INTERIM_DIRECTORY / MAPPED_ORDERS_FILENAME)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Amazon delivery dataset not found: {csv_path}\n"
            f"Place amazon_delivery.csv in data/raw/ before running this module."
        )
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"City map not found: {graphml_path}\n"
            f"Run 'python maps/create_city_map.py' to generate static_city_map.graphml."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load & clean raw dataset
    # ------------------------------------------------------------------ #
    logger.info(f"Loading dataset: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    logger.info(f"Raw dataset: {len(raw_df):,} orders, {raw_df.shape[1]} columns")

    for col in ("Weather", "Traffic", "Vehicle", "Area"):
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype(str).str.strip()

    raw_df["Traffic"] = raw_df["Traffic"].replace({"Jam": "High", "NaN": "Medium"})
    raw_df["Weather_Conditions"] = raw_df["Weather"]
    raw_df["Vehicle_Type"] = raw_df["Vehicle"]
    raw_df["Traffic_Level"] = raw_df["Traffic"]
    raw_df["Delay_Minutes"] = raw_df["Delivery_Time"]

    # ------------------------------------------------------------------ #
    # 2. Drop rows with missing coordinates (use all remaining rows)
    # ------------------------------------------------------------------ #
    coord_cols = ["Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude"]
    before = len(raw_df)
    regional_df = raw_df.dropna(subset=coord_cols).reset_index(drop=True)
    dropped = before - len(regional_df)
    if dropped:
        logger.warning(f"Dropped {dropped:,} rows with missing coordinates.")
    logger.info(f"Orders after coordinate check: {len(regional_df):,} (all rows used)")

    # ------------------------------------------------------------------ #
    # 3. Load the city map (100% offline)
    # ------------------------------------------------------------------ #
    logger.info(f"Loading city map: {graphml_path}")
    G = ox.load_graphml(graphml_path)
    logger.info(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ------------------------------------------------------------------ #
    # 4. Vectorised spatial snapping (osmnx ball-tree nearest-neighbour)
    # ------------------------------------------------------------------ #
    logger.info(f"Snapping {len(regional_df):,} orders to OSM nodes ...")
    t0 = time.time()

    store_nodes = ox.nearest_nodes(
        G,
        X=regional_df["Store_Longitude"].values,
        Y=regional_df["Store_Latitude"].values,
    )
    drop_nodes = ox.nearest_nodes(
        G,
        X=regional_df["Drop_Longitude"].values,
        Y=regional_df["Drop_Latitude"].values,
    )

    elapsed = time.time() - t0
    nfr_pass = elapsed < 10
    logger.info(
        f"Snapping completed in {elapsed:.2f}s "
        f"(target < 10 s for 5 000 orders: {'PASS' if nfr_pass else 'FAIL'})"
    )

    regional_df["Store_Node"] = store_nodes
    regional_df["Drop_Node"] = drop_nodes

    # ------------------------------------------------------------------ #
    # 5. Haversine distance & node validation (TC-E2-01)
    # ------------------------------------------------------------------ #
    regional_df["Distance_KM"] = _haversine_km(
        regional_df["Store_Latitude"].values,
        regional_df["Store_Longitude"].values,
        regional_df["Drop_Latitude"].values,
        regional_df["Drop_Longitude"].values,
    )

    graph_nodes = set(G.nodes())
    valid_mask = (
        regional_df["Store_Node"].isin(graph_nodes)
        & regional_df["Drop_Node"].isin(graph_nodes)
    )
    n_filtered = (~valid_mask).sum()
    if n_filtered > 0:
        logger.warning(f"Filtered {n_filtered} orders with nodes outside the graph.")
    regional_df = regional_df[valid_mask].copy()

    # ------------------------------------------------------------------ #
    # 6. Timestamps
    # ------------------------------------------------------------------ #
    if "Order_Date" in regional_df.columns and "Order_Time" in regional_df.columns:
        regional_df["Order_Timestamp"] = pd.to_datetime(
            regional_df["Order_Date"].astype(str) + " " + regional_df["Order_Time"].astype(str),
            errors="coerce",
        )
    else:
        regional_df["Order_Timestamp"] = pd.NaT

    # ------------------------------------------------------------------ #
    # 7. Export deliverable
    # ------------------------------------------------------------------ #
    output_cols = [
        "Order_ID",
        "Store_Node",
        "Drop_Node",
        "Distance_KM",
        "Weather_Conditions",
        "Vehicle_Type",
        "Traffic_Level",
        "Delay_Minutes",
        "Order_Timestamp",
    ]
    output_df = regional_df[output_cols].copy()
    output_df.to_pickle(output_path)

    # TC-E2-01 verification
    all_store_valid = all(n in graph_nodes for n in output_df["Store_Node"])
    all_drop_valid = all(n in graph_nodes for n in output_df["Drop_Node"])
    tc_pass = all_store_valid and all_drop_valid
    logger.info(
        f"Node validation - All Store_Node in graph: {all_store_valid} | "
        f"All Drop_Node in graph: {all_drop_valid} | "
        f"result: {'PASS' if tc_pass else 'FAIL'}"
    )
    logger.info(
        f"Deliverable saved: {output_path}  "
        f"({len(output_df):,} rows, {output_df['Store_Node'].nunique():,} unique store nodes)"
    )

    return output_df


def main() -> None:
    snap_nodes()


if __name__ == "__main__":
    main()
