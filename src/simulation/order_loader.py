from __future__ import annotations

from datetime import timedelta
import random
import pandas as pd

from src.models.order import Order
from src.config import (
    DATA_INTERIM_DIRECTORY,
    MAPPED_ORDERS_FILENAME,
    ORDER_MIN_WEIGHT_KG,
    ORDER_MAX_WEIGHT_KG,
    ORDER_MIN_VOLUME_M3,
    ORDER_MAX_VOLUME_M3,
    ORDER_DELIVERY_WINDOW_MINUTES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OrderLoader:
    def __init__(
        self,
        min_weight: float = ORDER_MIN_WEIGHT_KG,
        max_weight: float = ORDER_MAX_WEIGHT_KG,
        min_volume: float = ORDER_MIN_VOLUME_M3,
        max_volume: float = ORDER_MAX_VOLUME_M3,
        delivery_window_minutes: int = ORDER_DELIVERY_WINDOW_MINUTES,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.delivery_window = delivery_window_minutes

    def load(self, pickle_file: str = f"{DATA_INTERIM_DIRECTORY}/{MAPPED_ORDERS_FILENAME}") -> list[Order]:
        df = pd.read_pickle(pickle_file)
        orders: list[Order] = []
        filtered_count = 0
        
        for _, row in df.iterrows():
            pickup_node = int(row["Store_Node"])
            delivery_node = int(row["Drop_Node"])
            
            # Filter out orders where pickup and delivery nodes are the same
            if pickup_node == delivery_node:
                filtered_count += 1
                continue
            
            created = pd.to_datetime(row["Order_Timestamp"])
            order = Order(
                order_id=str(row["Order_ID"]),
                pickup_node=pickup_node,
                delivery_node=delivery_node,
                package_image="",
                created_time=created,
                deadline=created + timedelta(
                    minutes=self.delivery_window
                ),
                weight=random.uniform(
                    self.min_weight,
                    self.max_weight,
                ),
                volume=random.uniform(
                    self.min_volume,
                    self.max_volume,
                ),
                predicted_delay=float(
                    row["Delay_Minutes"]
                ),
                preferred_vehicle_type=row["Vehicle_Type"],
            )
            orders.append(order)

        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} orders with same pickup/delivery node")
        
        return orders