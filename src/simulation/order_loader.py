from __future__ import annotations

from datetime import timedelta
import random
import pandas as pd

from src.models.order import Order
from src.config import (DATA_INTERIM_DIRECTORY, MAPPED_ORDERS_FILENAME)

class OrderLoader:
    def __init__(
        self,
        min_weight: float = 0.5,
        max_weight: float = 15.0,
        min_volume: float = 0.01,
        max_volume: float = 0.20,
        delivery_window_minutes: int = 90,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.delivery_window = delivery_window_minutes

    def load(self, pickle_file: str = f"{DATA_INTERIM_DIRECTORY}/{MAPPED_ORDERS_FILENAME}") -> list[Order]:
        df = pd.read_pickle(pickle_file)
        orders: list[Order] = []
        
        for _, row in df.iterrows():
            created = pd.to_datetime(row["Order_Timestamp"])
            order = Order(
                order_id=str(row["Order_ID"]),
                pickup_node=int(row["Store_Node"]),
                delivery_node=int(row["Drop_Node"]),
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

        return orders