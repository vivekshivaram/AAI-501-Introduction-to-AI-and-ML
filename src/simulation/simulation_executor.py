from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.simulation_context import SimulationContext


class SimulationExecutor:
    def __init__(
        self,
        sampler,
        inspector,
        predictor,
        dispatcher,
        rl_environment,
        rl_agent,
        movement_engine,
    ):
        self.sampler = sampler
        self.inspector = inspector
        # predictor implements predict(context) — fulfilled by DelayPredictor
        # from src.analytics.delay_model, which populates context.delay_map
        # and sets order.predicted_delay before dispatching.
        self.predictor = predictor
        self.dispatcher = dispatcher
        self.environment = rl_environment
        self.agent = rl_agent
        self.movement_engine = movement_engine

    def execute_tick(
        self,
        context: SimulationContext,
    ):
        # 1 - Sample new orders
        orders = self.sampler.sample(context)

        # 2 - Inspect packages (reject damaged ones)
        inspected = self.inspector.inspect(orders)

        # 3 - Predict delays for routing
        self.predictor.predict(context)

        # 4 - Dispatch orders to vehicles
        self.dispatcher.dispatch(context)

        # 5 - Update RL environment state
        state = self.environment.update(context)

        # 6 - RL agent selects pricing action
        action = self.agent.act(state)
        context.surge_multiplier = action
        context.statistics.record_surge(action)

        # 7 - Move vehicles along their routes
        movement_results = self.movement_engine.move(context)
        
        # 8 - Handle completed deliveries
        self._handle_completed_deliveries(context, movement_results)

        context.tick += 1
    
    def _handle_completed_deliveries(self, context: SimulationContext, movement_results: dict):
        """
        Check for vehicles that completed their routes and handle delivery completion.
        
        Args:
            context: Simulation context
            movement_results: Dict of vehicle_id -> MovementResult
        """
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        
        for vehicle in context.vehicles:
            result = movement_results.get(vehicle.vehicle_id)
            if not result or not result.reached_destination:
                continue
            
            # Vehicle reached destination - complete delivery
            if vehicle.assigned_orders:
                for order_id in vehicle.assigned_orders:
                    # Find the order
                    order = None
                    for o in context.dispatched_orders:
                        if o.order_id == order_id:
                            order = o
                            break
                    
                    if order:
                        # Mark order as delivered
                        order.delivered = True
                        order.delivery_tick = context.tick
                        
                        # Calculate delivery time in minutes
                        if order.dispatch_tick is not None:
                            # Each tick = SIMULATION_STEP_SECONDS (60s = 1 min)
                            from src.config import SIMULATION_STEP_SECONDS
                            ticks_elapsed = context.tick - order.dispatch_tick
                            order.delivery_time_minutes = (ticks_elapsed * SIMULATION_STEP_SECONDS) / 60.0
                        
                        # Record delivery statistics
                        delivery_distance = vehicle.current_route.total_distance if vehicle.current_route else 0.0
                        context.statistics.record_delivery(
                            delivery_time_minutes=order.delivery_time_minutes or 0.0,
                            distance=delivery_distance
                        )
                        
                        # Move from dispatched to delivered
                        context.dispatched_orders.remove(order)
                        context.delivered_orders.append(order)
                        
                        logger.info(f"Order {order_id} delivered by {vehicle.vehicle_id} at tick {context.tick}")
            
            # Reset vehicle to available state
            vehicle.available = True
            vehicle.assigned_orders.clear()
            vehicle.current_route = None
            vehicle.current_load = 0.0
            
            # Reset position
            position = context.positions.get(vehicle.vehicle_id)
            if position:
                position.route = None
                position.route_index = 0
            
            logger.info(f"Vehicle {vehicle.vehicle_id} completed delivery and is now available")
