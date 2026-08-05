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
        # 1
        orders = self.sampler.sample(context)

        # 2
        inspected = self.inspector.inspect(orders)

        # 3
        self.predictor.predict(context)

        # 4
        self.dispatcher.dispatch(context)

        # 5
        state = self.environment.update(context)

        # 6
        action = self.agent.act(state)

        context.surge_multiplier = action

        # 7
        self.movement_engine.move(context)

        context.tick += 1