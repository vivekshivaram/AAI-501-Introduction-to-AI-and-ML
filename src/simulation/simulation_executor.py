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
        #TODO - Implementation pending

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