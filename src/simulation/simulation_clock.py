from datetime import datetime, timedelta


class SimulationClock:

    def __init__(
        self,
        start_time: datetime,
        step_seconds: int,
    ):

        self._start_time = start_time
        self._current_time = start_time
        self._step = timedelta(seconds=step_seconds)
        self._tick = 0

    @property
    def current_time(self) -> datetime:
        return self._current_time

    @property
    def tick_count(self) -> int:
        return self._tick

    def tick(self) -> datetime:
        self._current_time += self._step
        self._tick += 1
        return self._current_time

    def reset(self):
        self._current_time = self._start_time
        self._tick = 0