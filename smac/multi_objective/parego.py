from __future__ import annotations

from typing import Optional

import numpy as np

from smac.multi_objective.aggregation_strategy import AggregationStrategy


class ParEGO(AggregationStrategy):
    def __init__(
        self,
        rng: Optional[np.random.RandomState] = None,
        rho: float = 0.05,
    ):
        super(ParEGO, self).__init__(rng=rng)
        self.rho = rho

        # Will be set on starting an SMBO iteration
        self._theta: np.ndarray | None = None

    def update_on_iteration_start(self, n_objectives: int) -> None:  # noqa: D102
        self._theta = self.rng.rand(n_objectives)

        # Normalize so that all theta values sum up to 1
        self._theta = self._theta / (np.sum(self._theta) + 1e-10)

    def __call__(self, values: list[float]) -> float:
        """
        Transform a multi-objective loss to a single loss.

        Parameters
        ----------
        values : list[float]
            Normalized values.

        Returns
        -------
        cost : float
            Combined cost.
        """
        if self._theta is None:
            raise ValueError("Iteration not yet initalized; Call `update_on_iteration_start()` first")

        theta_f = self._theta * values
        return np.max(theta_f, axis=0) + self.rho * np.sum(theta_f, axis=0)
