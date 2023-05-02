from __future__ import annotations

import copy
import itertools
from abc import abstractmethod
from typing import Any, Callable, Iterator
from scipy.stats import binom

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration

import smac
from smac.callback import Callback
from smac.constants import MAXINT
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import (
    InstanceSeedBudgetKey,
    InstanceSeedKey,
    TrajectoryItem,
    TrialValue,
)
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash, print_config_changes
from smac.utils.logging import get_logger
from smac.utils.pareto_front import calculate_pareto_front, sort_by_crowding_distance, _get_costs

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


def _dominates(a, b) -> bool:
    # Checks if a dominates b
    a = np.array(a)
    b = np.array(b)
    return np.count_nonzero(a <= b) >= len(a) and np.count_nonzero(a < b) >= 1

class NewCostDominatesOldCost():

    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        if not hasattr(self, "_old_config_cost"):
            self._old_config_cost = {}  # TODO remove configuration when done

        new_cost = self.runhistory.average_cost(config, config_isb_keys)
        if config not in self._old_config_cost:
            self._old_config_cost[config] = new_cost
            return True

        old_cost = self._old_config_cost[config]
        if _dominates(new_cost, old_cost):
            self._old_config_cost[config] = new_cost
            return True
        return False

class NewCostDominatesOldCostSkipFirst():

    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """ Do the first comparison with the incumbent when the configuration dominates the cost after finishing its first trial

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        if not hasattr(self, "_old_config_cost"):
            self._old_config_cost = {}  # TODO remove configuration when done

        new_cost = self.runhistory.average_cost(config, config_isb_keys)
        if config not in self._old_config_cost:
            self._old_config_cost[config] = new_cost
            return False

        old_cost = self._old_config_cost[config]
        if _dominates(new_cost, old_cost):
            self._old_config_cost[config] = new_cost
            return True
        return False

class DoublingNComparison():

    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)
        config_id = self.runhistory.get_config_id(config)
        config_hash = get_config_hash(config)

        # Do not compare very early in the process
        # if len(config_isb_keys) < 4:
        #     return False

        # Find N in _queue
        N = None
        for c, cn in self._queue:
            if config == c:
                N = cn
                break

        if N is None:
            logger.debug(
                f"This should not happen, but config {config_hash} is not in the queue.")
            return False

        return len(config_isb_keys) == N

class DoublingNComparisonFour():

    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)
        config_id = self.runhistory.get_config_id(config)
        config_hash = get_config_hash(config)

        # Do not compare very early in the process
        if len(config_isb_keys) < 4:
            return False

        # Find N in _queue
        N = None
        for c, cn in self._queue:
            if config == c:
                N = cn
                break

        if N is None:
            logger.debug(
                f"This should not happen, but config {config_hash} is not in the queue.")
            return False

        return len(config_isb_keys) == N
