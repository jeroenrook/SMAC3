from __future__ import annotations

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


from abc import ABC, abstractmethod
from typing import Any, Iterator

import time
import traceback

import numpy as np
from ConfigSpace import Configuration

from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.scenario import Scenario
from smac.utils.logging import get_logger
from smac.runner.target_function_script_runner import TargetFunctionScriptRunner

logger = get_logger(__name__)

class ACLibRunner(TargetFunctionScriptRunner):
    def __call__(self, algorithm_kwargs: dict[str, Any]) -> tuple[str, str]:
        # TODO fill correct kwargs
        # kwargs has "instance", "seed" and "budget" --> translate those
        return super().__call__(algorithm_kwargs)

