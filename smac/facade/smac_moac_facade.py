import logging
from typing import Any

import numpy as np

from smac.epm.random_forest.rf_with_instances import RandomForestWithInstances
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.sobol_design import SobolDesign
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.optimizer.acquisition import LogEI, EHVI
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogScaledCost, RunHistory2EPM4Cost
from smac.multi_objective.aggregation_strategy import NoAggregationStrategy
from smac.intensification.sms_intensifier import SMSIntensifier
from smac.optimizer.acquisition.maximizer import MOLocalAndSortedRandomSearch

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SMAC4MOAC(SMAC4AC):
    """Facade to use SMAC for multi-objective algorithm configuration.

    see smac.facade.smac_Facade for API
    This facade overwrites options available via the SMAC facade


    See Also
    --------
    :class:`~smac.facade.smac_ac_facade.SMAC4AC` for documentation of parameters.

    Attributes
    ----------
    logger
    stats : Stats
    solver : SMBO
    runhistory : RunHistory
        List with information about previous runs
    trajectory : list
        List of all incumbents
    """

    def __init__(self, **kwargs: Any):
        scenario = kwargs["scenario"]

        kwargs["initial_design"] = kwargs.get("initial_design", LHDesign)  # CHANGED
        if len(scenario.cs.get_hyperparameters()) > 21201 and kwargs["initial_design"] is SobolDesign:
            raise ValueError(
                'The default initial design "Sobol sequence" can only handle up to 21201 dimensions. '
                'Please use a different initial design, such as "the Latin Hypercube design".',
            )

        init_kwargs = kwargs.get("initial_design_kwargs", dict())
        # init_kwargs["n_configs_x_params"] = init_kwargs.get("n_configs_x_params", 10)
        # init_kwargs["max_config_fracs"] = init_kwargs.get("max_config_fracs", 0.25)
        kwargs["initial_design_kwargs"] = init_kwargs

        # Intensification parameters - which intensifier to use and respective parameters
        if kwargs.get("intensifier") is not None:
            logging.log(logging.WARN, "Replacing intensifier argument with 'SMSIntensifier'")
        kwargs["intensifier"] = SMSIntensifier
        intensifier_kwargs = kwargs.get("intensifier_kwargs", dict())
        kwargs["intensifier_kwargs"] = intensifier_kwargs
        # intensifier_kwargs["min_chall"] = 1
        # scenario.intensification_percentage = 1e-10

        if kwargs.get("model") is None:
            model_class = RandomForestWithInstances
            kwargs["model"] = model_class

            # == static RF settings
            model_kwargs = kwargs.get("model_kwargs", dict())
            # model_kwargs["num_trees"] = model_kwargs.get("num_trees", 10)
            # model_kwargs["do_bootstrapping"] = model_kwargs.get("do_bootstrapping", True)
            # model_kwargs["ratio_features"] = model_kwargs.get("ratio_features", 1.0)
            # model_kwargs["min_samples_split"] = model_kwargs.get("min_samples_split", 2)
            # model_kwargs["min_samples_leaf"] = model_kwargs.get("min_samples_leaf", 1)
            # model_kwargs["log_y"] = model_kwargs.get("log_y", True)
            model_kwargs["max_marginalized_instance_features"] = model_kwargs.get("max_marginalized_instance_features", 100)
            kwargs["model_kwargs"] = model_kwargs

        # == Multi Objective Algorithm
        kwargs["multi_objective_algorithm"] = kwargs.get("multi_objective_algorithm", NoAggregationStrategy)

        # == Acquisition function
        kwargs["acquisition_function"] = kwargs.get("acquisition_function", EHVI)
        kwargs["runhistory2epm"] = kwargs.get("runhistory2epm", RunHistory2EPM4Cost)  # TODO changed. MO already normalizes.

        # assumes random chooser for random configs
        # random_config_chooser_kwargs = kwargs.get("random_configuration_chooser_kwargs", dict())
        # random_config_chooser_kwargs["prob"] = random_config_chooser_kwargs.get("prob", 0.2)
        # kwargs["random_configuration_chooser_kwargs"] = random_config_chooser_kwargs

        # better improve acquisition function optimization
        # 1. increase number of sls iterations
        acquisition_function_optimizer_kwargs = kwargs.get("acquisition_function_optimizer_kwargs", dict())
        acquisition_function_optimizer_kwargs["n_sls_iterations"] = 10
        kwargs["acquisition_function_optimizer_kwargs"] = acquisition_function_optimizer_kwargs

        kwargs["acquisition_function_optimizer"] = MOLocalAndSortedRandomSearch

        # Objective bounds
        if hasattr(scenario, "objective_bounds"):
            # TODO check if dimensions are as expected
            if not "runhistory_kwargs" in kwargs:
                kwargs["runhistory_kwargs"] = {}
            kwargs["runhistory_kwargs"]["objective_bounds"] = scenario.objective_bounds

        super().__init__(**kwargs)
        self.logger.info(self.__class__)

        # better improve acquisition function optimization
        # 2. more randomly sampled configurations
        self.solver.scenario.acq_opt_challengers = 5000  # type: ignore[attr-defined] # noqa F821

        # activate predict incumbent
        self.solver.epm_chooser.predict_x_best = True
