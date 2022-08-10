from __future__ import annotations

import typing

import logging

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.constants import MAXINT

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


def get_types(
    config_space: ConfigurationSpace,
    instance_features: typing.Optional[np.ndarray] = None,
) -> typing.Tuple[typing.List[int], typing.List[typing.Tuple[float, float]]]:
    """Return the types of the hyperparameters and the bounds of the
    hyperparameters and instance features.
    """
    # Extract types vector for rf from config space and the bounds
    types = [0] * len(config_space.get_hyperparameters())
    bounds = [(np.nan, np.nan)] * len(types)

    for i, param in enumerate(config_space.get_hyperparameters()):
        parents = config_space.get_parents_of(param.name)
        if len(parents) == 0:
            can_be_inactive = False
        else:
            can_be_inactive = True

        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            if can_be_inactive:
                n_cats = len(param.choices) + 1
            types[i] = n_cats
            bounds[i] = (int(n_cats), np.nan)
        elif isinstance(param, (OrdinalHyperparameter)):
            n_cats = len(param.sequence)
            types[i] = 0
            if can_be_inactive:
                bounds[i] = (0, int(n_cats))
            else:
                bounds[i] = (0, int(n_cats) - 1)
        elif isinstance(param, Constant):
            # for constants we simply set types to 0 which makes it a numerical
            # parameter
            if can_be_inactive:
                bounds[i] = (2, np.nan)
                types[i] = 2
            else:
                bounds[i] = (0, np.nan)
                types[i] = 0
            # and we leave the bounds to be 0 for now
        elif isinstance(param, UniformFloatHyperparameter):
            # Are sampled on the unit hypercube thus the bounds
            # are always 0.0, 1.0
            if can_be_inactive:
                bounds[i] = (-1.0, 1.0)
            else:
                bounds[i] = (0, 1.0)
        elif isinstance(param, UniformIntegerHyperparameter):
            if can_be_inactive:
                bounds[i] = (-1.0, 1.0)
            else:
                bounds[i] = (0, 1.0)
        elif isinstance(param, NormalFloatHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param._lower, param._upper)
        elif isinstance(param, NormalIntegerHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param.nfhp._lower, param.nfhp._upper)
        elif isinstance(param, BetaFloatHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param._lower, param._upper)
        elif isinstance(param, BetaIntegerHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param.bfhp._lower, param.bfhp._upper)
        elif not isinstance(
            param,
            (
                UniformFloatHyperparameter,
                UniformIntegerHyperparameter,
                OrdinalHyperparameter,
                CategoricalHyperparameter,
                NormalFloatHyperparameter,
                NormalIntegerHyperparameter,
                BetaFloatHyperparameter,
                BetaIntegerHyperparameter,
            ),
        ):
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    if instance_features is not None:
        types = types + [0] * instance_features.shape[1]

    return types, bounds


def check_subspace_points(
    X: np.ndarray,
    cont_dims: typing.Union[np.ndarray, typing.List] = [],
    cat_dims: typing.Union[np.ndarray, typing.List] = [],
    bounds_cont: typing.Optional[np.ndarray] = None,
    bounds_cat: typing.Optional[typing.List[typing.Tuple]] = None,
    expand_bound: bool = False,
) -> np.ndarray:
    """
    Check which points are place inside a given subspace
    Parameters
    ----------
    X: typing.Optional[np.ndarray(N,D)],
        points to be checked, where D = D_cont + D_cat
    cont_dims: typing.Union[np.ndarray(D_cont), typing.List]
        which dimensions represent continuous hyperparameters
    cat_dims: typing.Union[np.ndarray(D_cat), typing.List]
        which dimensions represent categorical hyperparameters
    bounds_cont: typing.optional[typing.List[typing.Tuple]]
        subspaces bounds of categorical hyperparameters, its length is the number of continuous hyperparameters
    bounds_cat: typing.Optional[typing.List[typing.Tuple]]
        subspaces bounds of continuous hyperparameters, its length is the number of categorical hyperparameters
    expand_bound: bool
        if the bound needs to be expanded to contain more points rather than the points inside the subregion
    Return
    ----------
    indices_in_ss:np.ndarray(N)
        indices of data that included in subspaces
    """
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    if len(cont_dims) == 0 and len(cat_dims) == 0:
        return np.ones(X.shape[0], dtype=bool)

    if len(cont_dims) > 0:
        if bounds_cont is None:
            raise ValueError("bounds_cont must be given if cont_dims provided")

        if len(bounds_cont.shape) != 2 or bounds_cont.shape[1] != 2 or bounds_cont.shape[0] != len(cont_dims):
            raise ValueError(
                f"bounds_cont (with shape  {bounds_cont.shape}) should be an array with shape of"
                f"({len(cont_dims)}, 2)"
            )

        data_in_ss = np.all(X[:, cont_dims] <= bounds_cont[:, 1], axis=1) & np.all(
            X[:, cont_dims] >= bounds_cont[:, 0], axis=1
        )

        if expand_bound:
            bound_left = bounds_cont[:, 0] - np.min(X[data_in_ss][:, cont_dims] - bounds_cont[:, 0], axis=0)
            bound_right = bounds_cont[:, 1] + np.min(bounds_cont[:, 1] - X[data_in_ss][:, cont_dims], axis=0)
            data_in_ss = np.all(X[:, cont_dims] <= bound_right, axis=1) & np.all(X[:, cont_dims] >= bound_left, axis=1)
    else:
        data_in_ss = np.ones(X.shape[0], dtype=bool)

    if len(cat_dims) == 0:
        return data_in_ss
    if bounds_cat is None:
        raise ValueError("bounds_cat must be given if cat_dims provided")

    if len(bounds_cat) != len(cat_dims):
        raise ValueError(
            f"bounds_cat ({len(bounds_cat)}) and cat_dims ({len(cat_dims)}) must have " f"the same number of elements"
        )

    for bound_cat, cat_dim in zip(bounds_cat, cat_dims):
        data_in_ss &= np.in1d(X[:, cat_dim], bound_cat)

    return data_in_ss
