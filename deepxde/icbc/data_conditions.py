"""Data conditions."""

__all__ = [
    "DataPoints"
]

import numbers
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from .. import backend as bkd
from .. import config
from .. import data
from .. import gradients as grad
from .. import utils
from ..backend import backend_name


class DataPoints:
    """Data condition class.

    Args:
        geom: A ``deepxde.geometry.Geometry`` instance.
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, points, values, link_fn=None, component=0, batch_size=None, shuffle=True):
        self.points = np.array(points, dtype=config.real(np))
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.component = component
        if isinstance(component, list) and backend_name != "pytorch":
            # TODO: Add support for multiple components in other backends
            raise RuntimeError(
                "multiple components only implemented for pytorch backend"
            )
        self.batch_size = batch_size
        # link_fn: a link function evaluating the values, link_fn(z, y)
        # z: pde prediction, y: data values
        if link_fn is None:
            # TODO:  better default choices? multi-backend support?
            def link_fn(z, y):
                return ((z - y)**2).mean()
        self.link_fn = link_fn

        if batch_size is not None:  # batch iterator and state
            if backend_name not in ["pytorch", "paddle"]:
                raise RuntimeError(
                    "batch_size only implemented for pytorch and paddle backend"
                )
            self.batch_sampler = data.sampler.BatchSampler(
                len(self), shuffle=shuffle)
            self.batch_indices = None

    def __len__(self):
        return self.points.shape[0]

    def collocation_points(self, X):
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        if self.batch_size is not None:
            if isinstance(self.component, numbers.Number):
                return (
                    self.link_fn(outputs[beg:end, self.component: self.component + 1],
                                 self.values[self.batch_indices])
                )
            return self.link_fn(outputs[beg:end, self.component], self.values[self.batch_indices])
        if isinstance(self.component, numbers.Number):
            return self.link_fn(outputs[beg:end, self.component: self.component + 1], self.values)
        return self.link_fn(outputs[beg:end, self.component], self.values)
