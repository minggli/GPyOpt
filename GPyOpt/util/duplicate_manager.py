# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from functools import partial

import numpy as np
from ..core.task.space import Design_space


class RoundedSet(set):
    """duplicated as of specified decimal places."""
    def __init__(self, *args, decimals=4):
        self._decimals = decimals
        self._round = partial(np.around, decimals=self._decimals)
        super(RoundedSet, self).__init__(*map(self._round, args))

    def add(self, a):
        super(RoundedSet, self).add(self._round(a))

    def update(self, iterable):
        super(RoundedSet, self).update(map(self._round, iterable))

    def __contains__(self, a):
        if super(RoundedSet, self).__contains__(self._round(a)):
            return True
        else:
            return False


class DuplicateManager(object):
    """
    Class to manage potential duplicates in the suggested candidates.

    :param space: object managing all the logic related the domain of the optimization
    :param zipped_X: matrix of evaluated configurations
    :param pending_zipped_X: matrix of configurations in the pending state
    :param ignored_zipped_X: matrix of configurations that the user desires to ignore (e.g., because they may have led to failures)
    """

    def __init__(self, space, zipped_X, pending_zipped_X=None, ignored_zipped_X=None, **kwargs):

        self.space = space

        self.unique_points = RoundedSet(decimals=kwargs.get("decimals", 4))
        self.unique_points.update(tuple(x.flatten()) for x in zipped_X)

        if np.any(pending_zipped_X):
            self.unique_points.update(tuple(x.flatten()) for x in pending_zipped_X)

        if np.any(ignored_zipped_X):
            self.unique_points.update(tuple(x.flatten()) for x in ignored_zipped_X)


    def is_zipped_x_duplicate(self, zipped_x):
        """
        param: zipped_x: configuration assumed to be zipped
        """
        return tuple(zipped_x.flatten()) in self.unique_points

    def is_unzipped_x_duplicate(self, unzipped_x):
        """
        param: unzipped_x: configuration assumed to be unzipped
        """
        return self.is_zipped_x_duplicate(self.space.zip_inputs(np.atleast_2d(unzipped_x)))
