# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#  This file (util.py) is part of neurone_loader                               -
#  (https://www.github.com/heilerich/neurone_loader)                           -
#  Copyright © 2019 Felix Heilmeyer.                                           -
#                                                                              -
#  This code is released under the MIT License                                 -
#  https://opensource.org/licenses/mit-license.php                             -
#  Please see the file LICENSE for details.                                    -
# ------------------------------------------------------------------------------
"""
Provides internal utility functions
"""

import logging
import sys
from functools import wraps


def get_logger():
    """
    Get the default logger for this module and set default settings.

    :return: the logger
    :rtype: logging.Logger
    """
    lgr = logging.getLogger('neurone_loader')
    lgr.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt='%(asctime)s [%(name)s.%(module)s]:%(levelname)s: %(message)s', datefmt='%I:%M:%S')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.setFormatter(fmt)
    lgr.addHandler(stdout_handler)
    return lgr


logger = get_logger()


class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, method):
        self.method = method
        self.name = method.__name__

    def __get__(self, obj, cls):
        if obj:
            return self._get_with_inst(obj, cls)
        else:
            return self._get_no_inst(cls)

    def _get_with_inst(self, obj, cls):
        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.method, assigned=('__name__', '__module__'))
        def _func(*args, **kwargs):
            return self.method(obj, *args, **kwargs)

        return self._use_parent_doc(_func, overridden)

    def _get_no_inst(self, cls):
        overridden = next((getattr(parent, self.name, None) for parent in cls.__mro__[1:]), None)

        @wraps(self.method, assigned=('__name__', '__module__'))
        def _func(*args, **kwargs):
            return self.method(*args, **kwargs)

        return self._use_parent_doc(_func, overridden)

    def _use_parent_doc(self, func, source):
        if source is None:
            raise NameError('Can\'t find {name} in parents'.format(name=self.name))
        func.__doc__ = source.__doc__
        return func


doc_inherit = DocInherit
