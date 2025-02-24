# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#  This file (__init__.py) is part of neurone_loader                           -
#  (https://www.github.com/heilerich/neurone_loader)                           -
#  Copyright © 2019 Felix Heilmeyer.                                           -
#                                                                              -
#  This code is released under the MIT License                                 -
#  https://opensource.org/licenses/mit-license.php                             -
#  Please see the file LICENSE for details.                                    -
# ------------------------------------------------------------------------------
"""
Provides classes to load, represent and export data recorded with the Bittium NeurOne device.
Use :py:class:`Recording`, :py:class:`Session` or :py:class:`Phase` to represent the respective part of the data.
"""

from .loader import Recording, Session, Phase

__version__ = '2.0.1'
