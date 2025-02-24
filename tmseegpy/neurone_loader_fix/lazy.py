# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#  This file (lazy.py) is part of neurone_loader                               -
#  (https://www.github.com/heilerich/neurone_loader)                           -
#  Copyright © 2019 Felix Heilmeyer.                                           -
#                                                                              -
#  This code is released under the MIT License                                 -
#  https://opensource.org/licenses/mit-license.php                             -
#  Please see the file LICENSE for details.                                    -
# ------------------------------------------------------------------------------
"""
Provides the `Lazy` decorator to construct properties that are evaluated only once and the
`preloadable` decorator to enable optional preloading of all lazy properties on initialization.
"""

from functools import update_wrapper
from .util import logger


# noinspection PyMethodOverriding
class Lazy(property):
    """
    Return a lazy property attribute.

    This decorator can be used exactly like the :py:class:`property` function to turn a function into an attribute, the
    difference being the following: A function decorated with :py:class:`property` is evaluated every time the attribute
    is accessed. A function decorated with :py:class:`.lazy.Lazy` is only evaluated once and the result is stored as a
    private attribute. Subsequently the private attribute is returned when the property constructed with
    :py:class:`.lazy.Lazy` is accessed. The lazy property can also be set manually or deleted, just like every other
    attribute. When the lazy attribute is deleted and then accessed again, the property function is called again and
    the result stored as a private attribute.

    :Example:

    >>> class Test:
    >>>     @Lazy
    >>>     def lazy_attribute(self):
    >>>         print('lazy function called')
    >>>         return 'lazy return'
    >>>
    >>>     @property
    >>>     def property_attribute(self):
    >>>         print('property function called')
    >>>         return 'property return'
    >>>
    >>> test_object = Test()
    >>> print(test_object.property_attribute)
    property function called
    property return
    >>> print(test_object.property_attribute) # A property function is evaluated on every call
    property function called
    property return
    >>> print(test_object.lazy_attribute) # The lazy function is evaluated on first call
    lazy function called
    lazy return
    >>> print(test_object.lazy_attribute) # but not on subsequent calls
    lazy return
    >>> del test_object.lazy_attribute    # When deleted the attribute is reset and the
    >>> print(test_object.lazy_attribute) # function is evaluated again on next call
    lazy function called
    lazy return

    .. seealso:: Decorate your class with the :py:class:`..lazy.preloadable` attribute to enable optional preloading of
                 all lazy attributes on initialization.
    """
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.private_name = "_{}".format(fget.__name__)
        doc = doc or fget.__doc__
        property.__init__(self, fget=fget, fset=fset, fdel=fdel, doc=doc)

        # noinspection PyTypeChecker
        update_wrapper(self, fget)

        if type(self.__doc__) is str:
            self.__doc__ = """
            .. note:: This property is a lazy property. For details see :py:class:`.lazy.Lazy`
            
            {old_doc}
            """.format(old_doc=self.__doc__)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if hasattr(instance, self.private_name):
            result = getattr(instance, self.private_name)
        else:
            logger.debug('(Lazy) loading {}.{}'.format(owner.__name__, self.fget.__name__))
            # noinspection PyArgumentList
            result = self.fget(instance)
            setattr(instance, self.private_name, result)
        return result

    def __set__(self, instance, value):
        if self.fset is None:
            setattr(instance, self.private_name, value)
        else:
            # noinspection PyArgumentList
            self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            if hasattr(instance, self.private_name):
                delattr(instance, self.private_name)
        else:
            # noinspection PyArgumentList
            self.fdel(instance)


def preloadable(cls):
    """
    Use this as a decorator for a class that contains properties constructed with :py:class:`.lazy.Lazy`. A class
    decorated like this can be initialized with ``preload=True`` to call every lazy property once and store it's return
    value.
    Optionally the :py:class:`preload` function can be used to do the same. It can also be used to reload all lazy
    properties without deleting them first.

    :Example:

    >>> @preloadable
    >>> class Test:
    >>>     @Lazy
    >>>     def lazy_attribute(self):
    >>>         print('lazy function called')
    >>>         return 'lazy return'
    >>>
    >>> test_object = Test(preload=True) # The lazy property is evaluated on initialization
    lazy function called
    >>> print(test_object.lazy_attribute) # The stored attribute is returned
    lazy return
    >>> del test_object.lazy_attribute    # When deleted the attribute is reset and the
    >>> print(test_object.lazy_attribute) # function is evaluated again on next call
    lazy function called
    lazy return
    >>> test_object.preload() # All properties are reloaded even though already stored
    lazy function called
    """

    def _preload(self):
        """
        Use this function to call all properties constructed with :py:class:`.lazy.Lazy`. It can also be used to
        reload all lazy properties without deleting them first.

        :Example:

        >>> @preloadable
        >>> class Test:
        >>>     @Lazy
        >>>     def lazy_attribute(self):
        >>>         print('lazy function called')
        >>>         return 'lazy return'
        >>>
        >>> test_object = Test(preload=False) # The lazy property is not evaluated on initialization
        >>> test_object.preload()
        lazy function called
        >>> print(test_object.lazy_attribute) # The stored attribute is returned
        lazy return
        >>> test_object.preload() # All properties are reloaded even though already stored
        lazy function called
        """
        obj_type = type(self)

        def _try_preload_child(obj):
            if hasattr(obj, 'preload'):
                preload_function = getattr(obj, 'preload')
                if callable(preload_function):
                    obj.preload()

        for attr in [attr for attr in dir(self) if not attr.startswith('__')]:
            possible_prop = getattr(obj_type, attr, None)
            if isinstance(possible_prop, Lazy):
                if not hasattr(self, possible_prop.private_name):
                    logger.debug('Preloading property {} of {}'.format(attr, self))
                    getattr(self, attr)

            attr_obj = getattr(self, attr)
            try:
                _ = (e for e in attr_obj)  # Iterable
                for x in attr_obj:
                    _try_preload_child(x)
            except TypeError:
                # not iterable
                _try_preload_child(attr_obj)

    original_init = cls.__init__
    cls.preload = _preload

    def _new_init(self, *args, **kwargs):
        if 'preload' in kwargs:
            preload_enabled = kwargs['preload']
            del kwargs['preload']
        else:
            preload_enabled = False

        original_init(self, *args, **kwargs)

        if preload_enabled:
            self.preload()

    cls.__init__ = _new_init

    return cls
