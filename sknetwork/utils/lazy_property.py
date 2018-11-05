class cached_property:
    """
    Decorator used in replacement of @property to put the value in cache automatically.

    The first time the attribute is used, it is computed on-demand and put in cache. Later accesses to the
    attributes will use the cached value.

    Technically, this is a "descriptor".

    Adapted from https://stackoverflow.com/questions/4037481/caching-attributes-of-classes-in-python.

    Cf. :class:`DeleteCacheMixin` for an example.
    """

    def __init__(self, factory: callable):
        """
        This code runs when the decorator is applied to the function (i.e. when the function is defined).

        :meth:`factory` is the function.
        """
        self._factory = factory
        self._attr_name = factory.__name__

    def __get__(self, instance: object, owner: object) -> object:
        """
        This code runs only when the decorated function is directly called (which happens only when the value is not
        in cache).
        """
        # This hack is used so that the decorated function has the same docstring as the original function.
        if instance is None:
            def f():
                pass
            f.__doc__ = self._factory.__doc__
            f = property(f)
            return f
        # Compute the value.
        value = self._factory(instance)
        # Create the attribute and assign the value.
        # In the Python precedence order, the attribute "hides" the function of the same name.
        setattr(instance, self._attr_name, value)
        # Add the attribute name in the set cached_properties of the instance.
        try:
            instance.cached_properties.add(self._attr_name)
        except AttributeError:
            instance.cached_properties = {self._attr_name}
        # Return the value.
        return value


class DeleteCacheMixin:
    """
    Mixin used to delete cached properties.

    Cf. decorator :class:"cached_property".

    >>> class Example(DeleteCacheMixin):
    ...     @cached_property
    ...     def x(self):
    ...         print('Big computation...')
    ...         return 6 * 7
    >>> a = Example()
    >>> a.x
    Big computation...
    42
    >>> a.x
    42
    >>> a.delete_cache()
    >>> a.x
    Big computation...
    42
    """
    def delete_cache(self) -> None:
        if not hasattr(self, 'cached_properties'):
            return
        for p in self.cached_properties:
            del self.__dict__[p]
        # noinspection PyAttributeOutsideInit
        self.cached_properties = set()
