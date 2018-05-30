# -*- coding: utf-8 -*-
"""
Copyright François Durand
fradurand@gmail.com
This file is part of My Toy Package.
    My Toy Package is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    My Toy Package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with My Toy Package.  If not, see <http://www.gnu.org/licenses/>.
"""
from sknetwork.hierarchy.MyClass2 import MyClass2


class MyClass1:
    """A whatever-you-are-doing.
    :param a: the `a` of the system. Must be nonnegative.
    :param b: the `b` of the system.
    :var str my_string: a nice string.
    :raise ValueError: if :attr:`a` is negative.
    Note: document the :meth:`__init__` method in the docstring of the class
    itself, because the docstring of the :meth:`__init__` method does not
    appear in the documentation.
    * Refer to a class this way: :class:`MyClass2`.
    * Refer to a method this way: :meth:`addition`.
    * Refer to a method in another class: :meth:`MyClass2.addition`.
    * Refer to an parameter or variable this way: :attr:`a`.
    >>> my_object = MyClass1(a=5, b=3)
    """

    #: This is a nice constant.
    A_NICE_CONSTANT = 42
    #:
    A_VERY_NICE_CONSTANT = 51

    def __init__(self, a: float, b: float):
        if a < 0:
            raise ValueError('Expected nonnegative a, got: ', a)
        self.a = a
        self.b = b
        self.my_string = 'a = %s and b = %s' % (a, b)           # type: str

    def __repr__(self) -> str:
        return '<MyClass1: a=%s, b=%s>' % (self.a, self.b)

    def __str__(self) -> str:
        return '(a, b) = %s, %s' % (self.a, self.b)

    def divide_a_by_c_and_add_d(self, c: float, d: float) -> float:
        """
        Divide :attr:`a` by something and add something else.
        :param c: a non-zero number. If you want to say many things about this
            parameter, you must indent the following lines, like this.
        :param d: a beautiful number.
        :return: :attr:`a` / :attr:`c` + :attr:`d`.
        :raise ZeroDivisionError: if :attr:`c` = 0.
        This function gives an example of Sphinx documentation with typical
        features.
        >>> my_object = MyClass1(a=5, b=3)
        >>> my_object.divide_a_by_c_and_add_d(c=2, d=10)
        12.5
        """
        return self.a / c + d

    def addition(self) -> float:
        """
        Add :attr:`a` and :attr:`b`.
        :return: :attr:`a` + :attr:`b`.
        >>> my_object = MyClass1(a=5, b=3)
        >>> my_object.addition()
        8
        """
        return MyClass2(self.a, self.b).addition()

    # noinspection PyProtectedMember
    def _secret_function(self) -> float:
        """
        Difference between :attr:`a` and :attr:`b`.
        :return: :attr:`a` - :attr:`b`.
        Since the name of this function starts with _, it does not appear in
        the Sphinx documentation.
        >>> my_object = MyClass1(a=5, b=3)
        >>> my_object._secret_function()
        2
        """
        return self.a - self.b


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    print('Do some little tests here')
    test = MyClass1(a=42, b=51)
    print(test.addition())
