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


class MyClass3:
    """A whatever-you-are-doing.
    :param a: the `a` of the system.
    :param b: the `b` of the system.
    >>> my_object = MyClass3(a = 5, b = 3)
    """

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def addition(self) -> float:
        """
        Add :attr:`a` and :attr:`b`
        :return: :attr:`a` + :attr:`b`.
        >>> my_object = MyClass3(a=5, b=3)
        >>> my_object.addition()
        8
        """
        return self.a + self.b


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    print('Do some little tests here')
    test = MyClass3(a=42, b=51)
    print(test.addition())
