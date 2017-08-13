###############################################################################
# bisection algorithm                                                         #
# Copyright (C) 2017 FUNNYDMAN                                                #
#                                                                             #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#                                                                             #
###############################################################################
"""Метод нахождения корней основан на следствии из теоремы Бальзано - Коши:
Если функция непрерывна на некотором отрезке и
на концах этого отрезка принимает значения противоположных знаков,
то существует точка, в которой значение функции равно нулю.
Алгоритм:
1) Находим отрезок [a, b] на котором присутствует корень(esp - это точность,
с которой будем искать корень).
2) Находим среднюю точку отрезка [a, b] -> t = (a+b)/2
3) Проверяем условие, если f(t)=0, то корень найден.
4) Если f(a)*f(t)<0, т.е. f(a) и f(c) имеют разные знаки, тогда b=t, иначе a=t
Пример:
>>> bisection_method(1, 2, 0.001)
>>> 1.732421875
Аналитически решая уравнение x^2 - 3, находим корень равный sqrt(3);
"""
def f(x):
    return x**2 - 3

def bisection_method(a, b, eps):
    while (b - a) > 2 * eps:
        t = (a + b)/2
        if f(t) == 0:
            return t
        if f(a) * f(t) < 0:
            b = t
        elif f(a) * f(t) > 0:
            a = t
    return t
            

