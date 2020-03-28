# -*- coding: utf-8 -*-

import numpy as np
from const import (INTEGER_VALUES, INTEGER_MAX, INTEGER_MIN, OPERATOR_VALUES, OPERATOR_MAX, OPERATOR_MIN,
                   INTERPRET_VALUES, INTERPRET_MAX, INTERPRET_MIN, ANALYTICAL_VALUES, ANALYTICAL_MAX, ANALYTICAL_MIN)


class MAttribute(object):
    """
        This is the superclass for all mathematical attributes for algebra component in the problem.
        Each mathematical attribute will have a predefined value set and a specified index,
        which is sampled by random, within a range from min_level to max_level.
     """
    def __init__(self, name):
        self.name = name
        self.level = "MAttribute"

    def sample(self):
        pass

    def get_value(self):
        pass

    def set_value(self):
        pass

    def __repr__(self):
        return self.level + "." + self.name

    def __str__(self):
        return self.level + "." + self.name


class Integer(MAttribute):

    def __init__(self, n_sample=3, min_level=INTEGER_MIN, max_level=INTEGER_MAX):
        super(Integer, self).__init__("Integer")
        self.value_level = []
        self.values = INTEGER_VALUES
        self.min_level = min_level
        self.max_level = max_level
        self.n_sample = n_sample

    def sample(self, n_sample=3, min_level=INTEGER_MIN, max_level=INTEGER_MAX):
        self.value_level = []
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        for i in range(n_sample):
            n = np.random.choice(range(min_level, max_level + 1))
            self.value_level.append(n)

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        self.value_level = value_level

    def get_value(self, value_level=None):
        integer_value = []
        if value_level is None:
            value_level = self.value_level
        for level in value_level:
            value = self.values[level]
            integer_value.append(value)
        return integer_value

    def set_value(self, value_list):
        self.value_level = []
        for value in value_list:
            level = value - self.values[0]
            self.value_level.append(level)


class Operator(MAttribute):

    def __init__(self, n_sample=1, min_level=OPERATOR_MIN, max_level=OPERATOR_MAX):
        super(Operator, self).__init__("Operator")
        self.value_level = []
        self.values = OPERATOR_VALUES
        self.min_level = min_level
        self.max_level = max_level
        self.n_sample = n_sample

    def sample(self, n_sample=1, min_level=OPERATOR_MIN, max_level=OPERATOR_MAX):
        self.value_level = []
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        for i in range(n_sample):
            n = np.random.choice(range(min_level, max_level + 1))
            self.value_level.append(n)

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        self.value_level = value_level

    def get_value(self, value_level=None):
        operator_value = []
        if value_level is None:
            value_level = self.value_level
        for level in value_level:
            value = self.values[level]
            operator_value.append(value)
        return operator_value

    def set_value(self, value_list):
        self.value_level = []
        for value in value_list:
            if value == "+":
                self.value_level.append(0)
            elif value == "-":
                self.value_level.append(1)
            elif value == "*":
                self.value_level.append(2)
            elif value == "/":
                self.value_level.append(3)


class Interpret(MAttribute):

    def __init__(self, min_level=INTERPRET_MIN, max_level=INTERPRET_MAX):
        super(Interpret, self).__init__("Interpret")
        self.value_level = 0
        self.values = INTERPRET_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=INTERPRET_MIN, max_level=INTERPRET_MAX):
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        self.value_level = np.random.choice(range(min_level, max_level + 1))

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        self.value_level = value_level

    def get_value(self, value_level=None):
        if value_level is None:
            value_level = self.value_level
        return self.values[value_level]

    def set_value(self, value):
        if value == "holistic":
            self.value_level = 0
        elif value == "analytical":
            self.value_level = 1


class Analytical(MAttribute):

    def __init__(self, min_level=ANALYTICAL_MIN, max_level=ANALYTICAL_MAX):
        super(Analytical, self).__init__("Analytical")
        self.value_level = 0
        self.values = ANALYTICAL_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=ANALYTICAL_MIN, max_level=ANALYTICAL_MAX):
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        self.value_level = np.random.choice(range(min_level, max_level + 1))

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        self.value_level = value_level

    def get_value(self, value_level=None):
        if value_level is None:
            value_level = self.value_level
        return self.values[value_level]

    def set_value(self, value):
        if value == 2:
            self.value_level = 0
        elif value == 3:
            self.value_level = 1
        elif value == 4:
            self.value_level = 2