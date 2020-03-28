# -*- coding: utf-8 -*-

import numpy as np
from const import (TYPE_VALUES, TYPES_MAX, TYPES_MIN, RELATION_VALUES, RELATION_MAX, RELATION_MIN,
                   FORMAT_VALUES, FORMAT_MAX, FORMAT_MIN, PART_VALUES, PART_MAX, PART_MIN)


class GAttribute(object):
    """
    This is the superclass for all geometrical attributes for layout component in the problem.
    Each geometrical attribute will have a predefined value set and a specified index,
    which is sampled by random, within a range from min_level to max_level.
    """
    def __init__(self, name):
        self.name = name
        self.level = "GAttribute1"

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


class Type(GAttribute):

    def __init__(self, min_level=TYPES_MIN, max_level=TYPES_MAX):
        super(Type, self).__init__("Type")
        self.value_level = 0
        self.values = TYPE_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=TYPES_MIN, max_level=TYPES_MAX):
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


class GRelation(GAttribute):

    def __init__(self, min_level=RELATION_MIN, max_level=RELATION_MAX):
        super(GRelation, self).__init__("GRelation")
        self.value_level = 0
        self.values = RELATION_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=RELATION_MIN, max_level=RELATION_MAX):
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


class Format(GAttribute):

    def __init__(self, min_level=FORMAT_MIN, max_level=FORMAT_MAX):
        super(Format, self).__init__("Format")
        self.value_level = 0
        self.values = FORMAT_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=FORMAT_MIN, max_level=FORMAT_MAX):
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


class Part(GAttribute):

    def __init__(self, min_level=PART_MIN, max_level=PART_MAX):
        super(Part, self).__init__("Part")
        self.value_level = 0
        self.values = PART_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=PART_MIN, max_level=PART_MAX):
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

