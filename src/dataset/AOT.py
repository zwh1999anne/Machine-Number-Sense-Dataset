# -*- coding: utf-8 -*-

import numpy as np
from Geometry import (Type, GRelation, Format, Part)
from Mathematics import (Integer, Operator, Interpret, Analytical)


class AOTNode(object):
    """
    This is the superclass for Nodes in And-Or Tree
    """

    levels_next = {"Root": "Problem",
                   "Problem": "Compartment",
                   "Compartment": "Condition"}

    def __init__(self, name, level, node_type, is_pg=False):
        self.name = name
        self.level = level
        self.node_type = node_type
        self.children = []
        self.is_pg = is_pg

    def insert(self, node):
        """
        For public use
        Arguments:
            node(AOTNode): a node to insert
        """
        assert isinstance(node, AOTNode)
        assert self.node_type != "leaf"
        assert node.level == self.levels_next[self.level]
        self.children.append(node)

    def _insert(self, node):
        """
        For private use
        Arguments:
            node(AOTNode): a node to insert
        """
        assert isinstance(node, AOTNode)
        assert self.node_type != "leaf"
        assert node.level == self.levels_next[self.level]
        self.children.append(node)

    def _resample(self):
        """
        For resampling the problem settings
        """
        assert self.is_pg
        if self.node_type == "and":
            for child in self.children:
                child._resample()
        else:
            self.children[0]._resample()

    def __repr__(self):
        return self.level + "." + self.name

    def __str__(self):
        return self.level + "." + self.name


# Node for the whole problem scene
class Root(AOTNode):

    def __init__(self, name, is_pg=False):
        super(Root, self).__init__(name, level="Root", node_type="or", is_pg=is_pg)

    def sample(self):
        """
        Returns:
            A newly instantiated And-Or Tree
        """
        if self.is_pg:
            raise ValueError("Could not sample on a PG")
        new_node = Root(self.name, is_pg=True)
        selected = np.random.choice(self.children)
        new_node.insert(selected._sample())
        return new_node

    def resample(self):
        self._resample()

    def prepare(self):
        """
        This function prepares the And-Or Tree for rendering
        Returns:
            problem.name(str): indicate the problem type
            Conditions(list of Object): used for rendering each layout and algebra condition
        """
        assert self.is_pg
        assert self.level == "Root"
        problem = self.children[0]
        compartments = []
        for child in problem.children:
            compartments.append(child)
        conditions = []
        for compartment in compartments:
            for child in compartment.children:
                conditions.append(child)
        return problem.name, conditions


# Nodes for three types of problems
class Problem(AOTNode):

    def __init__(self, name, is_pg=False):
        super(Problem, self).__init__(name, level="Problem", node_type="and", is_pg=is_pg)

    def _sample(self):
        if self.is_pg:
            raise ValueError("Could not sample on a PG")
        new_node = Problem(self.name, is_pg=True)
        for child in self.children:
            new_node.insert(child._sample())
        return new_node


# Nodes for layout and algebra components
class Compartment(AOTNode):

    def __init__(self, name, is_pg=False):
        super(Compartment, self).__init__(name, level="Compartment", node_type="and", is_pg=is_pg)

    def _sample(self):
        if self.is_pg:
            raise ValueError("Could not sample on a PG")
        new_node = Compartment(self.name, is_pg=True)
        for child in self.children:
            child.resample()
            new_node.insert(child)
        return new_node


# Node for the first condition of layout component -- types of geometrical objects
class LCondition1(AOTNode):

    def __init__(self, name, type_constraint, is_pg=False):
        super(LCondition1, self).__init__(name, level="Condition", node_type="leaf", is_pg=is_pg)
        self.type_constraint = type_constraint
        self.type = Type(min_level=self.type_constraint[0], max_level=self.type_constraint[1])
        self.type.sample()
        # self.value = self.type.get_value()

    def reset_constraint(self, min_level, max_level):
        self.type_constraint = [min_level, max_level]

    def resample(self):
        self.type.sample()


# Node for the second condition of layout component, different in different types of problems
class LCondition2(AOTNode):

    def __init__(self, name, is_pg=False):
        super(LCondition2, self).__init__(name, level="Condition", node_type="leaf", is_pg=is_pg)
        # sample the geometrical relation for combination problems
        self.grelation = GRelation()
        self.grelation.sample()
        # sample the format of arrangement for composition problem
        self.format = Format()
        self.format.sample()
        # sample the number of parts for partition problem
        self.part = Part()
        self.part.sample()
        # self.value = [self.grelation.get_value(), self.format.get_value(), self.part.get_value()]

    def resample(self):
        self.grelation.sample()
        self.format.sample()
        self.part.sample()


# Node for the first condition of algebra component ---- the integer constant and list of operators
class ACondition1(AOTNode):

    def __init__(self, name, number_constraint, is_pg=False):
        super(ACondition1, self).__init__(name, level="Condition", node_type="leaf", is_pg=is_pg)
        self.number_constraint = number_constraint
        self.integer = Integer()
        self.integer.sample()
        self.operator = Operator(n_sample=self.number_constraint)
        self.operator.sample(n_sample=self.number_constraint)
        # self.value = [self.integer.get_value(), self.operator.get_value()]

    def reset_constraint(self, new_sample):
        self.number_constraint = new_sample

    def resample(self):
        self.integer.sample()
        self.operator.sample(n_sample=self.number_constraint)


# Node for the second condition of algebra component ---- interpretation and number of groupings for analytical problem
class ACondition2(AOTNode):

    def __init__(self, name, is_pg=False):
        super(ACondition2, self).__init__(name, level="Condition", node_type="leaf", is_pg=is_pg)
        self.interpret = Interpret()
        self.interpret.sample()
        self.analytical = Analytical()
        self.analytical.sample()
        # self.value = [self.interpret.get_value(), self.analytical.get_value()]

    def resample(self):
        self.interpret.sample()
        self.analytical.sample()









