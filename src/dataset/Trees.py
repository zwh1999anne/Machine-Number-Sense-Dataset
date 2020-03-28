# -*- coding: utf-8 -*-

import copy
from AOT import (Root, Problem, Compartment, LCondition1, ACondition1, LCondition2, ACondition2)
from const import (Type_Constraint_1, Type_Constraint_2, Type_Constraint_3, NUM_CONSTRAINT)

'''
The first three functions combination_prob(), composition_prob(), partition_prob() can generate problems randomly.
The last function prune_problem_tree() can generate problem  with given parameters.
'''


def combination_prob():
    # Build And-Or Tree
    root = Root("Scene")
    # Problem Type: Combination
    problem = Problem("Combination")
    # Compartment 1: Layout
    geometry = Compartment("Layout")
    # Compartment 2: Algebra
    mathematics = Compartment("Algebra")
    # Layout Condition 1: Geometrical Type
    geom_object = LCondition1("geom_object", Type_Constraint_1)
    # Layout Condition 2: Geometrical Relation
    relation = LCondition2("relation")
    # Algebra Condition 1: Mathematical Object (including integers and operators)
    math_object = ACondition1("math_object", NUM_CONSTRAINT)
    # Algebra Condition 2: Interpretation Style
    interpretation = ACondition2("interpretation")

    geometry.insert(geom_object)
    geometry.insert(relation)
    mathematics.insert(math_object)
    mathematics.insert(interpretation)
    problem.insert(geometry)
    problem.insert(mathematics)
    root.insert(problem)

    return root


def composition_prob():

    root = Root("Scene")
    # Problem Type: Composition
    problem = Problem("Composition")
    geometry = Compartment("Layout")
    mathematics = Compartment("Algebra")
    geom_object = LCondition1("geom_object", Type_Constraint_2)
    # Layout Condition 2: Format of Arrangement
    format = LCondition2("format")
    math_object = ACondition1("math_object", NUM_CONSTRAINT)
    interpretation = ACondition2("interpretation")

    geometry.insert(geom_object)
    geometry.insert(format)
    mathematics.insert(math_object)
    mathematics.insert(interpretation)
    problem.insert(geometry)
    problem.insert(mathematics)
    root.insert(problem)

    return root


def partition_prob():

    root = Root("Scene")
    # Problem Type: Partition
    problem = Problem("Partition")
    geometry = Compartment("Layout")
    mathematics = Compartment("Algebra")
    geom_object = LCondition1("geom_object", Type_Constraint_3)
    # Geometrical Condition 2: Number of Parts
    part = LCondition2("part")
    math_object = ACondition1("math_object", NUM_CONSTRAINT)
    interpretation = ACondition2("interpretation")

    geometry.insert(geom_object)
    geometry.insert(part)
    mathematics.insert(math_object)
    mathematics.insert(interpretation)
    problem.insert(geometry)
    problem.insert(mathematics)
    root.insert(problem)

    return root


def prune_problem_tree(root, gcondition_1, gcondition_2, interpret, analytical_part):
    new_root = copy.deepcopy(root)
    problem_type, conditions = new_root.prepare()
    geom_condition_1 = conditions[0]
    geom_condition_2 = conditions[1]
    math_condition = conditions[3]

    # Set all the geometrical and mathematical attributes with given parameters
    cond_type = [("triangle", 0), ("square", 1), ("circle", 2), ("hexagon", 3), ("rectangle", 4)]
    for (x, y) in cond_type:
        if x == gcondition_1:
            geom_level_1 = y
    geom_condition_1.type.set_value_level(geom_level_1)

    if problem_type == "Combination":
        cond_relation = [("overlap", 0), ("include", 1), ("tangent", 2)]
        for (x, y) in cond_relation:
            if x == gcondition_2:
                geom_level_2 = y
        geom_condition_2.grelation.set_value_level(geom_level_2)

    elif problem_type == "Composition":
        cond_format = [("line", 0), ("cross", 1), ("triangle", 2), ("square", 3), ("circle", 4)]
        for (x, y) in cond_format:
            if x == gcondition_2:
                geom_level_2 = y
        geom_condition_2.format.set_value_level(geom_level_2)

    elif problem_type == "Partition":
        cond_part = [(2, 0), (4, 1), (6, 2), (8, 3)]
        for (x, y) in cond_part:
            if x == gcondition_2:
                geom_level_2 = y
        geom_condition_2.part.set_value_level(geom_level_2)

    math_condition.interpret.set_value(interpret)
    math_condition.analytical.set_value(analytical_part)

    return new_root
