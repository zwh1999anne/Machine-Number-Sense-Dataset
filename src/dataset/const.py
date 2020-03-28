# -*- coding: utf-8 -*-


PROBLEM_TYPE = ["combination_prob", "composition_prob", "partition_prob"]


# Parameters for all geometrical objects
# Type(This parameter has different value sets in different problems)
TYPE_VALUES = ["triangle", "square", "circle", "hexagon", "rectangle"]
TYPES_MAX = len(TYPE_VALUES)-1
TYPES_MIN = 0

# The compatible geometrical types in each type of problem
Type_Constraint_1 = [TYPES_MIN, TYPES_MAX]
Type_Constraint_2 = [TYPES_MIN, TYPES_MAX - 2]
Type_Constraint_3 = [TYPES_MIN + 1, TYPES_MAX - 1]

# The maximum amount of operators need to be sampled in each type of problem
NUM_CONSTRAINT = 25

# Parameters for layout in combination problem
# Relation between geometrical objects
RELATION_VALUES = ["overlap", "include", "tangent"]
RELATION_MAX = len(RELATION_VALUES)-1
RELATION_MIN = 0
# Length of edges
LENGTH_1 = 0.3

# Parameters for layout in composition problem
# Composition format
FORMAT_VALUES = ["line", "cross", "triangle", "square", "circle"]
FORMAT_MAX = len(FORMAT_VALUES)-1
FORMAT_MIN = 0
# Length of edges
LENGTH_2 = 0.05

# Parameters for layout in partition problem
# Number of parts
PART_VALUES = [2, 4, 6, 8]
PART_MAX = len(PART_VALUES)-1
PART_MIN = 0
# Length of edges
LENGTH_3 = 0.4


# Parameters for algebra component
# This is the integer constant in calculations like "x operator y operator z = constant"
INTEGER_VALUES = range(5, 95)
INTEGER_MAX = len(INTEGER_VALUES)-1
INTEGER_MIN = 0

# Operation between values
OPERATOR_VALUES = ["+", "-", "*", "/"]
OPERATOR_MAX = len(OPERATOR_VALUES)-1
OPERATOR_MIN = 0

# Mathematical interpretation style
INTERPRET_VALUES = ["holistic", "analytical"]
INTERPRET_MAX = len(INTERPRET_VALUES)-1
INTERPRET_MIN = 0

# Number of analytical parts
ANALYTICAL_VALUES = [2, 3, 4]
ANALYTICAL_MAX = len(INTERPRET_VALUES)-1
ANALYTICAL_MIN = 0

# Canvas parameters
PANEL_SIZE = 300
CENTER = (PANEL_SIZE/2, PANEL_SIZE/2)
# Centers for geometrical objects in combination problem
CENTER_1_1 = (int(PANEL_SIZE/2 - LENGTH_1 * PANEL_SIZE/2), PANEL_SIZE/2)
CENTER_1_2 = (int(PANEL_SIZE/2 + LENGTH_1 * PANEL_SIZE/2), PANEL_SIZE/2)
DEFAULT_WIDTH = 2




