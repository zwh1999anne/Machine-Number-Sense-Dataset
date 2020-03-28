# -*- coding: utf-8 -*-

import cv2
import random
import numpy as np
from const import CENTER
from Calculator_Tree import (build_calculator_tree, number_sampler, fill_expression, middle_to_after)


# prune the algebra attributes and arrange the placement of integers in each problem
def math_parser(math_conditions, geom_conditions, prob_type, positions, show_center):
    num_pos = len(positions)
    pos_list = []
    gcondition_1, gcondition_2 = geom_conditions[0], geom_conditions[1]
    condition_1, condition_2 = math_conditions[0], math_conditions[1]
    constant = condition_1.integer
    operators = condition_1.operator
    constants = constant.get_value()
    const_value_0 = constants[0]
    operator_value = operators.get_value()
    interpret_style = condition_2.interpret.get_value()
    analytical_mode, analytical_part = 0, 0

    if num_pos < 4:
        interpret_style = "holistic"
        condition_2.interpret.set_value("holistic")
    '''
    As a trade-off of problem difficulty:
    if the constants are shown in the center of each panel, the constants in different panels are different
    if the constants are hidden, the constants in different panels keep consistent
    '''
    if show_center:
        const_value = constants
    else:
        const_value = [const_value_0, const_value_0, const_value_0]
        condition_1.integer.set_value(const_value)

    if interpret_style == "holistic":
        operator_list, int_list_1, int_list_2, int_list_3 = holistic_parser(num_pos, const_value, operator_value)
        condition_1.operator.set_value(operator_list)
        pos_list = positions
    else:
        analytical_part = condition_2.analytical.get_value()
        while (num_pos % analytical_part != 0) or (num_pos / analytical_part < 2):
            condition_2.analytical.sample()
            analytical_part = condition_2.analytical.get_value()
        operator_list, int_list_1, int_list_2, int_list_3 = analytical_parser(num_pos, const_value, operator_value, analytical_part)
        condition_1.operator.set_value(operator_list)
        '''
        Analytical mode represents the perceptual grouping principles (e.g. Gestalt laws), analytical mode == 1 refers 
        the law of proximity, analytical mode == 2 refers to the law of symmetry.
        analytical mode 3 or 4 is based on analytical mode 1 or 2, but has a little adjustments.
        In the some special cases, small revision of number placement was made to render the grouping of numbers 
        more close to human intuition.
        '''
        mode = np.random.choice([1, 2])
        if prob_type == "Combination":
            if gcondition_1.type.get_value() == "triangle" and gcondition_2.grelation.get_value() == "include":
                if analytical_part == 2:
                    mode = 1
                elif analytical_part == 3:
                    mode = 2
        elif prob_type == "Composition":
            if gcondition_2.format.get_value() in ["circle", "square", "triangle"]:
                mode = 2

        if mode == 1:
            if prob_type == "Partition":
                if gcondition_1.type.get_value() in ["hexagon", "circle"] and (gcondition_2.part.get_value() == 6 and analytical_part == 2):
                    analytical_mode = 3
                    pos_list_1 = [positions[0], positions[1], positions[2]]
                    pos_list_2 = [positions[5], positions[4], positions[3]]
                    pos_list = pos_list_1 + pos_list_2
                elif gcondition_1.type.get_value() in ["square", "circle"] and (gcondition_2.part.get_value() == 8 and analytical_part == 2):
                    analytical_mode = 3
                    pos_list_1 = [positions[0], positions[1], positions[2], positions[3]]
                    pos_list_2 = [positions[7], positions[6], positions[5], positions[4]]
                    pos_list = pos_list_1 + pos_list_2
                else:
                    analytical_mode = 1
                    pos_list = positions
            elif prob_type == "Combination":
                if gcondition_1.type.get_value() == "hexagon" and gcondition_2.grelation.get_value() == "overlap":
                    analytical_mode = 3
                    pos_list_1 = [positions[0], positions[1]]
                    pos_list_2 = [positions[3], positions[2]]
                    pos_list = pos_list_1 + pos_list_2
                else:
                    analytical_mode = 1
                    pos_list = positions
            elif prob_type == "Composition":
                analytical_mode = 1
                pos_list = positions

        elif mode == 2:
            if analytical_part == 2:
                pos_list_1, pos_list_2 = [], []
                for i in range(num_pos):
                    if i % 2 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 2 == 1:
                        pos_list_2.append(positions[i])
                if (prob_type == "Combination" and gcondition_2.grelation.get_value() == "include") or \
                        (prob_type == "Composition" and gcondition_2.format.get_value() == "cross"):
                    analytical_mode = 4
                    pos_list_3 = [pos_list_1[0], pos_list_1[2], pos_list_1[3], pos_list_1[1]]
                    pos_list_4 = [pos_list_2[0], pos_list_2[2], pos_list_2[3], pos_list_2[1]]
                    pos_list = pos_list_3 + pos_list_4
                else:
                    analytical_mode = 2
                    pos_list = pos_list_1 + pos_list_2
            elif analytical_part == 3:
                analytical_mode = 2
                pos_list_1, pos_list_2, pos_list_3 = [], [], []
                for i in range(num_pos):
                    if i % 3 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 3 == 1:
                        pos_list_2.append(positions[i])
                    elif i % 3 == 2:
                        pos_list_3.append(positions[i])
                pos_list = pos_list_1 + pos_list_2 + pos_list_3
            elif analytical_part == 4:
                analytical_mode = 2
                pos_list_1, pos_list_2, pos_list_3, pos_list_4 = [], [], [], []
                for i in range(num_pos):
                    if i % 4 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 4 == 1:
                        pos_list_2.append(positions[i])
                    elif i % 4 == 2:
                        pos_list_3.append(positions[i])
                    elif i % 4 == 3:
                        pos_list_4.append(positions[i])
                pos_list = pos_list_1 + pos_list_2 + pos_list_3 + pos_list_4

    if show_center:
        int_list_1.append(const_value[0])
        int_list_2.append(const_value[1])
        int_list_3.append(const_value[2])
        pos_list.append(CENTER)

    return interpret_style, analytical_mode, analytical_part, operator_list, int_list_1, int_list_2, int_list_3, pos_list


# generate the numbers and operational relations for holistic problems
def holistic_parser(num_pos, const_value, operator_value):
    num_blanks = num_pos - 1
    operator_list = operator_prune(num_blanks, operator_value)
    integer_list_1, integer_list_2, integer_list_3 = [], [], []
    qualified_1, qualified_2, qualified_3 = False, False, False
    while (not qualified_1) or (not qualified_2) or (not qualified_3):
        operator_list = operator_prune(num_blanks, operator_value)
        integer_list_1, qualified_1 = int_generator(operator_list, const_value[0])
        integer_list_2, qualified_2 = int_generator(operator_list, const_value[1])
        integer_list_3, qualified_3 = int_generator(operator_list, const_value[2])
        random.shuffle(operator_value)
    return operator_list, integer_list_1, integer_list_2, integer_list_3


# generate the numbers and operational relations for analytical problem
def analytical_parser(num_pos, const_value, operator_value, analytical_part):
    num_blanks = num_pos/analytical_part - 1
    operator_list = operator_prune(num_blanks, operator_value)
    integer_list_1, integer_list_2, integer_list_3 = [], [], []
    qualified_1, qualified_2, qualified_3, qualified_4, qualified_5, qualified_6 = False, False, False, False, False, False
    qualified_7, qualified_8, qualified_9, qualified_10, qualified_11, qualified_12 = False, False, False, False, False, False

    if analytical_part == 2:
        while not (qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6):
            operator_list = operator_prune(num_blanks, operator_value)
            part_1_1, qualified_1 = int_generator(operator_list, const_value[0])
            part_1_2, qualified_2 = int_generator(operator_list, const_value[0])
            part_2_1, qualified_3 = int_generator(operator_list, const_value[1])
            part_2_2, qualified_4 = int_generator(operator_list, const_value[1])
            part_3_1, qualified_5 = int_generator(operator_list, const_value[2])
            part_3_2, qualified_6 = int_generator(operator_list, const_value[2])
            random.shuffle(operator_value)
        integer_list_1 = part_1_1 + part_1_2
        integer_list_2 = part_2_1 + part_2_2
        integer_list_3 = part_3_1 + part_3_2

    if analytical_part == 3:
        while not (qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6
                   and qualified_7 and qualified_8 and qualified_9):
            operator_list = operator_prune(num_blanks, operator_value)
            part_1_1, qualified_1 = int_generator(operator_list, const_value[0])
            part_1_2, qualified_2 = int_generator(operator_list, const_value[0])
            part_1_3, qualified_3 = int_generator(operator_list, const_value[0])
            part_2_1, qualified_4 = int_generator(operator_list, const_value[1])
            part_2_2, qualified_5 = int_generator(operator_list, const_value[1])
            part_2_3, qualified_6 = int_generator(operator_list, const_value[1])
            part_3_1, qualified_7 = int_generator(operator_list, const_value[2])
            part_3_2, qualified_8 = int_generator(operator_list, const_value[2])
            part_3_3, qualified_9 = int_generator(operator_list, const_value[2])
            random.shuffle(operator_value)
        integer_list_1 = part_1_1 + part_1_2 + part_1_3
        integer_list_2 = part_2_1 + part_2_2 + part_2_3
        integer_list_3 = part_3_1 + part_3_2 + part_3_3

    if analytical_part == 4:
        while not (qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6
                   and qualified_7 and qualified_8 and qualified_9 and qualified_10 and qualified_11 and qualified_12):
            operator_list = operator_prune(num_blanks, operator_value)
            part_1_1, qualified_1 = int_generator(operator_list, const_value[0])
            part_1_2, qualified_2 = int_generator(operator_list, const_value[0])
            part_1_3, qualified_3 = int_generator(operator_list, const_value[0])
            part_1_4, qualified_4 = int_generator(operator_list, const_value[0])
            part_2_1, qualified_5 = int_generator(operator_list, const_value[1])
            part_2_2, qualified_6 = int_generator(operator_list, const_value[1])
            part_2_3, qualified_7 = int_generator(operator_list, const_value[1])
            part_2_4, qualified_8 = int_generator(operator_list, const_value[1])
            part_3_1, qualified_9 = int_generator(operator_list, const_value[2])
            part_3_2, qualified_10 = int_generator(operator_list, const_value[2])
            part_3_3, qualified_11 = int_generator(operator_list, const_value[2])
            part_3_4, qualified_12 = int_generator(operator_list, const_value[2])
            random.shuffle(operator_value)
        integer_list_1 = part_1_1 + part_1_2 + part_1_3 + part_1_4
        integer_list_2 = part_2_1 + part_2_2 + part_2_3 + part_2_4
        integer_list_3 = part_3_1 + part_3_2 + part_3_3 + part_3_4

    return operator_list, integer_list_1, integer_list_2, integer_list_3


# Decide whether to display the constant on the center of each panel as hints
def whether_center(prob_type, geom_conditions):
    show_center = False
    if prob_type == "Composition":
        show_center = np.random.choice([True, False])
    elif prob_type == "Combination":
        condition_2 = geom_conditions[1]
        geom_relation = condition_2.grelation
        if geom_relation == "overlap" or geom_relation == "include":
            show_center = np.random.choice([True, False])
    return show_center


# Prune the sampled operators for number generation use
def operator_prune(num_blanks, operator_value):
    insert_paren = 0
    operator_list = []
    if num_blanks == 1:
        operator_list.append(operator_value[0])
    else: # randomly introduce parentheses (only once, use insert_paren to mark it)
        for i in range(num_blanks):
            if operator_value[i] == "+" or operator_value[i] == "-":
                if insert_paren == 0 and i < num_blanks - 1:
                    insert_paren = np.random.choice([0, 1])
            if insert_paren == 1:
                operator_list.extend(["(", operator_value[i], ")"])
                insert_paren = 2
            else:
                operator_list.append(operator_value[i])
    return operator_list


# generate the numbers based on a calculation process defined by the given operators and constants
def int_generator(operator_list, const_value):
    filled_expression = fill_expression(operator_list)
    after_expression = middle_to_after(filled_expression)
    node_list, levels = build_calculator_tree(after_expression)
    new_node_list, integer_list, qualified = number_sampler(node_list, const_value, levels)
    # document the expression for calculation
    final_expression = str(integer_list[0])
    previous_index = 0
    previous_num = str(integer_list[previous_index])
    for i in range(len(operator_list)):
        if operator_list[i] == "(" and i == 0:
            final_expression = "(" + previous_num
        elif i < len(operator_list) - 1 and operator_list[i+1] == "(":
            final_expression += operator_list[i]
        elif operator_list[i] == ")":
            final_expression += ")"
        else:
            previous_index += 1
            previous_num = str(integer_list[previous_index])
            final_expression += operator_list[i] + previous_num
    final_expression += "=" + str(const_value)
    return integer_list, qualified


# display the number and hints on the panels, save the answer
def disp_num(img, int_list, pos_list, blank, font_size, show_center):
    num_pos = len(pos_list)
    mark_position = "none"
    answer = 0
    if blank:
        if show_center:
            mark = np.random.choice(range(0, num_pos - 1))
            mark_position = mark
        else:
            mark = np.random.choice(range(0, num_pos))
            mark_position = mark
        for i in range(0, num_pos):
            (x, y) = pos_list[i]
            if i != mark:
                cv2.putText(img, str(int_list[i]), (x-9, y+4), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 2)
            else:
                cv2.putText(img, "?", (x-9, y+4), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 2)
                answer = int_list[i]
    else:
        for i in range(0, num_pos):
            (x, y) = pos_list[i]
            cv2.putText(img, str(int_list[i]), (x-9, y+4), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 2)
    return answer, img, mark_position


# permutate the integer list to render its order as the clockwise order of the integers appeared on the panel
def reverse_arrange(int_list_1, int_list_2, int_list_3, mode, part):
    num = len(int_list_1)
    new_list_1, new_list_2, new_list_3 = [], [], []
    if mode == 2:
        if part == 2:
            list_1_1, list_1_2 = int_list_1[0: num / 2], int_list_1[num / 2: num]
            list_2_1, list_2_2 = int_list_2[0: num / 2], int_list_2[num / 2: num]
            list_3_1, list_3_2 = int_list_3[0: num / 2], int_list_3[num / 2: num]
            for i in range(len(list_1_1)):
                new_list_1.extend([list_1_1[i], list_1_2[i]])
                new_list_2.extend([list_2_1[i], list_2_2[i]])
                new_list_3.extend([list_3_1[i], list_3_2[i]])
        elif part == 3:
            list_1_1, list_1_2, list_1_3 = int_list_1[0: num / 3], int_list_1[num / 3: 2 * num / 3], int_list_1[2 * num / 3: num]
            list_2_1, list_2_2, list_2_3 = int_list_2[0: num / 3], int_list_2[num / 3: 2 * num / 3], int_list_2[2 * num / 3: num]
            list_3_1, list_3_2, list_3_3 = int_list_3[0: num / 3], int_list_3[num / 3: 2 * num / 3], int_list_3[2 * num / 3: num]
            for i in range(len(list_1_1)):
                new_list_1.extend([list_1_1[i], list_1_2[i], list_1_3[i]])
                new_list_2.extend([list_2_1[i], list_2_2[i], list_2_3[i]])
                new_list_3.extend([list_3_1[i], list_3_2[i], list_3_3[i]])
        elif part == 4:
            list_1_1, list_1_2 = int_list_1[0: num / 4], int_list_1[num / 4: num / 2]
            list_1_3, list_1_4 = int_list_1[num / 2: 3 * num / 4], int_list_1[3 * num / 4: num]
            list_2_1, list_2_2 = int_list_2[0: num / 4], int_list_2[num / 4: num / 2]
            list_2_3, list_2_4 = int_list_2[num / 2: 3 * num / 4], int_list_2[3 * num / 4: num]
            list_3_1, list_3_2 = int_list_3[0: num / 4], int_list_3[num / 4: num / 2]
            list_3_3, list_3_4 = int_list_3[num / 2: 3 * num / 4], int_list_3[3 * num / 4: num]
            for i in range(len(list_1_1)):
                new_list_1.extend([list_1_1[i], list_1_2[i], list_1_3[i], list_1_4[i]])
                new_list_2.extend([list_2_1[i], list_2_2[i], list_2_3[i], list_2_4[i]])
                new_list_3.extend([list_3_1[i], list_3_2[i], list_3_3[i], list_3_4[i]])
    elif mode == 3:
        new_list_1, list_1_2 = int_list_1[0: num / 2], int_list_1[num / 2: num]
        new_list_2, list_2_2 = int_list_2[0: num / 2], int_list_2[num / 2: num]
        new_list_3, list_3_2 = int_list_3[0: num / 2], int_list_3[num / 2: num]
        if num == 8:
            new_list_1.extend([list_1_2[3], list_1_2[2], list_1_2[1], list_1_2[0]])
            new_list_2.extend([list_2_2[3], list_2_2[2], list_2_2[1], list_2_2[0]])
            new_list_3.extend([list_3_2[3], list_3_2[2], list_3_2[1], list_3_2[0]])
        elif num == 6:
            new_list_1.extend([list_1_2[2], list_1_2[1], list_1_2[0]])
            new_list_2.extend([list_2_2[2], list_2_2[1], list_2_2[0]])
            new_list_3.extend([list_3_2[2], list_3_2[1], list_3_2[0]])
        elif num == 4:
            new_list_1.extend([list_1_2[1], list_1_2[0]])
            new_list_2.extend([list_2_2[1], list_2_2[0]])
            new_list_3.extend([list_3_2[1], list_3_2[0]])
    elif mode == 4 and num == 8:
        list_1_1, list_1_2 = int_list_1[0: num / 2], int_list_1[num / 2: num]
        list_2_1, list_2_2 = int_list_2[0: num / 2], int_list_2[num / 2: num]
        list_3_1, list_3_2 = int_list_3[0: num / 2], int_list_3[num / 2: num]
        new_list_1 = [list_1_1[0], list_1_2[0], list_1_1[3], list_1_2[3], list_1_1[1], list_1_2[1], list_1_1[2], list_1_2[2]]
        new_list_2 = [list_2_1[0], list_2_2[0], list_2_1[3], list_2_2[3], list_2_1[1], list_2_2[1], list_2_1[2], list_2_2[2]]
        new_list_3 = [list_3_1[0], list_3_2[0], list_3_1[3], list_3_2[3], list_3_1[1], list_3_2[1], list_3_1[2], list_3_2[2]]

    return new_list_1, new_list_2, new_list_3

'''
        if prob_type == "Combination" and gcondition_1.type.get_value() == "triangle" and gcondition_2.grelation.get_value() == "include":
            if analytical_part == 2:
                pos_list = positions
                analytical_mode = 1
            elif analytical_part == 3:
                analytical_mode = 2
                pos_list_1, pos_list_2, pos_list_3 = [], [], []
                for i in range(num_pos):
                    if i % 3 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 3 == 1:
                        pos_list_2.append(positions[i])
                    elif i % 3 == 2:
                        pos_list_3.append(positions[i])
                pos_list = pos_list_1 + pos_list_2 + pos_list_3

        elif prob_type == "Composition" and (gcondition_2.format.get_value() == "circle" or gcondition_2.format.get_value() == "square"):
            if analytical_part == 2:
                analytical_mode = 2
                pos_list_1, pos_list_2 = [], []
                for i in range(num_pos):
                    if i % 2 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 2 == 1:
                        pos_list_2.append(positions[i])
                pos_list = pos_list_1 + pos_list_2
            elif analytical_part == 4:
                analytical_mode = 2
                pos_list_1, pos_list_2, pos_list_3, pos_list_4 = [], [], [], []
                for i in range(num_pos):
                    if i % 4 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 4 == 1:
                        pos_list_2.append(positions[i])
                    elif i % 4 == 2:
                        pos_list_3.append(positions[i])
                    elif i % 4 == 3:
                        pos_list_4.append(positions[i])
                pos_list = pos_list_1 + pos_list_2 + pos_list_3 + pos_list_4

        elif prob_type == "Composition" and gcondition_2.format.get_value() == "triangle":
            if analytical_part == 2:
                analytical_mode = 2
                pos_list_1, pos_list_2 = [], []
                for i in range(num_pos):
                    if i % 2 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 2 == 1:
                        pos_list_2.append(positions[i])
                pos_list = pos_list_1 + pos_list_2
            elif analytical_part == 3:
                analytical_mode = 2
                pos_list_1, pos_list_2, pos_list_3 = [], [], []
                for i in range(num_pos):
                    if i % 3 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 3 == 1:
                        pos_list_2.append(positions[i])
                    elif i % 3 == 2:
                        pos_list_3.append(positions[i])
                pos_list = pos_list_1 + pos_list_2 + pos_list_3
'''