# -*- coding: utf-8 -*-

import numpy as np
import time
import copy
from Calculator_Tree import (generate_tree, prune_generate_tree, NStack, middle_search, post_search)


"""
The solver solves the problem by searching algorithms
The inputs include whether there is a centered number and the number list of each panel
Given predefined set of hypotheses, possible calculator orders, possible calculator trees, it search through the problem
space to get a qualified answer
"""

"""
Here are the predefined hypotheses,
including the interpretation style of numbers, the number of analytical parts, and the structure of calculator trees
"""
interpretations = ["analytical", "holistic"]
analytical_parts = [2, 3, 4]
tree_list_2 = generate_tree(2)
prune_tree_list_2 = prune_generate_tree(tree_list_2)
tree_list_3 = generate_tree(3)
prune_tree_list_3 = prune_generate_tree(tree_list_3)
tree_list_4 = generate_tree(4)
prune_tree_list_4 = prune_generate_tree(tree_list_4)
tree_list_6 = generate_tree(6)
prune_tree_list_6 = prune_generate_tree(tree_list_6)
tree_list_8 = generate_tree(8)
prune_tree_list_8 = prune_generate_tree(tree_list_8)
tree_dict = {2: prune_tree_list_2,
             3: prune_tree_list_3,
             4: prune_tree_list_4,
             6: prune_tree_list_6,
             8: prune_tree_list_8}


# generate all the possible orders of calculation
def possible_orders(num):
    order_list = []
    for i in range(num):
        order = []
        start = i
        for j in range(num):
            order.append(start)
            start += 1
            if start > num - 1:
                start = num - start
        order_list.append(order)
    return order_list


# permute the numbers according to the generated order
def arrange_number(order, num_list):
    new_num_list = []
    for index in order:
        new_num_list.append(num_list[index])
    return new_num_list


# generate all the possible combination of operators
def possible_operator(num):
    operator_list = []
    if num == 1:
        operator_list = [["+"], ["-"], ["*"], ["/"]]
    else:
        previous_list = possible_operator(num - 1)
        for previous_operator in previous_list:
            current_operator_1, current_operator_2 = copy.deepcopy(previous_operator), copy.deepcopy(previous_operator)
            current_operator_3, current_operator_4 = copy.deepcopy(previous_operator), copy.deepcopy(previous_operator)
            current_operator_1.append("+")
            current_operator_2.append("-")
            current_operator_3.append("*")
            current_operator_4.append("/")
            operator_list.extend([current_operator_1, current_operator_2, current_operator_3, current_operator_4])
    return operator_list


# annotate the generated operators on the nodes of the calculator tree
def middle_annotation(operators, input_root):
    i = 0
    root = copy.deepcopy(input_root)
    node_search_list = middle_search(root)
    for node in node_search_list:
        if (not node.is_leaf()) and node.operator == "none":
            node.set_operator(operators[i])
            i += 1
    return root


# use the integers on each panel to calculate the hidden constant(the value attribute of root node)
def back_calculation(int_list, root):
    node_search_list = middle_search(root)
    i = 0
    for node in node_search_list:
        if node.is_leaf():
            node.set_value(int(int_list[i]))
            i += 1
    node_stack = post_search(root)
    # calculate the root value in a bottom-up manner
    final_value = 0
    qualified = True
    while not node_stack.is_empty():
        node = node_stack.pop()
        if not node.is_leaf():
            value_1 = node.lchild.get_value()
            value_2 = node.rchild.get_value()
            operate = node.operator
            if operate == "-":
                if value_1 <= value_2:
                    qualified = False
                    break
            elif operate == "/":
                if value_2 == 0 or value_1 % value_2 != 0:
                    qualified = False
                    break
            expression = str(value_1) + str(operate) + str(value_2)
            final_value = eval(expression)
            node.set_value(final_value)
    if final_value < 1 or final_value > 99:
        qualified = False

    return final_value, qualified


# Apply the rules and constants of the first two panels to the third panel, and get the answer
def get_answer(calculate_root, calculate_list, outcome):
    answer = 0
    qualified = True
    calculate_root.set_value(int(outcome))
    node_search_list = middle_search(calculate_root)
    i = 0
    for node in node_search_list:
        if node.is_leaf():
            if calculate_list[i] != "mark":
                node.set_value(int(calculate_list[i]))
            else:
                node.set_value(calculate_list[i])
            i += 1
    # calculate the values of most nodes in a bottom-up manner
    node_stack = post_search(calculate_root)
    while not node_stack.is_empty():
        node = node_stack.pop()
        if not node.is_leaf():
            if node.lchild.get_value() != "none" and node.lchild.get_value() != "mark" \
                    and node.rchild.get_value() != "none" and node.rchild.get_value() != "mark":
                value_1 = node.lchild.get_value()
                value_2 = node.rchild.get_value()
                operate = node.operator
                if operate == "-" and value_1 <= value_2:
                    qualified = False
                    break
                elif operate == "/" and (value_2 == 0 or value_1 % value_2 != 0):
                    qualified = False
                    break
                expression = str(value_1) + str(operate) + str(value_2)
                operate_value = eval(expression)
                node.set_value(operate_value)
    # calculate the answer in a top-down way (by pre-order traversal)
    if qualified:
        stack_3 = NStack()
        node = calculate_root
        while (node != "none") or (not stack_3.is_empty()):
            while node != "none":
                if (node.get_value() == "none" or node.get_value() == "mark") and node.parent.get_value() != "none":
                    node_neighbor, direction = node.get_neighbor()
                    if node.parent.get_operator() == "+":
                        current_value = node.parent.get_value() - node_neighbor.get_value()
                        if node.parent.get_value() <= node_neighbor.get_value():
                            qualified = False
                            break
                    if node.parent.get_operator() == "-":
                        if direction == "left neighbor":
                            current_value = node_neighbor.get_value() - node.parent.get_value()
                            if node.parent.get_value() >= node_neighbor.get_value():
                                qualified = False
                                break
                        elif direction == "right neighbor":
                            current_value = node_neighbor.get_value() + node.parent.get_value()
                    if node.parent.get_operator() == "*":
                        current_value = node.parent.get_value() / node_neighbor.get_value()
                        if node.parent.get_value() % node_neighbor.get_value() != 0:
                            qualified = False
                            break
                    if node.parent.get_operator() == "/":
                        if direction == "left neighbor":
                            current_value = node_neighbor.get_value() / node.parent.get_value()
                            if node_neighbor.get_value() % node.parent.get_value() != 0:
                                qualified = False
                                break
                        elif direction == "right neighbor":
                            current_value = node_neighbor.get_value() * node.parent.get_value()
                    node.set_value(current_value)
                    if node.is_leaf():
                        answer = current_value

                stack_3.push(node)
                node = node.lchild
            if not qualified:
                break
            node = stack_3.pop()
            node = node.rchild

    if answer < 1 or answer > 99:
        qualified = False

    return answer, qualified


# search the answer for holistic problems (问题：是否可以把order和operator提前生成，而后直接调用？）
def search_holistic(max_trials, previous_trials, tlist, list_1, list_2, list_3, center, geom_confirm):
    answer = "none"
    num = len(list_1)
    order_list = possible_orders(num)
    if geom_confirm:
        order_list = [order_list[0]]
    operator_list = possible_operator(num - 1)

    range_for_order = range(len(order_list))
    for a in range(len(order_list)):
        order_index = np.random.choice(range_for_order)
        order = order_list[order_index]
        range_for_order.remove(order_index)
        flag_0 = False
        new_list_1, new_list_2, new_list_3 = arrange_number(order, list_1), arrange_number(order, list_2), arrange_number(order, list_3)

        range_for_operator = range(len(operator_list))
        for i in range(len(operator_list)):
            operator_index = np.random.choice(range_for_operator)
            operator = operator_list[operator_index]
            range_for_operator.remove(operator_index)
            flag_1 = False

            trees = copy.deepcopy(tlist)
            for j in range(len(tlist)):
                tree_root = np.random.choice(trees)
                trees.remove(tree_root)
                previous_trials += 1
                if previous_trials >= max_trials:
                    break
                root = middle_annotation(operator, tree_root)
                root_1 = copy.deepcopy(root)
                root_2 = copy.deepcopy(root)
                outcome_1, qualified_1 = back_calculation(new_list_1, root_1)
                outcome_2, qualified_2 = back_calculation(new_list_2, root_2)
                if center[0] == "none":
                    if outcome_1 == outcome_2 and qualified_1 and qualified_2:
                        outcome = outcome_1
                        calculate_root = copy.deepcopy(root)
                        calculate_list = new_list_3
                        possible_answer, qualified_3 = get_answer(calculate_root, calculate_list, outcome)
                        if qualified_3:
                            answer = possible_answer
                            flag_1 = True
                            break
                else:
                    if outcome_1 == center[0] and outcome_2 == center[1] and qualified_1 and qualified_2:
                        outcome = center[2]
                        calculate_root = copy.deepcopy(root)
                        calculate_list = new_list_3
                        possible_answer, qualified_3 = get_answer(calculate_root, calculate_list, outcome)
                        if qualified_3:
                            answer = possible_answer
                            flag_1 = True
                            break
            if flag_1 or (previous_trials >= max_trials):
                flag_0 = True
                break
        if flag_0:
            break
    return answer, previous_trials


# search the answer for two-part analytical problems
def search_analytical_2(max_trials, previous_trials, tlist, list_1, list_2, list_3, center, analytical_mode, geom_confirm):
    answer = "none"
    num = len(list_1)
    order_list = possible_orders(num)
    if geom_confirm:
        order_list = [order_list[0]]
    operator_list = possible_operator(num/2 - 1)

    range_for_order = range(len(order_list))
    for a in range(len(order_list)):
        order_index = np.random.choice(range_for_order)
        order = order_list[order_index]
        range_for_order.remove(order_index)
        flag_0 = False
        new_list_1 = arrange_number(order, list_1)
        new_list_2 = arrange_number(order, list_2)
        new_list_3 = arrange_number(order, list_3)

        modes = copy.deepcopy(analytical_mode)
        for j in range(len(analytical_mode)):
            mode = np.random.choice(modes)
            modes.remove(mode)
            flag_1 = False
            if mode == 1 or mode == 3:
                list_1_1, list_1_2 = new_list_1[0: num/2], new_list_1[num/2: num]
                list_2_1, list_2_2 = new_list_2[0: num/2], new_list_2[num/2: num]
                list_3_1, list_3_2 = new_list_3[0: num/2], new_list_3[num/2: num]
                if mode == 3:
                    if len(list_1_2) == 2:
                        new_list_1_2, new_list_2_2, new_list_3_2 = [list_1_2[1], list_1_2[0]], [list_2_2[1], list_2_2[0]], [list_3_2[1], list_3_2[0]]
                    elif len(list_1_2) == 3:
                        new_list_1_2, new_list_2_2, new_list_3_2 = [list_1_2[2], list_1_2[1], list_1_2[0]], [list_2_2[2], list_2_2[1], list_2_2[0]], \
                                                                   [list_3_2[2], list_3_2[1], list_3_2[0]]
                    elif len(list_1_2) == 4:
                        new_list_1_2, new_list_2_2, new_list_3_2 = [list_1_2[3], list_1_2[2], list_1_2[1], list_1_2[0]], \
                                                                   [list_2_2[3], list_2_2[2], list_2_2[1], list_2_2[0]], \
                                                                   [list_3_2[3], list_3_2[2], list_3_2[1], list_3_2[0]]
                    list_1_1, list_2_1, list_3_1, list_1_2, list_2_2, list_3_2 = list_1_1, list_2_1, list_3_1, new_list_1_2, new_list_2_2, new_list_3_2
            else:
                list_1_1, list_1_2, list_2_1, list_2_2, list_3_1, list_3_2 = [], [], [], [], [], []
                for i in range(num):
                    if i % 2 == 0:
                        list_1_1.append(new_list_1[i])
                        list_2_1.append(new_list_2[i])
                        list_3_1.append(new_list_3[i])
                    else:
                        list_1_2.append(new_list_1[i])
                        list_2_2.append(new_list_2[i])
                        list_3_2.append(new_list_3[i])
                if len(list_1_1) == 4 and mode == 4:
                    new_list_1_1, new_list_1_2 = [list_1_1[0], list_1_1[2], list_1_1[3], list_1_1[1]], [list_1_2[0], list_1_2[2], list_1_2[3], list_1_2[1]]
                    new_list_2_1, new_list_2_2 = [list_2_1[0], list_2_1[2], list_2_1[3], list_2_1[1]], [list_2_2[0], list_2_2[2], list_2_2[3], list_2_2[1]]
                    new_list_3_1, new_list_3_2 = [list_3_1[0], list_3_1[2], list_3_1[3], list_3_1[1]], [list_3_2[0], list_3_2[2], list_3_2[3], list_3_2[1]]
                    list_1_1, list_2_1, list_3_1, list_1_2, list_2_2, list_3_2 = new_list_1_1, new_list_2_1, new_list_3_1, new_list_1_2, new_list_2_2, new_list_3_2

            choose_range = range(len(operator_list))
            for p in range(len(operator_list)):
                operator_index = np.random.choice(choose_range)
                operator = operator_list[operator_index]
                choose_range.remove(operator_index)
                flag_2 = False

                trees = copy.deepcopy(tlist)
                for q in range(len(tlist)):
                    tree_root = np.random.choice(trees)
                    trees.remove(tree_root)
                    previous_trials += 1
                    if previous_trials >= max_trials:
                        break
                    root = middle_annotation(operator, tree_root)
                    root_1, root_2, root_3 = copy.deepcopy(root), copy.deepcopy(root), copy.deepcopy(root)
                    root_4, root_5, calculate_root = copy.deepcopy(root), copy.deepcopy(root), copy.deepcopy(root)
                    outcome_1, qualified_1 = back_calculation(list_1_1, root_1)
                    outcome_2, qualified_2 = back_calculation(list_1_2, root_2)
                    outcome_3, qualified_3 = back_calculation(list_2_1, root_3)
                    outcome_4, qualified_4 = back_calculation(list_2_2, root_4)
                    if center[0] == "none":
                        if outcome_1 == outcome_2 and outcome_3 == outcome_4 and outcome_1 == outcome_3 and \
                                qualified_1 and qualified_2 and qualified_3 and qualified_4:
                            if "mark" in list_3_1:
                                outcome_5, qualified_5 = back_calculation(list_3_2, root_5)
                                calculate_list = list_3_1
                            elif "mark" in list_3_2:
                                outcome_5, qualified_5 = back_calculation(list_3_1, root_5)
                                calculate_list = list_3_2
                            outcome = outcome_1
                            possible_answer, qualified_6 = get_answer(calculate_root, calculate_list, outcome)
                            if outcome_5 == outcome_1 and qualified_5 and qualified_6:
                                answer = possible_answer
                                flag_2 = True
                                break
                    else:
                        if outcome_1 == center[0] and outcome_2 == center[0] and outcome_3 == center[1] and outcome_4 == center[1] and \
                                qualified_1 and qualified_2 and qualified_3 and qualified_4:
                            if "mark" in list_3_1:
                                outcome_5, qualified_5 = back_calculation(list_3_2, root_5)
                                calculate_list = list_3_1
                            elif "mark" in list_3_2:
                                outcome_5, qualified_5 = back_calculation(list_3_1, root_5)
                                calculate_list = list_3_2
                            outcome = center[2]
                            possible_answer, qualified_6 = get_answer(calculate_root, calculate_list, outcome)
                            if outcome_5 == center[2] and qualified_5 and qualified_6:
                                answer = possible_answer
                                flag_2 = True
                                break
                if flag_2 or (previous_trials >= max_trials):
                    flag_1 = True
                    break
            if flag_1:
                flag_0 = True
                break
        if flag_0:
            break

    return answer, previous_trials


# search the answer for three-part analytical problems
def search_analytical_3(max_trials, previous_trials, tlist, list_1, list_2, list_3, center, analytical_mode, geom_confirm):
    answer = "none"
    num = len(list_1)
    order_list = possible_orders(num)
    if geom_confirm:
        order_list = [order_list[0]]
    operator_list = possible_operator(num/3 - 1)

    range_for_order = range(len(order_list))
    for a in range(len(order_list)):
        order_index = np.random.choice(range_for_order)
        order = order_list[order_index]
        range_for_order.remove(order_index)
        flag_0 = False
        new_list_1 = arrange_number(order, list_1)
        new_list_2 = arrange_number(order, list_2)
        new_list_3 = arrange_number(order, list_3)

        modes = copy.deepcopy(analytical_mode)
        for j in range(len(analytical_mode)):
            mode = np.random.choice(modes)
            modes.remove(mode)
            flag_1 = False
            list_1_1, list_1_2, list_1_3, list_2_1, list_2_2, list_2_3, list_3_1, list_3_2, list_3_3 = [], [], [], [], [], [], [], [], []
            if mode == 1 or mode == 3:
                list_1_1, list_1_2, list_1_3 = new_list_1[0: num/3], new_list_1[num/3: 2*num/3], new_list_1[2*num/3: num]
                list_2_1, list_2_2, list_2_3 = new_list_2[0: num/3], new_list_2[num/3: 2*num/3], new_list_2[2*num/3: num]
                list_3_1, list_3_2, list_3_3 = new_list_3[0: num/3], new_list_3[num/3: 2*num/3], new_list_3[2*num/3: num]
                if mode == 3:
                    new_list_1_3, new_list_2_3, new_list_3_3 = [list_1_3[1], list_1_3[0]], [list_2_3[1], list_2_3[0]], [list_3_3[1], list_3_3[0]]
                    list_1_1, list_2_1, list_3_1, list_1_3, list_2_3, list_3_3 = list_1_1, list_2_1, list_3_1, new_list_1_3, new_list_2_3, new_list_3_3
            elif mode == 2:
                for i in range(num):
                    if i % 3 == 0:
                        list_1_1.append(new_list_1[i])
                        list_2_1.append(new_list_2[i])
                        list_3_1.append(new_list_3[i])
                    elif i % 3 == 1:
                        list_1_2.append(new_list_1[i])
                        list_2_2.append(new_list_2[i])
                        list_3_2.append(new_list_3[i])
                    else:
                        list_1_3.append(new_list_1[i])
                        list_2_3.append(new_list_2[i])
                        list_3_3.append(new_list_3[i])

            choose_range = range(len(operator_list))
            for p in range(len(operator_list)):
                operator_index = np.random.choice(choose_range)
                operator = operator_list[operator_index]
                choose_range.remove(operator_index)
                flag_2 = False

                trees = copy.deepcopy(tlist)
                for q in range(len(tlist)):
                    tree_root = np.random.choice(trees)
                    trees.remove(tree_root)
                    previous_trials += 1
                    if previous_trials >= max_trials:
                        break
                    root = middle_annotation(operator, tree_root)
                    root_1, root_2, root_3, root_4, root_5, root_6, root_7, root_8 = copy.deepcopy(root), copy.deepcopy(root), \
                                                                                     copy.deepcopy(root), copy.deepcopy(root), \
                                                                                     copy.deepcopy(root), copy.deepcopy(root), \
                                                                                     copy.deepcopy(root), copy.deepcopy(root)
                    calculate_root = copy.deepcopy(root)
                    outcome_1, qualified_1 = back_calculation(list_1_1, root_1)
                    outcome_2, qualified_2 = back_calculation(list_1_2, root_2)
                    outcome_3, qualified_3 = back_calculation(list_1_3, root_3)
                    outcome_4, qualified_4 = back_calculation(list_2_1, root_4)
                    outcome_5, qualified_5 = back_calculation(list_2_2, root_5)
                    outcome_6, qualified_6 = back_calculation(list_2_3, root_6)
                    if center[0] == "none":
                        if outcome_1 == outcome_2 and outcome_1 == outcome_3 and \
                                outcome_4 == outcome_5 and outcome_4 == outcome_6 and outcome_1 == outcome_4 and \
                                qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6:
                            if "mark" in list_3_1:
                                outcome_7, qualified_7 = back_calculation(list_3_2, root_7)
                                outcome_8, qualified_8 = back_calculation(list_3_3, root_8)
                                calculate_list = list_3_1
                            elif "mark" in list_3_2:
                                outcome_7, qualified_7 = back_calculation(list_3_1, root_7)
                                outcome_8, qualified_8 = back_calculation(list_3_3, root_8)
                                calculate_list = list_3_2
                            elif "mark" in list_3_3:
                                outcome_7, qualified_7 = back_calculation(list_3_1, root_7)
                                outcome_8, qualified_8 = back_calculation(list_3_2, root_8)
                                calculate_list = list_3_3
                            outcome = outcome_1
                            possible_answer, qualified_9 = get_answer(calculate_root, calculate_list, outcome)
                            if outcome_7 == outcome_1 and outcome_8 == outcome_1 and qualified_7 and qualified_8 and qualified_9:
                                answer = possible_answer
                                flag_2 = True
                                break
                    else:
                        if outcome_1 == center[0] and outcome_2 == center[0] and outcome_3 == center[0] and \
                                outcome_4 == center[1] and outcome_5 == center[1] and outcome_6 == center[1] and \
                                qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6:
                            if "mark" in list_3_1:
                                outcome_7, qualified_7 = back_calculation(list_3_2, root_7)
                                outcome_8, qualified_8 = back_calculation(list_3_3, root_8)
                                calculate_list = list_3_1
                            elif "mark" in list_3_2:
                                outcome_7, qualified_7 = back_calculation(list_3_1, root_7)
                                outcome_8, qualified_8 = back_calculation(list_3_3, root_8)
                                calculate_list = list_3_2
                            elif "mark" in list_3_3:
                                outcome_7, qualified_7 = back_calculation(list_3_1, root_7)
                                outcome_8, qualified_8 = back_calculation(list_3_2, root_8)
                                calculate_list = list_3_3
                            outcome = center[2]
                            possible_answer, qualified_9 = get_answer(calculate_root, calculate_list, outcome)
                            if outcome_7 == center[2] and outcome_8 == center[2] and qualified_7 and qualified_8 and qualified_9:
                                answer = possible_answer
                                flag_2 = True
                                break
                if flag_2 or (previous_trials >= max_trials):
                    flag_1 = True
                    break
            if flag_1:
                flag_0 = True
                break
        if flag_0:
            break

    return answer, previous_trials


# search the answer for four-part analytical problems
def search_analytical_4(max_trials, previous_trials, tlist, list_1, list_2, list_3, center, analytical_mode, geom_confirm):
    answer = "none"
    num = len(list_1)
    order_list = possible_orders(num)
    if geom_confirm:
        order_list = [order_list[0]]
    operator_list = possible_operator(num/4 - 1)

    range_for_order = range(len(order_list))
    for a in range(len(order_list)):
        order_index = np.random.choice(range_for_order)
        order = order_list[order_index]
        range_for_order.remove(order_index)
        flag_0 = False
        new_list_1 = arrange_number(order, list_1)
        new_list_2 = arrange_number(order, list_2)
        new_list_3 = arrange_number(order, list_3)

        modes = copy.deepcopy(analytical_mode)
        for j in range(len(analytical_mode)):
            mode = np.random.choice(modes)
            modes.remove(mode)
            flag_1 = False
            list_1_1, list_1_2, list_1_3, list_1_4, list_2_1, list_2_2, list_2_3, list_2_4, \
            list_3_1, list_3_2, list_3_3, list_3_4 = [], [], [], [], [], [], [], [], [], [], [], []
            if mode == 1 or mode == 3:
                list_1_1, list_1_2, list_1_3, list_1_4 = new_list_1[0: num/4], new_list_1[num/4: num/2], new_list_1[num/2: 3*num/4], new_list_1[3*num/4: num]
                list_2_1, list_2_2, list_2_3, list_2_4 = new_list_2[0: num/4], new_list_2[num/4: num/2], new_list_2[num/2: 3*num/4], new_list_2[3*num/4: num]
                list_3_1, list_3_2, list_3_3, list_3_4 = new_list_3[0: num/4], new_list_3[num/4: num/2], new_list_3[num/2: 3*num/4], new_list_3[3*num/4: num]
                if mode == 3:
                    new_list_1_4, new_list_2_4, new_list_3_4 = [list_1_4[1], list_1_4[0]], [list_2_4[1], list_2_4[0]], [list_3_4[1], list_3_4[0]]
                    list_1_2, list_2_2, list_3_2, list_1_4, list_2_4, list_3_4 = list_1_2, list_2_2, list_3_2, new_list_1_4, new_list_2_4, new_list_3_4
            elif mode == 2:
                for i in range(num):
                    if i % 4 == 0:
                        list_1_1.append(new_list_1[i])
                        list_2_1.append(new_list_2[i])
                        list_3_1.append(new_list_3[i])
                    elif i % 4 == 1:
                        list_1_2.append(new_list_1[i])
                        list_2_2.append(new_list_2[i])
                        list_3_2.append(new_list_3[i])
                    elif i % 4 == 2:
                        list_1_3.append(new_list_1[i])
                        list_2_3.append(new_list_2[i])
                        list_3_3.append(new_list_3[i])
                    else:
                        list_1_4.append(new_list_1[i])
                        list_2_4.append(new_list_2[i])
                        list_3_4.append(new_list_3[i])

            choose_range = range(len(operator_list))
            for p in range(len(operator_list)):
                operator_index = np.random.choice(choose_range)
                operator = operator_list[operator_index]
                choose_range.remove(operator_index)
                flag_2 = False

                trees = copy.deepcopy(tlist)
                for q in range(len(tlist)):
                    tree_root = np.random.choice(trees)
                    trees.remove(tree_root)
                    previous_trials += 1
                    if previous_trials >= max_trials:
                        break
                    root = middle_annotation(operator, tree_root)
                    root_1, root_2, root_3, root_4, root_5, root_6, root_7, root_8, root_9, root_10, root_11 = copy.deepcopy(root), copy.deepcopy(root), \
                                                                                                               copy.deepcopy(root), copy.deepcopy(root), \
                                                                                                               copy.deepcopy(root), copy.deepcopy(root), \
                                                                                                               copy.deepcopy(root), copy.deepcopy(root), \
                                                                                                               copy.deepcopy(root), copy.deepcopy(root), copy.deepcopy(root)
                    calculate_root = copy.deepcopy(root)
                    outcome_1, qualified_1 = back_calculation(list_1_1, root_1)
                    outcome_2, qualified_2 = back_calculation(list_1_2, root_2)
                    outcome_3, qualified_3 = back_calculation(list_1_3, root_3)
                    outcome_4, qualified_4 = back_calculation(list_1_4, root_4)
                    outcome_5, qualified_5 = back_calculation(list_2_1, root_5)
                    outcome_6, qualified_6 = back_calculation(list_2_2, root_6)
                    outcome_7, qualified_7 = back_calculation(list_2_3, root_7)
                    outcome_8, qualified_8 = back_calculation(list_2_4, root_8)
                    if center[0] == "none":
                        if outcome_1 == outcome_2 and outcome_1 == outcome_3 and outcome_1 == outcome_4 and outcome_5 == outcome_6 \
                                and outcome_5 == outcome_7 and outcome_5 == outcome_8 and outcome_1 == outcome_5 and \
                                qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6 and qualified_7 and qualified_8:
                            if "mark" in list_3_1:
                                outcome_9, qualified_9 = back_calculation(list_3_2, root_9)
                                outcome_10, qualified_10 = back_calculation(list_3_3, root_10)
                                outcome_11, qualified_11 = back_calculation(list_3_4, root_11)
                                calculate_list = list_3_1
                            elif "mark" in list_3_2:
                                outcome_9, qualified_9 = back_calculation(list_3_1, root_9)
                                outcome_10, qualified_10 = back_calculation(list_3_3, root_10)
                                outcome_11, qualified_11 = back_calculation(list_3_4, root_11)
                                calculate_list = list_3_2
                            elif "mark" in list_3_3:
                                outcome_9, qualified_9 = back_calculation(list_3_1, root_9)
                                outcome_10, qualified_10 = back_calculation(list_3_2, root_10)
                                outcome_11, qualified_11 = back_calculation(list_3_4, root_11)
                                calculate_list = list_3_3
                            elif "mark" in list_3_4:
                                outcome_9, qualified_9 = back_calculation(list_3_1, root_9)
                                outcome_10, qualified_10 = back_calculation(list_3_2, root_10)
                                outcome_11, qualified_11 = back_calculation(list_3_3, root_11)
                                calculate_list = list_3_4
                            outcome = outcome_1
                            possible_answer, qualified_12 = get_answer(calculate_root, calculate_list, outcome)
                            if outcome_9 == outcome_1 and outcome_10 == outcome_1 and outcome_11 == outcome_1 and \
                                    qualified_9 and qualified_10 and qualified_11 and qualified_12:
                                answer = possible_answer
                                flag_2 = True
                                break
                    else:
                        if outcome_1 == center[0] and outcome_2 == center[0] and outcome_3 == center[0] and outcome_4 == center[0] and \
                                outcome_5 == center[1] and outcome_6 == center[1] and outcome_7 == center[1] and outcome_8 == center[1] and \
                                qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6 and qualified_7 and qualified_8:
                            if "mark" in list_3_1:
                                outcome_9, qualified_9 = back_calculation(list_3_2, root_9)
                                outcome_10, qualified_10 = back_calculation(list_3_3, root_10)
                                outcome_11, qualified_11 = back_calculation(list_3_4, root_11)
                                calculate_list = list_3_1
                            elif "mark" in list_3_2:
                                outcome_9, qualified_9 = back_calculation(list_3_1, root_9)
                                outcome_10, qualified_10 = back_calculation(list_3_3, root_10)
                                outcome_11, qualified_11 = back_calculation(list_3_4, root_11)
                                calculate_list = list_3_2
                            elif "mark" in list_3_3:
                                outcome_9, qualified_9 = back_calculation(list_3_1, root_9)
                                outcome_10, qualified_10 = back_calculation(list_3_2, root_10)
                                outcome_11, qualified_11 = back_calculation(list_3_4, root_11)
                                calculate_list = list_3_3
                            elif "mark" in list_3_4:
                                outcome_9, qualified_9 = back_calculation(list_3_1, root_9)
                                outcome_10, qualified_10 = back_calculation(list_3_2, root_10)
                                outcome_11, qualified_11 = back_calculation(list_3_3, root_11)
                                calculate_list = list_3_4
                            outcome = center[2]
                            possible_answer, qualified_12 = get_answer(calculate_root, calculate_list, outcome)
                            if outcome_9 == center[2] and outcome_10 == center[2] and outcome_11 == center[2] and \
                                    qualified_9 and qualified_10 and qualified_11 and qualified_12:
                                answer = possible_answer
                                flag_2 = True
                                break
                if flag_2 or (previous_trials >= max_trials):
                    flag_1 = True
                    break
            if flag_1:
                flag_0 = True
                break
        if flag_0:
            break

    return answer, previous_trials


# search the answer based on pure symbolic input
def math_solve(max_trials, center_number, number_list_1, number_list_2, number_list_3, interpretations, analytical_parts):
    num_of_number = len(number_list_1)
    modes_0 = [1, 2, 3]
    modes_1 = [1, 2, 3, 4]
    start = time.clock()
    if num_of_number == 0:
        print "no input"
    if num_of_number < 4:
        tree_list = tree_dict[num_of_number]
        answer, trials = search_holistic(max_trials, 0, tree_list, number_list_1, number_list_2, number_list_3, center_number, False)
    else:
        trials = 0
        for i in range(2):
            interpret = np.random.choice(interpretations)
            interpretations.remove(interpret)
            flag = False
            if interpret == "holistic":
                tree_list = tree_dict[num_of_number]
                answer, trials_update = search_holistic(max_trials, trials, tree_list, number_list_1, number_list_2,
                                                        number_list_3, center_number, False)
                trials = trials_update
                if answer != "none":
                    flag = True
                if trials >= max_trials:
                    break
            elif interpret == "analytical":
                for analytical_part in analytical_parts:
                    if num_of_number % analytical_part == 0 and num_of_number > analytical_part:
                        tree_list = tree_dict[num_of_number/analytical_part]
                        if analytical_part == 2:
                            if num_of_number == 8:
                                analytical_mode = modes_1
                            else:
                                analytical_mode = modes_0
                            answer, trials_update = search_analytical_2(max_trials, trials, tree_list, number_list_1,
                                                                        number_list_2, number_list_3, center_number,
                                                                        analytical_mode, False)
                            trials = trials_update
                        elif analytical_part == 3:
                            analytical_mode = modes_0
                            answer, trials_update = search_analytical_3(max_trials, trials, tree_list, number_list_1,
                                                                        number_list_2, number_list_3, center_number,
                                                                        analytical_mode, False)
                            trials = trials_update
                        elif analytical_part == 4:
                            analytical_mode = modes_0
                            answer, trials_update = search_analytical_4(max_trials, trials, tree_list, number_list_1,
                                                                        number_list_2, number_list_3, center_number,
                                                                        analytical_mode, False)
                            trials = trials_update
                        if answer != "none":
                            flag = True
                            break
                        if trials >= max_trials:
                            break
            if flag or trials >= max_trials:
                break

    elapsed = time.clock() - start
    # print "The solver used " + str(elapsed) + "s to get the answer"
    return elapsed, answer


# search the answer based on both symbolic input and geometrical information
def geom_math_solve(max_trials, prob_type, geom_conditions, center_number, number_list_1, number_list_2, number_list_3, interpretations, analytical_parts):
    num_of_number = len(number_list_1)
    gcondition_1, gcondition_2 = geom_conditions[0], geom_conditions[1]
    modes_0, modes_1, modes_2, modes_3, modes_4 = [1, 2], [3, 2], [1, 4], [1], [2]
    start = time.clock()
    if num_of_number == 0:
        print "no input"
    if num_of_number < 4:
        tree_list = tree_dict[num_of_number]
        answer, trials = search_holistic(max_trials, 0, tree_list, number_list_1, number_list_2, number_list_3, center_number, True)
    else:
        trials = 0
        for i in range(2):
            interpret = np.random.choice(interpretations)
            interpretations.remove(interpret)
            flag = False
            if interpret == "holistic":
                tree_list = tree_dict[num_of_number]
                answer, trials_update = search_holistic(max_trials, trials, tree_list, number_list_1, number_list_2,
                                                        number_list_3, center_number, True)
                trials = trials_update
                if answer != "none":
                    flag = True
                if trials >= max_trials:
                    break

            elif interpret == "analytical":
                for analytical_part in analytical_parts:
                    if num_of_number % analytical_part == 0 and num_of_number > analytical_part:
                        tree_list = tree_dict[num_of_number/analytical_part]
                        if analytical_part == 2:
                            if prob_type == "Partition" and (gcondition_1 in ["hexagon", "circle"]) and (gcondition_2 in [6, 8]):
                                analytical_mode = modes_1
                            elif prob_type == "Combination" and gcondition_1 == "hexagon" and gcondition_2 == "overlap":
                                analytical_mode = modes_1
                            elif (prob_type == "Combination" and gcondition_2 == "include") \
                                    or (prob_type == "Composition" and gcondition_2 == "cross"):
                                analytical_mode = modes_2
                            elif prob_type == "Combination" and gcondition_1 == "triangle" and gcondition_2 == "include":
                                analytical_mode = modes_3
                            elif prob_type == "Composition" and (gcondition_2 in ["circle", "square", "triangle"]):
                                analytical_mode = modes_4
                            else:
                                analytical_mode = modes_0
                            answer, trials_update = search_analytical_2(max_trials, trials, tree_list, number_list_1,
                                                                        number_list_2, number_list_3, center_number,
                                                                        analytical_mode, True)
                            trials = trials_update

                        elif analytical_part == 3:
                            if prob_type == "Combination" and gcondition_1 == "triangle" and gcondition_2 == "include":
                                analytical_mode = modes_4
                            elif prob_type == "Composition" and gcondition_2 == "triangle":
                                analytical_mode = modes_4
                            else:
                                analytical_mode = modes_0
                            answer, trials_update = search_analytical_3(max_trials, trials, tree_list, number_list_1,
                                                                        number_list_2, number_list_3, center_number,
                                                                        analytical_mode, True)
                            trials = trials_update

                        elif analytical_part == 4:
                            if prob_type == "Composition" and (gcondition_2 in ["circle", "square"]):
                                analytical_mode = modes_4
                            else:
                                analytical_mode = modes_0
                            answer, trials_update = search_analytical_4(max_trials, trials, tree_list, number_list_1,
                                                                        number_list_2, number_list_3, center_number,
                                                                        analytical_mode, True)
                            trials = trials_update
                        if answer != "none":
                            flag = True
                            break
                        if trials >= max_trials:
                            break
            if flag or trials >= max_trials:
                break

    elapsed = time.clock() - start
    # print "The solver used " + str(elapsed) + "s to get the answer"
    return elapsed, answer
