import numpy as np
import copy


class CTNode(object):
    """
    This is the superclass for nodes in Calculator Tree
    Each node has two attributes ---- value and operator
    """

    def __init__(self, value, operator, lchild, rchild, parent):
        self.value = value
        self.operator = operator
        self.lchild = lchild
        self.rchild = rchild
        self.parent = parent
        self.level = 0
        self.order = 0
        self.visit = False

    def sample(self, min_level, max_level):
        self.value = np.random.choice(range(min_level, max_level + 1))

    def set_level(self, level):
        self.level = level

    def set_order(self, order):
        self.order = order

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def get_operator(self):
        return self.operator

    def set_operator(self, new_operator):
        self.operator = new_operator

    def get_neighbor(self):
        if self == self.parent.lchild:
            neighbor = self.parent.rchild
            direction = "right neighbor"
        else:
            neighbor = self.parent.lchild
            direction = "left neighbor"
        return neighbor, direction

    def visit_node(self):
        self.visit = True

    def is_leaf(self):
        leaf = False
        if self.rchild == "none" and self.lchild == "none":
            leaf = True
        return leaf


class NStack(object):
    """
    This is the superclass for the stack of numbers.
    """

    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()


# Build a tree representation to generate the calculation process from postfix expression
def build_calculator_tree(expression):
    number_stack = NStack()
    node_stack_1 = NStack()
    root_node = CTNode("none", "none", "none", "none", "none")
    node_list = []
    for item in expression:
        # if the item is an operator, pop the first two numbers out of the stack
        # set the value attribute of their parent node as the operation result of the two numbers
        if item in ["+", "-", "*", "/"]:
            num_2 = number_stack.pop()
            num_1 = number_stack.pop()
            operate_result = "(" + num_1 + item + num_2 + ")"
            number_stack.push(operate_result)
            right_child = node_stack_1.pop()
            left_child = node_stack_1.pop()
            root_node = CTNode(operate_result, item, left_child, right_child, "none")
            node_stack_1.push(root_node)
            right_child.parent = root_node
            left_child.parent = root_node
        # if the item is a number, push them into the number stack and node stack
        else:
            number_stack.push(item)
            number_node = CTNode(item, "none", "none", "none", "none")
            node_stack_1.push(number_node)

    node_list.append(root_node)
    current_level = [root_node]
    level = 1
    # rendering a list of all the tree nodes according to the depth
    while child_exist(current_level):
        next_level = []
        for node in current_level:
            if not node.is_leaf():
                next_level.append(node.lchild)
                next_level.append(node.rchild)
                node.lchild.set_level(level)
                node.rchild.set_level(level)
        node_list.extend(next_level)
        current_level = next_level
        level += 1
    return node_list, level


# Sample the integer value for each node in the calculator tree
def number_sampler(node_list, constant, levels):
    # The value of the root node is the predefined constant
    integer_list = []
    root_node = node_list[0]
    root_node.value = constant
    level_start = [0]
    for i in range(1, len(node_list)):
        node_1 = node_list[i]
        node_2 = node_list[i-1]
        if node_1.level != node_2.level:
            level_start.append(i)
    level_start.append(len(node_list))
    qualified = True

    # From the root-level nodes to the leaf-level nodes, sample the integer in the form of calculation pairs
    k = 1
    error = False
    error_num = 0
    while k < levels:
        pairs = (level_start[k + 1] - level_start[k])/2
        for i in range(pairs):
            node_1 = node_list[2*i + level_start[k]]
            node_2 = node_list[2*i + level_start[k] + 1]
            if node_1.parent.get_operator() == "+":
                result = node_1.parent.get_value()
                if result == 1:
                    error = True
                    break
                else:
                    node_1.sample(1, result-1)
                    value_1 = node_1.get_value()
                    value_2 = result - value_1
                    node_2.set_value(value_2)
            if node_1.parent.get_operator() == "-":
                result = node_1.parent.get_value()
                if result == 99:
                    error = True
                    break
                else:
                    node_1.sample(result + 1, 99)
                    value_1 = node_1.get_value()
                    value_2 = value_1 - result
                    node_2.set_value(value_2)
            if node_1.parent.get_operator() == "*":
                result = node_1.parent.get_value()
                node_1.sample(1, result)
                num_circulation = 0
                while result % node_1.get_value() != 0:
                    node_1.sample(1, result)
                    num_circulation += 1
                    if num_circulation > 100:
                        error = True
                        break
                if error:
                    break
                value_1 = node_1.get_value()
                value_2 = result / value_1
                node_2.set_value(value_2)
            if node_1.parent.get_operator() == "/":
                result = node_1.parent.get_value()
                node_1.sample(result, 99)
                num_circulation = 0
                while node_1.get_value() % result != 0:
                    node_1.sample(result, 99)
                    num_circulation += 1
                    if num_circulation > 100:
                        error = True
                        break
                if error:
                    break
                value_1 = node_1.get_value()
                value_2 = value_1 / result
                node_2.set_value(value_2)

        if error_num > 10:
            qualified = False
            break
        if error:
            error = False
            error_num += 1
            integer_list = []
            k = 1
        else:
            k += 1

    node_search_list = middle_search(root_node)
    for node in node_search_list:
        if node.is_leaf():
            integer_list.append(node.get_value())

    return node_search_list, integer_list, qualified


# Generate possible tree structures for a calculation with given amount of numbers(leaves)
def generate_tree(num_leaf):
    if num_leaf == 2:
        root_node = CTNode("none", "none", "none", "none", "none")
        node_1 = CTNode("none", "none", "none", "none", "none")
        node_2 = CTNode("none", "none", "none", "none", "none")
        node_1.parent = root_node
        node_2.parent = root_node
        node_1.set_level(1)
        node_2.set_level(1)
        root_node.lchild = node_1
        root_node.rchild = node_2
        tree_list = [root_node]
    else:
        previous_tree = generate_tree(num_leaf-1)  # generate tree structures recursively
        tree_list = []
        # To generate new structures, add two leaves to a certain leaf in previous structure
        # use the i-j relation to ensure the leaves are added to different leaves of a previous tree each time
        for tree_root in previous_tree:
            for i in range(num_leaf - 1):
                new_root = copy.deepcopy(tree_root)
                node_list = middle_search(new_root)
                flag = False
                j = 0
                for node in node_list:
                    if node.is_leaf():
                        j += 1
                        if j == i + 1:
                            new_node_1 = CTNode("none", "none", "none", "none", "none")
                            new_node_2 = CTNode("none", "none", "none", "none", "none")
                            new_node_1.parent = node
                            new_node_2.parent = node
                            new_node_1.set_level(node.level + 1)
                            new_node_2.set_level(node.level + 1)
                            node.lchild = new_node_1
                            node.rchild = new_node_2
                            flag = True
                            break
                if flag:
                    tree_list.append(new_root)

    return tree_list


# Prune the redundant generated tree structures
def prune_generate_tree(tree_list):
    order_list = []
    pruned_tree_list = []
    for root in tree_list:
        # use pre-order traversal to give each tree node a order tag
        queue = []
        node = root
        queue.append(node)
        i = 0
        while queue:
            node = queue.pop(0)
            node.set_order(i)
            i += 1
            if node.lchild != "none":
                queue.append(node.lchild)
            if node.rchild != "none":
                queue.append(node.rchild)
        # in-order traverse the tree and generate a ordered list of order tags for the traversed nodes
        stack = NStack()
        node_search_list = []
        node = root
        while (node != "none") or (not stack.is_empty()):
            while (node != "none"):
                stack.push(node)
                node = node.lchild
            node = stack.pop()
            node_search_list.append(node.order)
            node = node.rchild
        #  same tree structures will have same order lists, these redundant structures can be pruned
        if not(node_search_list in order_list):
            order_list.append(node_search_list)
            pruned_tree_list.append(root)

    return pruned_tree_list


# fill the intervals in the operator list with letters to render an expression
def fill_expression(operator_list):
    o_num = len(operator_list)
    n_num = 0
    expression = "a"
    for i in range(o_num):
        if operator_list[i] == "(":
            end_char = expression[-1]
            expression = expression[:-1]
            expression = expression + "(" + " " + end_char
        elif operator_list[i] == ")":
            expression = expression + " " + ")"
        else:
            expression = expression + " " + operator_list[i] + " " + chr(98 + n_num)
            n_num += 1
    return expression


# calculation rules: '*' and '/' come first
ops_rule = {
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2
}


# change the infix expression to postfix expression
def middle_to_after(s):
    expression = []
    ops = [] # The stack of operators
    ss = s.split(' ')
    for item in ss:
        if item in ['+', '-', '*', '/']:
            while len(ops) >= 0:
                if len(ops) == 0:
                    ops.append(item)
                    break
                op = ops.pop()
                if op == '(' or ops_rule[item] > ops_rule[op]:
                    ops.append(op)
                    ops.append(item)
                    break
                else:
                    expression.append(op)
        elif item == '(':
            ops.append(item)
        elif item == ')':
            while len(ops) > 0:
                op = ops.pop()
                if op == '(':
                    break
                else:
                    expression.append(op)
        else:
            expression.append(item)

    while len(ops) > 0:
        expression.append(ops.pop())

    return expression


# check whether any node in the list has children
def child_exist(node_list):
    exist = False
    for node in node_list:
        if node != "none":
            if node.lchild != "none" or node.rchild != "none":
                exist = True
    return exist


# In-order traversal of the nodes in calculator tree
def middle_search(root):
    stack = NStack()
    node_search_list = []
    node = root
    while (node != "none") or (not stack.is_empty()):
        while (node != "none"):
            stack.push(node)
            node = node.lchild
        node = stack.pop()
        node_search_list.append(node)
        node = node.rchild

    return node_search_list


# Post-order traversal of the nodes in calculator tree
def post_search(root):
    stack_1 = NStack()
    stack_2 = NStack()
    stack_1.push(root)
    while not stack_1.is_empty():
        node = stack_1.pop()
        if node.lchild != "none":
            stack_1.push(node.lchild)
        if node.rchild != "none":
            stack_1.push(node.rchild)
        stack_2.push(node)
    return stack_2






