# -*- coding: utf-8 -*-

import cv2
import math
import copy
import numpy as np
from PIL import Image
from AOT import Root
from const import (PANEL_SIZE, CENTER, CENTER_1_1, CENTER_1_2, DEFAULT_WIDTH, LENGTH_1, LENGTH_2, LENGTH_3)
from Num_Arrange import (disp_num, math_parser, whether_center)

'''
How to draw panels: According to the sampled parameters about the problem, the function for each problem ----
draw_combination(), draw_composition(), draw_partition() will draw the geometrical figures and return a list of
coordinates to place the numbers. Then display numbers and question mark on these given positions.
'''


def imshow(array):
    image = Image.fromarray(array)
    image.show()


def imsave(array, file_path):
    image = Image.fromarray(array)
    image.save(file_path)


def rendering_panels(img1, img2, img3, panel_size):
    image = 255 * np.ones((2*panel_size + 30, 2*panel_size + 30), np.uint8)
    image[0: panel_size, 0: panel_size] = img1
    image[0: panel_size, panel_size + 30: 2*panel_size + 30] = img2
    image[panel_size + 10: 2*panel_size + 10, int(0.5 * panel_size): int(1.5 * panel_size)] = img3
    return image


def drawing_panels(root):
    # Decompose the panel into layout(geometrical) and algebra(mathematical) parts
    assert isinstance(root, Root)
    prob_type, conditions = root.prepare()
    geom_conditions = [conditions[0], conditions[1]]
    math_conditions = [conditions[2], conditions[3]]
    # draw the geometrical figures for each type of problem
    font_size = 0.6
    if prob_type == "Combination":
        img, positions = draw_combination(geom_conditions)
    elif prob_type == "Composition":
        img, positions = draw_composition(geom_conditions)
        font_size = 0.5
    elif prob_type == "Partition":
        img, positions = draw_partition(geom_conditions)
    # Arrange and display the numbers on the panel
    show_center = whether_center(prob_type, geom_conditions)
    interpret, mode, part, operator_list, int_list_1, int_list_2, int_list_3, pos_list = math_parser(math_conditions,
                                                                                                     geom_conditions,
                                                                                                     prob_type,
                                                                                                     positions,
                                                                                                     show_center)
    img1, img2, img3 = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)
    answer_1, img_1, mark_1 = disp_num(img1, int_list_1, pos_list, False, font_size, show_center)
    answer_2, img_2, mark_2 = disp_num(img2, int_list_2, pos_list, False, font_size, show_center)
    answer_3, img_3, mark_3 = disp_num(img3, int_list_3, pos_list, True, font_size, show_center)
    prob_answer = answer_3
    prob_operator = operator_list
    prob_images = [img_1, img_2, img_3]
    return prob_answer, prob_operator, int_list_1, int_list_2, int_list_3, show_center, interpret, mode, part, mark_3, prob_images


def draw_combination(geom_conditions):
    img = 255 * np.ones((PANEL_SIZE, PANEL_SIZE), np.uint8)
    position = []
    condition_1 = geom_conditions[0]
    condition_2 = geom_conditions[1]
    geom_type = condition_1.type.get_value()
    geom_relation = condition_2.grelation.get_value()
    if geom_relation == "overlap":
        posit = draw_overlap(img, geom_type)
        position.extend(posit)
    elif geom_relation == "include":
        posit = draw_include(img, geom_type)
        position.extend(posit)
    elif geom_relation == "tangent":
        posit = draw_tangent(img, geom_type)
        position.extend(posit)
    return img, position


def draw_composition(geom_conditions):
    img = 255 * np.ones((PANEL_SIZE, PANEL_SIZE), np.uint8)
    position = []
    condition_1 = geom_conditions[0]
    condition_2 = geom_conditions[1]
    geom_type = condition_1.type.get_value()
    geom_format = condition_2.format.get_value()
    (x, y) = CENTER
    L = LENGTH_2 * PANEL_SIZE + 2
    L2 = 2 * LENGTH_2 * PANEL_SIZE + 4
    if geom_format == "line":
        posit = line_pos((x, y), L, L2)
    elif geom_format == "cross":
        posit = cross_pos((x, y), L, L2)
    elif geom_format == "triangle":
        posit = triangle_pos((x, y), L, L2)
    if geom_format == "square":
        posit = square_pos((x, y), L, L2)
    if geom_format == "circle":
        posit = circle_pos((x, y), L, L2)

    position.extend(posit)
    if geom_type == "triangle":
        for c in posit:
            draw_triangle(img, c, L2, False, DEFAULT_WIDTH)
    elif geom_type == "square":
        for c in posit:
            draw_rectangle(img, c, L2, 0, DEFAULT_WIDTH)
    elif geom_type == "circle":
        for c in posit:
            draw_circle(img, c, L, DEFAULT_WIDTH)
    return img, position


def draw_partition(geom_conditions):
    img = 255 * np.ones((PANEL_SIZE, PANEL_SIZE), np.uint8)
    position = []
    condition_1 = geom_conditions[0]
    condition_2 = geom_conditions[1]
    geom_type = condition_1.type.get_value()
    geom_part = condition_2.part.get_value()
    # It is not proper for hexagons to be partitioned into 4 or 8 parts, so the sampled value 4 and 8 will be changed.
    # The same is true with squares.
    if geom_type == "hexagon":
        if geom_part == 4:
            condition_2.part.set_value_level(0)
            geom_part = condition_2.part.get_value()
        elif geom_part == 8:
            condition_2.part.set_value_level(2)
            geom_part = condition_2.part.get_value()
    if geom_type == "square":
        if geom_part == 2:
            condition_2.part.set_value_level(1)
            geom_part = condition_2.part.get_value()
        elif geom_part == 6:
            condition_2.part.set_value_level(3)
            geom_part = condition_2.part.get_value()
    L = LENGTH_3 * PANEL_SIZE
    L2 = 2 * LENGTH_3 * PANEL_SIZE
    if geom_type == "square":
        position = partition_square(img, CENTER, L2, geom_part, DEFAULT_WIDTH)
    elif geom_type == "circle":
        position = partition_circle(img, CENTER, L, geom_part, DEFAULT_WIDTH)
    elif geom_type == "hexagon":
        position = partition_hexagon(img, CENTER, L, geom_part, DEFAULT_WIDTH)
    return img, position


'''
Below are more detailed functions used to draw panels.
1) Functions draw_overlap(), draw_include(), draw_tangent() draw the panels with different geometrical relations in 
combination problems. 
2) Functions line_pos(), cross_pos(), triangle_pos() and so on compute the coordinates of geometrical shapes in 
different arrangement formats of composition problems. 
3) Functions  partition_square(),  partition_hexagon(), partition_circle() cut certain geometrical shapes into certain 
parts. Function partition_pos() computes the coordinates of integers in partitioned shapes.
4) Functions draw_triangle(), draw_circle(), draw_hexagon(), draw_rectangle() draw basic geometrical shapes by lines, 
and return the coordinates that can be used to place integers.
'''


def draw_overlap(img, geom_type):
    if geom_type == "triangle": # An upright triangle is overlapped with an inverse triangle.
        pos_1, pos_2, pos_3 = draw_triangle(img, CENTER, LENGTH_1 * 2 * PANEL_SIZE, False, DEFAULT_WIDTH)
        pos_4, pos_5, pos_6 = draw_triangle(img, CENTER, LENGTH_1 * 2 * PANEL_SIZE, True, DEFAULT_WIDTH)
        posit = [pos_4, pos_1, pos_5, pos_2, pos_6, pos_3]
    if geom_type == "square":
        pos_1, pos_2, pos_3, pos_4 = draw_rectangle(img, CENTER_1_1, LENGTH_1 * 2 * PANEL_SIZE, 0, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_rectangle(img, CENTER_1_2, LENGTH_1 * 2 * PANEL_SIZE, 0, DEFAULT_WIDTH)
        posit = [pos_5, pos_3]
    if geom_type == "rectangle": # Two rectangles with different orientations are overlapped.
        pos_1, pos_2, pos_3, pos_4 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE, 1, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE, 2, DEFAULT_WIDTH)
        posit = [pos_5, pos_2, pos_7, pos_4]
    if geom_type == "circle":
        pos_1, pos_2, pos_3, pos_4 = draw_circle(img, CENTER_1_1, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_circle(img, CENTER_1_2, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        posit = [pos_5, pos_3]
    if geom_type == "hexagon":
        pos_1, pos_2, pos_3, pos_4 = draw_hexagon(img, CENTER_1_1, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_hexagon(img, CENTER_1_2, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        posit = [pos_5, pos_2, pos_3, pos_8]
    return posit


def draw_include(img, geom_type):
    if geom_type == "triangle":
        pos_1, pos_2, pos_3 = draw_triangle(img, CENTER, LENGTH_1 * 2 * PANEL_SIZE, False, DEFAULT_WIDTH)
        pos_4, pos_5, pos_6 = draw_triangle(img, CENTER, LENGTH_1 * PANEL_SIZE, False, DEFAULT_WIDTH)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6]
    if geom_type == "square":
        pos_1, pos_2, pos_3, pos_4 = draw_rectangle(img, CENTER, LENGTH_1 * 2 * PANEL_SIZE, 0, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE, 0, DEFAULT_WIDTH)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    if geom_type == "rectangle":
        pos_1, pos_2, pos_3, pos_4 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE, 1, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE / 2, 1, DEFAULT_WIDTH)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    if geom_type == "circle":
        pos_1, pos_2, pos_3, pos_4 = draw_circle(img, CENTER, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_circle(img, CENTER, LENGTH_1 * PANEL_SIZE / 2, DEFAULT_WIDTH)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    if geom_type == "hexagon":
        pos_1, pos_2, pos_3, pos_4 = draw_hexagon(img, CENTER, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_hexagon(img, CENTER, LENGTH_1 * PANEL_SIZE / 2, DEFAULT_WIDTH)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    return posit


def draw_tangent(img, geom_type):
    (x, y) = CENTER
    L = LENGTH_1 * PANEL_SIZE
    L2 = 2*L/3
    if geom_type == "triangle":
        c1 = (int(x - L/2), int(y + math.sqrt(3)*L/6))
        c2 = (int(x), int(y - math.sqrt(3)*L/3))
        c3 = (int(x + L/2), int(y + math.sqrt(3)*L/6))
        draw_triangle(img, c1, L, True, DEFAULT_WIDTH)
        draw_triangle(img, c2, L, True, DEFAULT_WIDTH)
        draw_triangle(img, c3, L, True, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    if geom_type == "square":
        c1 = (int(x - L/2), int(y - L/2))
        c2 = (int(x + L/2), int(y - L/2))
        c3 = (int(x), int(y + L/2))
        draw_rectangle(img, c1, L, 0, DEFAULT_WIDTH)
        draw_rectangle(img, c2, L, 0, DEFAULT_WIDTH)
        draw_rectangle(img, c3, L, 0, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    if geom_type == "rectangle":
        c1 = (int(x - L2), int(y - L2/2))
        c2 = (int(x + L2), int(y - L2/2))
        c3 = (int(x), int(y + L2/2))
        draw_rectangle(img, c1, L2, 1, DEFAULT_WIDTH)
        draw_rectangle(img, c2, L2, 1, DEFAULT_WIDTH)
        draw_rectangle(img, c3, L2, 1, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    if geom_type == "circle":
        c1 = (int(x - L2), int(y + math.sqrt(3)*L2/3))
        c2 = (int(x), int(y - 2*math.sqrt(3)*L2/3))
        c3 = (int(x + L2), int(y + math.sqrt(3)*L2/3))
        draw_circle(img, c1, L2, DEFAULT_WIDTH)
        draw_circle(img, c2, L2, DEFAULT_WIDTH)
        draw_circle(img, c3, L2, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    if geom_type == "hexagon":
        c1 = (int(x - L), int(y))
        c2 = (int(x + L/2), int(y - math.sqrt(3)*L/2))
        c3 = (int(x + L/2), int(y + math.sqrt(3)*L/2))
        draw_hexagon(img, c1, L, DEFAULT_WIDTH)
        draw_hexagon(img, c2, L, DEFAULT_WIDTH)
        draw_hexagon(img, c3, L, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    return posit


def line_pos((x, y), L, L2):
    c1 = (int(x), int(y - 3 * L2 - L))
    c2 = (int(x), int(y + 3 * L2 + L))
    c3 = (int(x), int(y - L2 - L))
    c4 = (int(x), int(y + L2 + L))
    posit = [c1, c2, c3, c4]
    return posit


def cross_pos((x, y), L, L2):
    c1 = (int(x - 3 * L2 - L), y)
    c2 = (x, int(y - 3 * L2 - L))
    c3 = (int(x + 3 * L2 + L), y)
    c4 = (x, int(y + 3 * L2 + L))
    c5 = (int(x - L2 - L), y)
    c6 = (x, int(y - L2 - L))
    c7 = (int(x + L2 + L), y)
    c8 = (x, int(y + L2 + L))
    posit = [c1, c2, c3, c4, c5, c6, c7, c8]
    return posit


def triangle_pos((x, y), L, L2):
    c1 = (int(x - 2 * L2), int(y + 2 * math.sqrt(3) * L2 / 3))
    c2 = (int(x - L2), int(y - math.sqrt(3) * L2 / 3))
    c3 = (int(x), int(y - 4 * math.sqrt(3) * L2 / 3))
    c4 = (int(x + L2), int(y - math.sqrt(3) * L2 / 3))
    c5 = (int(x + 2 * L2), int(y + 2 * math.sqrt(3) * L2 / 3))
    c6 = (int(x), int(y + 2 * math.sqrt(3) * L2 / 3))
    posit = [c1, c2, c3, c4, c5, c6]
    return posit


def square_pos((x, y), L, L2):
    c1 = (int(x - 2 * L2), int(y - 2 * L2))
    c2 = (x, int(y - 2 * L2))
    c3 = (int(x + 2 * L2), int(y - 2 * L2))
    c4 = (int(x + 2 * L2), y)
    c5 = (int(x + 2 * L2), int(y + 2 * L2))
    c6 = (x, int(y + 2 * L2))
    c7 = (int(x - 2 * L2), int(y + 2 * L2))
    c8 = (int(x - 2 * L2), y)
    posit = [c1, c2, c3, c4, c5, c6, c7, c8]
    return posit


def circle_pos((x, y), L, L2):
    c1 = (int(x - 2 * math.sqrt(2) * L2), int(y))
    c2 = (int(x - 2 * L2), int(y - 2 * L2))
    c3 = (int(x), int(y - 2 * math.sqrt(2) * L2))
    c4 = (int(x + 2 * L2), int(y - 2 * L2))
    c5 = (int(x + 2 * math.sqrt(2) * L2), int(y))
    c6 = (int(x + 2 * L2), int(y + 2 * L2))
    c7 = (int(x), int(y + 2 * math.sqrt(2) * L2))
    c8 = (int(x - 2 * L2), int(y + 2 * L2))
    posit = [c1, c2, c3, c4, c5, c6, c7, c8]
    return posit


def partition_square(img, (x, y), l, part, thickness):
    draw_rectangle(img, (x, y), l, 0, thickness)
    p1 = (int(x - l/2), int(y - l/2))
    p2 = (int(x + l/2), int(y + l/2))
    p3 = (int(x - l/2), int(y + l/2))
    p4 = (int(x + l/2), int(y - l/2))
    p5 = (int(x - l/2), int(y))
    p6 = (int(x + l/2), int(y))
    p7 = (int(x), int(y - l/2))
    p8 = (int(x), int(y + l/2))
    if part == 2 or part == 4:
        # if sampled value of part is 4 or 2, partition the square into 4 parts
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        posit = partition_pos(4, (x, y), math.sqrt(2)*l/4)
    if part == 6 or part == 8:
        # if sampled value of part is 8 or 6, partition the square into 8 parts
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        cv2.line(img, p7, p8, (0, 0, 0), thickness)
        posit = partition_pos(8, (x, y), math.sqrt(2)*l/4)
    return posit


def partition_hexagon(img, (x, y), l, part, thickness):
    draw_hexagon(img, (x, y), l, thickness)
    p1 = (int(x - l), int(y))
    p2 = (int(x + l), int(y))
    p3 = (int(x - l/2), int(y - math.sqrt(3)*l/2))
    p4 = (int(x + l/2), int(y + math.sqrt(3)*l/2))
    p5 = (int(x - l/2), int(y + math.sqrt(3)*l/2))
    p6 = (int(x + l/2), int(y - math.sqrt(3)*l/2))
    if part == 2 or part == 4:
        # if sampled value of part is 4 or 2, partition the hexagon into 2 parts
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        posit = partition_pos(2, (x, y), l/2)
    if part == 6 or part == 8:
        # if sampled value of part is 8 or 6, partition the hexagon into 6 parts
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        posit = partition_pos(6, (x, y), l/2)
    return posit


def partition_circle(img, (x, y), l, part, thickness):
    draw_circle(img, (x, y), l, thickness)
    p1 = (int(x - math.sqrt(2)*l/2), int(y - math.sqrt(2)*l/2))
    p2 = (int(x + math.sqrt(2)*l/2), int(y + math.sqrt(2)*l/2))
    p3 = (int(x - math.sqrt(2)*l/2), int(y + math.sqrt(2)*l/2))
    p4 = (int(x + math.sqrt(2)*l/2), int(y - math.sqrt(2)*l/2))
    p5 = (int(x - l), int(y))
    p6 = (int(x + l), int(y))
    p7 = (int(x), int(y - l))
    p8 = (int(x), int(y + l))
    p9 = (int(x - l/2), int(y - math.sqrt(3)*l/2))
    p10 = (int(x + l/2), int(y + math.sqrt(3)*l/2))
    p11 = (int(x - l/2), int(y + math.sqrt(3)*l/2))
    p12 = (int(x + l/2), int(y - math.sqrt(3)*l/2))
    if part == 2:
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        posit = partition_pos(2, (x, y), l/2)
    if part == 4:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        posit = partition_pos(4, (x, y), l/2)
    if part == 6:
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        cv2.line(img, p9, p10, (0, 0, 0), thickness)
        cv2.line(img, p11, p12, (0, 0, 0), thickness)
        posit = partition_pos(6, (x, y), l/2)
    if part == 8:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        cv2.line(img, p7, p8, (0, 0, 0), thickness)
        posit = partition_pos(8, (x, y), l/2)
    return posit


def partition_pos(part, (x, y), r):
    # Compute the integer positions after partition.
    # These positions are on a circle with te same center as the partitioned geometrical shape.
    posit = []
    if part == 2:
        pos_1 = (int(x), int(y - r))
        pos_2 = (int(x), int(y + r))
        posit = [pos_1, pos_2]
    elif part == 4:
        pos_1 = (int(x - r), int(y))
        pos_2 = (int(x), int(y - r))
        pos_3 = (int(x + r), int(y))
        pos_4 = (int(x), int(y + r))
        posit = [pos_1, pos_2, pos_3, pos_4]
    elif part == 6:
        pos_1 = (int(x - math.sqrt(3)*r/2), int(y - r/2))
        pos_2 = (int(x), int(y - r))
        pos_3 = (int(x + math.sqrt(3)*r/2), int(y - r/2))
        pos_4 = (int(x + math.sqrt(3)*r/2), int(y + r/2))
        pos_5 = (int(x), int(y + r))
        pos_6 = (int(x - math.sqrt(3)*r/2), int(y + r/2))
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6]
    elif part == 8:
        pos_1 = (int(x - math.sqrt(3)*r/2), int(y - r/2))
        pos_2 = (int(x - r/2), int(y - math.sqrt(3)*r/2))
        pos_3 = (int(x + r/2), int(y - math.sqrt(3)*r/2))
        pos_4 = (int(x + math.sqrt(3)*r/2), int(y - r/2))
        pos_5 = (int(x + math.sqrt(3)*r/2), int(y + r/2))
        pos_6 = (int(x + r/2), int(y + math.sqrt(3)*r/2))
        pos_7 = (int(x - r/2), int(y + math.sqrt(3)*r/2))
        pos_8 = (int(x - math.sqrt(3)*r/2), int(y + r/2))
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    return posit


def draw_triangle(img, (x, y), l, inverse, thickness):
    # Parameter "inverse" indicates the orientation of triangle.
    # Draw upward triangle if parameter "inverse" is False, and draw downward triangle if parameter "inverse" is True.
    if inverse:
        p1 = (int(x - l/2), int(y - math.sqrt(3)*l/6))
        p2 = (int(x + l/2), int(y - math.sqrt(3)*l/6))
        p3 = (int(x), int(y + math.sqrt(3)*l/3))
        pos_1 = (int(x - 3*l/8), int(y + math.sqrt(3)*l/8))
        pos_2 = (int(x), int(y - math.sqrt(3)*l/4))
        pos_3 = (int(x + 3*l/8), int(y + math.sqrt(3)*l/8))
    else:
        p1 = (int(x - l/2), int(y + math.sqrt(3)*l/6))
        p2 = (int(x + l/2), int(y + math.sqrt(3)*l/6))
        p3 = (int(x), int(y - math.sqrt(3)*l/3))
        pos_1 = (int(x - 3*l/8), int(y - math.sqrt(3)*l/8))
        pos_2 = (int(x + 3*l/8), int(y - math.sqrt(3)*l/8))
        pos_3 = (int(x), int(y + math.sqrt(3)*l/4))
    cv2.line(img, p1, p2, (0, 0, 0), thickness)
    cv2.line(img, p2, p3, (0, 0, 0), thickness)
    cv2.line(img, p1, p3, (0, 0, 0), thickness)
    return pos_1, pos_2, pos_3


def draw_rectangle(img, (x, y), l, shape, thickness):
    # Parameter "shape" == 0 indicates a square
    # Parameter "shape" == 1 or 2 indicates the orientation of rectangle
    if shape == 0:
        p1 = (int(x - l/2), int(y - l/2))
        p2 = (int(x + l/2), int(y + l/2))
        pos_1 = (int(x - 3*l/4), int(y))
        pos_2 = (int(x), int(y - 3*l/4))
        pos_3 = (int(x + 3*l/4), int(y))
        pos_4 = (int(x), int(y + 3*l/4))
    elif shape == 1:
        p1 = (int(x - l), int(y - l/2))
        p2 = (int(x + l), int(y + l/2))
        pos_1 = (int(x - 5*l/4), int(y))
        pos_2 = (int(x), int(y - 3*l/4))
        pos_3 = (int(x + 5*l/4), int(y))
        pos_4 = (int(x), int(y + 3*l/4))
    elif shape == 2:
        p1 = (int(x - l/2), int(y - l))
        p2 = (int(x + l/2), int(y + l))
        pos_1 = (int(x - 3*l/4), int(y))
        pos_2 = (int(x), int(y - 5*l/4))
        pos_3 = (int(x + 3*l/4), int(y))
        pos_4 = (int(x), int(y + 5*l/4))
    cv2.rectangle(img, p1, p2, (0, 0, 0), thickness)
    return pos_1, pos_2, pos_3, pos_4


def draw_circle(img, (x, y), l, thickness):
    center = (int(x), int(y))
    pos_1 = (int(x - 3*l/2), int(y))
    pos_2 = (int(x), int(y - 3*l/2))
    pos_3 = (int(x + 3*l/2), int(y))
    pos_4 = (int(x), int(y + 3*l/2))
    cv2.circle(img, center, int(l), (0, 0, 0), thickness)
    return pos_1, pos_2, pos_3, pos_4


def draw_hexagon(img, (x, y), l, thickness):
    p1 = (int(x - l/2), int(y - math.sqrt(3)*l/2))
    p2 = (int(x + l/2), int(y - math.sqrt(3)*l/2))
    p3 = (int(x + l), int(y))
    p4 = (int(x + l/2), int(y + math.sqrt(3)*l/2))
    p5 = (int(x - l/2), int(y + math.sqrt(3)*l/2))
    p6 = (int(x - l), int(y))
    pos_1 = (int(x - 9*l/8), int(y - 3*math.sqrt(3)*l/8))
    pos_2 = (int(x + 9*l/8), int(y - 3*math.sqrt(3)*l/8))
    pos_3 = (int(x + 9*l/8), int(y + 3*math.sqrt(3)*l/8))
    pos_4 = (int(x - 9*l/8), int(y + 3*math.sqrt(3)*l/8))
    cv2.line(img, p1, p2, (0, 0, 0), thickness)
    cv2.line(img, p2, p3, (0, 0, 0), thickness)
    cv2.line(img, p3, p4, (0, 0, 0), thickness)
    cv2.line(img, p4, p5, (0, 0, 0), thickness)
    cv2.line(img, p5, p6, (0, 0, 0), thickness)
    cv2.line(img, p6, p1, (0, 0, 0), thickness)
    return pos_1, pos_2, pos_3, pos_4

