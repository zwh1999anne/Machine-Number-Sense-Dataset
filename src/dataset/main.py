# -*- coding: utf-8 -*-
import numpy as np
from tqdm import trange
from const import PANEL_SIZE
from Trees import (combination_prob, composition_prob, partition_prob, prune_problem_tree)
from Drawing import (imshow, imsave, drawing_panels, rendering_panels)
from Num_Arrange import reverse_arrange


# generate problems that pruned by given parameters
def generate_problem(problem_configs, num_of_problem, key, geom_type, geom_condition, interpretation, analytical_part, seperate):
    for n in trange(num_of_problem):
        count = n % 10
        if count in [0, 2, 4, 6, 8, 9]:
            set_name = "train_set"
        elif count in [1, 7]:
            set_name = "val_set"
        else:
            set_name = "test_set"

        root = problem_configs[key]
        new_root = root.sample()
        pruned_root = prune_problem_tree(new_root, geom_type, geom_condition, interpretation, analytical_part)
        answer, operator, list_1, list_2, list_3, center, interpret, mode, part, mark, images = drawing_panels(pruned_root)
        prob_type, conditions = pruned_root.prepare()
        geometry = [conditions[0], conditions[1]]
        mathematics = [conditions[2], conditions[3]]

        if prob_type == "Combination":
            gcondition1 = geometry[0].type.get_value()
            gcondition2 = geometry[1].grelation.get_value()
        elif prob_type == "Composition":
            gcondition1 = geometry[0].type.get_value()
            gcondition2 = geometry[1].format.get_value()
        elif prob_type == "Partition":
            gcondition1 = geometry[0].type.get_value()
            gcondition2 = geometry[1].part.get_value()
        geometrial_conditions = [gcondition1, gcondition2]

        math_operators = mathematics[0].operator.get_value()
        math_constant = mathematics[0].integer.get_value()
        interpret_style = mathematics[1].interpret.get_value()
        if interpret_style == "holistic":
            analytical = 0
        else:
            analytical = mathematics[1].analytical.get_value()
        mathematical_conditions = [math_constant, interpret_style, analytical]

        if center:
            center_num = [list_1[-1], list_2[-1], list_3[-1]]
            list_1 = list_1[:-1]
            list_2 = list_2[:-1]
            list_3 = list_3[:-1]
        else:
            center_num = ["none", "none", "none"]
        list_3[mark] = "mark"

        if mode in [2, 3, 4]:
            new_list_1, new_list_2, new_list_3 = reverse_arrange(list_1, list_2, list_3, mode, part)
        else:
            new_list_1, new_list_2, new_list_3 = list_1, list_2, list_3

        target = answer
        if seperate:
            np.savez("ProbSet/{}/{}/prob_{}_{}_{}_{}_{}".format(key, set_name, n, geom_type, geom_condition, interpretation, analytical_part),
                     image=images, center=center_num, int_list_1=new_list_1, int_list_2=new_list_2, int_list_3=new_list_3, target=target,
                     prob_type=prob_type, math_operators=math_operators, geometrial_conditions=geometrial_conditions, mathematical_conditions=mathematical_conditions)
        else:
            np.savez("ProbSet/{}/prob_{}_{}_{}_{}_{}_{}".format(set_name, n, key, geom_type, geom_condition, interpretation, analytical_part),
                     image=images, center=center_num, int_list_1=new_list_1, int_list_2=new_list_2, int_list_3=new_list_3, target=target,
                     prob_type=prob_type, math_operators=math_operators, geometrial_conditions=geometrial_conditions, mathematical_conditions=mathematical_conditions)

        # To Visualize the problem
        """
        problem_image = rendering_panels(images[0], images[1], images[2], PANEL_SIZE)
        imsave(problem_image, "ProbSet/{}/prob_{}_{}_{}_{}_{}_{}.jpg".format(set_name, n, key, geom_type, geom_condition, interpretation, analytical_part))
        """


# generate the dataset containing equal amount of problems on all possible conditions
def generation(problem_configs, num_of_problem):
    for key in problem_configs.keys():
        pruned_types = ["circle"]
        if key == "combination_problem":
            geom_conditions = ["overlap", "include", "tangent"]
            for geom_condition in geom_conditions:
                for pr_type in pruned_types:
                    if geom_condition == "include" or (geom_condition == "overlap" and pr_type in ["rectangle", "triangle", "hexagon"]):
                        interpretations = ["holistic", "analytical"]
                    else:
                        interpretations = ["holistic"]
                    for interpretation in interpretations:
                        if interpretation == "analytical":
                            if geom_condition == "include":
                                if pr_type == "triangle":
                                    analytical_parts = [2, 3]
                                else:
                                    analytical_parts = [2, 4]
                            elif geom_condition == "overlap":
                                if pr_type == "triangle":
                                    analytical_parts = [2, 3]
                                elif pr_type in ["rectangle", "hexagon"]:
                                    analytical_parts = [2]
                        else:
                            analytical_parts = [0]
                        for analytical_part in analytical_parts:
                            generate_problem(problem_configs, num_of_problem, key, pr_type, geom_condition, interpretation, analytical_part, False)

        elif key == "composition_problem":
            geom_conditions = ["line", "cross", "triangle", "square", "circle"]
            for geom_condition in geom_conditions:
                for pr_type in pruned_types:
                    interpretations = ["holistic", "analytical"]
                    for interpretation in interpretations:
                        if interpretation == "analytical":
                            if geom_condition == "triangle":
                                analytical_parts = [2, 3]
                            elif geom_condition == "line":
                                analytical_parts = [2]
                            else:
                                analytical_parts = [2, 4]
                        else:
                            analytical_parts = [0]
                        for analytical_part in analytical_parts:
                            generate_problem(problem_configs, num_of_problem, key, pr_type, geom_condition, interpretation, analytical_part, False)

        elif key == "partition_problem":
            geom_conditions = [2, 4, 6, 8]
            for geom_condition in geom_conditions:
                for pr_type in pruned_types:
                    if geom_condition == 2:
                        interpretations = ["holistic"]
                    else:
                        interpretations = ["holistic", "analytical"]
                    for interpretation in interpretations:
                        if interpretation == "analytical":
                            if geom_condition == 4:
                                analytical_parts = [2]
                            elif geom_condition == 6:
                                analytical_parts = [2, 3]
                            elif geom_condition == 8:
                                analytical_parts = [2, 4]
                        else:
                            analytical_parts = [0]
                        for analytical_part in analytical_parts:
                            generate_problem(problem_configs, num_of_problem, key, pr_type, geom_condition, interpretation, analytical_part, False)


def main():
    problems = {"composition_problem": composition_prob(),
                "combination_problem": combination_prob(),
                "partition_problem": partition_prob()}
    num_prob = 10000
    generation(problems, num_prob)


if __name__ == "__main__":
    main()