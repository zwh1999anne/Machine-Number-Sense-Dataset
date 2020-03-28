# -*- coding: utf-8 -*-

import numpy as np
from tqdm import trange
from Solver import math_solve, geom_math_solve

keys = ["partition_problem", "combination_problem",  "composition_problem"]
max_trial = 10000  # The upper bound of searching steps, can be modified in experiments
num_of_problem = 10000

acc_1 = 0
acc_1_interpret = {"holistic": 0, "analytical": 0}
acc_1_integers = {2: 0, 3: 0, 4: 0, 6: 0, 8: 0}
acc_1_problems = {"Combination": 0, "Composition": 0, "Partition": 0}

acc_2 = 0
acc_2_interpret = {"holistic": 0, "analytical": 0}
acc_2_integers = {2: 0, 3: 0, 4: 0, 6: 0, 8: 0}
acc_2_problems = {"Combination": 0, "Composition": 0, "Partition": 0}

num_interpret = {"holistic": 0, "analytical": 0}
num_integers = {2: 0, 3: 0, 4: 0, 6: 0, 8: 0}
num_problems = {"Combination": 0, "Composition": 0, "Partition": 0}


overall_num = 0
for key in keys:
    pruned_types = ["circle"]
    if key == "combination_problem":
        geom_conditions = ["tangent", "overlap", "include"]
        for geom_condition in geom_conditions:
            for type in pruned_types:
                if geom_condition == "include":
                    interpretations = ["holistic", "analytical"]
                elif geom_condition == "overlap" and (type in ["rectangle", "triangle", "hexagon"]):
                    interpretations = ["holistic", "analytical"]
                else:
                    interpretations = ["holistic"]
                for interpretation in interpretations:
                    if interpretation == "analytical":
                        if geom_condition == "include":
                            if type == "triangle":
                                analytical_parts = [2, 3]
                            else:
                                analytical_parts = [2, 4]
                        elif geom_condition == "overlap":
                            if type == "triangle":
                                analytical_parts = [2, 3]
                            elif type in ["rectangle", "hexagon"]:
                                analytical_parts = [2]
                    else:
                        analytical_parts = [0]
                    for analytical_part in analytical_parts:
                        accurate_1 = 0
                        accurate_2 = 0
                        for n in trange(num_of_problem):
                            count = n % 10
                            if count == 3 or count == 5:
                                data = np.load("ProbSet/test_set/prob_{}_{}_{}_{}_{}_{}.npz".format(n, key, type, geom_condition, interpretation, analytical_part), allow_pickle=True)
                                prob_type = data["prob_type"]
                                geom_conditions = data["geometrial_conditions"]
                                center_num = data["center"]
                                int_list_1 = data["int_list_1"]
                                int_list_2 = data["int_list_2"]
                                int_list_3 = data["int_list_3"]
                                target = data["target"]
                                target = int(target)
                                time_1, predicted_1 = math_solve(max_trial, center_num, int_list_1, int_list_2, int_list_3, ["analytical", "holistic"], [2, 3, 4])
                                time_2, predicted_2 = geom_math_solve(max_trial, prob_type, geom_conditions, center_num, int_list_1, int_list_2, int_list_3, ["analytical", "holistic"], [2, 3, 4])
                                overall_num += 1
                                num_interpret[str(interpretation)] += 1
                                num_problems[str(prob_type)] += 1
                                num_integers[len(int_list_1)] += 1
                                if predicted_1 == target:
                                    acc_1 += 1
                                    accurate_1 += 1
                                    acc_1_interpret[str(interpretation)] += 1
                                    acc_1_problems[str(prob_type)] += 1
                                    acc_1_integers[len(int_list_1)] += 1
                                if predicted_2 == target:
                                    acc_2 += 1
                                    accurate_2 += 1
                                    acc_2_interpret[str(interpretation)] += 1
                                    acc_2_problems[str(prob_type)] += 1
                                    acc_2_integers[len(int_list_1)] += 1
                        # print "##############"
                        # print key, geom_condition, interpretation, analytical_part
                        # print float(accurate_1)/2000, float(accurate_2)/2000
                        # print "##############"

    elif key == "composition_problem":
        geom_conditions = ["line", "cross", "triangle", "square", "circle"]
        for geom_condition in geom_conditions:
            for type in pruned_types:
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
                        accurate_1 = 0
                        accurate_2 = 0
                        for n in trange(num_of_problem):
                            count = n % 10
                            if count == 3 or count == 5:
                                data = np.load("ProbSet/test_set/prob_{}_{}_{}_{}_{}_{}.npz".format(n, key, type, geom_condition, interpretation, analytical_part), allow_pickle=True)
                                prob_type = data["prob_type"]
                                geom_conditions = data["geometrial_conditions"]
                                center_num = data["center"]
                                int_list_1 = data["int_list_1"]
                                int_list_2 = data["int_list_2"]
                                int_list_3 = data["int_list_3"]
                                target = data["target"]
                                target = int(target)
                                time_1, predicted_1 = math_solve(max_trial, center_num, int_list_1, int_list_2, int_list_3, ["analytical", "holistic"], [2, 3, 4])
                                time_2, predicted_2 = geom_math_solve(max_trial, prob_type, geom_conditions, center_num, int_list_1, int_list_2, int_list_3, ["analytical", "holistic"], [2, 3, 4])
                                overall_num += 1
                                num_interpret[str(interpretation)] += 1
                                num_problems[str(prob_type)] += 1
                                num_integers[len(int_list_1)] += 1
                                if predicted_1 == target:
                                    acc_1 += 1
                                    accurate_1 += 1
                                    acc_1_interpret[str(interpretation)] += 1
                                    acc_1_problems[str(prob_type)] += 1
                                    acc_1_integers[len(int_list_1)] += 1
                                if predicted_2 == target:
                                    acc_2 += 1
                                    accurate_2 += 1
                                    acc_2_interpret[str(interpretation)] += 1
                                    acc_2_problems[str(prob_type)] += 1
                                    acc_2_integers[len(int_list_1)] += 1
                        # print "##############"
                        # print key, geom_condition, interpretation, analytical_part
                        # print float(accurate_1) / 2000, float(accurate_2) / 2000
                        # print "##############"

    elif key == "partition_problem":
        geom_conditions = [2, 4, 6, 8]
        for geom_condition in geom_conditions:
            for type in pruned_types:
                if geom_condition == 2:
                    interpretations = ["holistic"]
                else:
                    interpretations = ["holistic", "analytical"]
                for interpretation in interpretations:
                    if interpretation == "analytical":
                        if geom_condition == 6:
                            analytical_parts = [2, 3]
                        elif geom_condition == 4:
                            analytical_parts = [2]
                        elif geom_condition == 8:
                            analytical_parts = [2, 4]
                    else:
                        analytical_parts = [0]
                    for analytical_part in analytical_parts:
                        accurate_1 = 0
                        accurate_2 = 0
                        for n in trange(num_of_problem):
                            count = n % 10
                            if count == 3 or count == 5:
                                data = np.load("ProbSet/test_set/prob_{}_{}_{}_{}_{}_{}.npz".format(n, key, type, geom_condition, interpretation, analytical_part), allow_pickle=True)
                                prob_type = data["prob_type"]
                                geom_conditions = data["geometrial_conditions"]
                                center_num = data["center"]
                                int_list_1 = data["int_list_1"]
                                int_list_2 = data["int_list_2"]
                                int_list_3 = data["int_list_3"]
                                target = data["target"]
                                target = int(target)
                                time_1, predicted_1 = math_solve(max_trial, center_num, int_list_1, int_list_2, int_list_3, ["analytical", "holistic"], [2, 3, 4])
                                time_2, predicted_2 = geom_math_solve(max_trial, prob_type, geom_conditions, center_num, int_list_1, int_list_2, int_list_3, ["analytical", "holistic"], [2, 3, 4])
                                overall_num += 1
                                num_interpret[str(interpretation)] += 1
                                num_problems[str(prob_type)] += 1
                                num_integers[len(int_list_1)] += 1
                                if predicted_1 == target:
                                    acc_1 += 1
                                    accurate_1 += 1
                                    acc_1_interpret[str(interpretation)] += 1
                                    acc_1_problems[str(prob_type)] += 1
                                    acc_1_integers[len(int_list_1)] += 1
                                if predicted_2 == target:
                                    acc_2 += 1
                                    accurate_2 += 1
                                    acc_2_interpret[str(interpretation)] += 1
                                    acc_2_problems[str(prob_type)] += 1
                                    acc_2_integers[len(int_list_1)] += 1
                        # print "##############"
                        # print key, geom_condition, interpretation, analytical_part
                        # print float(accurate_1) / 2000, float(accurate_2) / 2000
                        # print "##############"

print "The accuracy of math_solver is ", float(acc_1)/overall_num
print "The accuracy of geom_math_solver is ", float(acc_2)/overall_num
print " "
print "math_solver's accuracy on holistic problems: ", float(acc_1_interpret["holistic"])/num_interpret["holistic"]
print "geom_math_solver's accuracy on holistic problems: ", float(acc_2_interpret["holistic"])/num_interpret["holistic"]
print "math_solver's accuracy on analytical problems: ", float(acc_1_interpret["analytical"])/num_interpret["analytical"]
print "geom_math_solver's accuracy on analytical problems: ", float(acc_2_interpret["analytical"])/num_interpret["analytical"]
print " "
print "math_solver's accuracy on combination problems: ", float(acc_1_problems["Combination"])/num_problems["Combination"]
print "geom_math_solver's accuracy on combination problems: ", float(acc_2_problems["Combination"])/num_problems["Combination"]
print "math_solver's accuracy on composition problems: ", float(acc_1_problems["Composition"])/num_problems["Composition"]
print "geom_math_solver's accuracy on composition problems: ", float(acc_2_problems["Composition"])/num_problems["Composition"]
print "math_solver's accuracy on partition problems: ", float(acc_1_problems["Partition"])/num_problems["Partition"]
print "geom_math_solver's accuracy on partition problems: ", float(acc_1_problems["Partition"])/num_problems["Partition"]
print " "
print "math_solver's accuracy on 2-integer problems: ", float(acc_1_integers[2])/num_integers[2]
print "geom_math_solver's accuracy on 2-integer problems: ", float(acc_2_integers[2])/num_integers[2]
print "math_solver's accuracy on 3-integer problems: ", float(acc_1_integers[3])/num_integers[3]
print "geom_math_solver's accuracy on 3-integer problems: ", float(acc_2_integers[3])/num_integers[3]
print "math_solver's accuracy on 4-integer problems: ", float(acc_1_integers[4])/num_integers[4]
print "geom_math_solver's accuracy on 4-integer problems: ", float(acc_2_integers[4])/num_integers[4]
print "math_solver's accuracy on 6-integer problems: ", float(acc_1_integers[6])/num_integers[6]
print "geom_math_solver's accuracy on 6-integer problems: ", float(acc_2_integers[6])/num_integers[6]
print "math_solver's accuracy on 8-integer problems: ", float(acc_1_integers[8])/num_integers[8]
print "geom_math_solver's accuracy on 8-integer problems: ", float(acc_2_integers[8])/num_integers[8]

