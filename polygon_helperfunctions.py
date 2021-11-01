import itertools
from helperfunctions import group_pairs_to_nested_list, rect_from_pre
from polygon_calc_wrapper import PolygonCalc


# maximum distance of words to be recognized as belonging to the same word in terms of letter height
MAX_WORD_DISTANCE_RATIO = 0.75


# group ocr detected letters that belong to the same word together
def group_words(filtered_res_tuples, max_word_distance_ratio=MAX_WORD_DISTANCE_RATIO, pos_start_index=0):

    L2 = []

    comb = itertools.combinations(filtered_res_tuples, 2)

    pc = PolygonCalc()

    for elem in comb:

        pre_p1 = tuple(elem[0][pos_start_index: pos_start_index + 4])
        pre_p2 = tuple(elem[1][pos_start_index: pos_start_index + 4])

        p1 = rect_from_pre(pre_p1)
        p2 = rect_from_pre(pre_p2)

        p1_height = pre_p1[3] - pre_p1[1]
        p2_height = pre_p2[3] - pre_p2[1]

        p_height = min(p1_height, p2_height)

        max_word_dist = max_word_distance_ratio * p_height

        min_dist = pc.min_poly_distance(p1, p2)

        if min_dist < max_word_dist:
            L2.append(elem)

    del pc

    for elem in filtered_res_tuples:
        L2.append((elem, elem))

    word_grouped_tuples = group_pairs_to_nested_list(L2)

    return word_grouped_tuples
