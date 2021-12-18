import itertools
from helperfunctions import group_pairs_to_nested_list, rect_from_pre, grouper
from polygon_calc_wrapper import PolygonCalc
from tqdm import tqdm


# maximum distance of words to be recognized as belonging to the same word in terms of letter height
MAX_CHARACTER_DISTANCE_RATIO = 0.20

# maximum distance used for the grouper. Increasing it will remove possible false positives, making it smaller decreases
#   the execution time.
MAX_PRE_GROUPING_CHARACTER_DISTANCE_RATIO = 0.25


# group ocr detected letters that belong to the same word together
def group_words(filtered_res_tuples, max_word_distance_ratio=MAX_CHARACTER_DISTANCE_RATIO, pos_start_index=0):

    word_grouped_tuples = []

    filtered_res_tuples.sort(key=lambda x: x[1])

    pc = PolygonCalc()

    all_max_dist = MAX_PRE_GROUPING_CHARACTER_DISTANCE_RATIO * max([el[3] - el[1] for el in filtered_res_tuples])

    for group in grouper(filtered_res_tuples, interval=all_max_dist):

        L2 = []

        comb = itertools.combinations(group, 2)

        # pc = PolygonCalc()

        for elem in tqdm(comb):

            pre_p1 = tuple(elem[0][pos_start_index: pos_start_index + 4])
            pre_p2 = tuple(elem[1][pos_start_index: pos_start_index + 4])

            p1 = rect_from_pre(pre_p1)
            p2 = rect_from_pre(pre_p2)

            p1_height = pre_p1[3] - pre_p1[1]
            p2_height = pre_p2[3] - pre_p2[1]

            p_height = min(p1_height, p2_height)

            max_word_dist = max_word_distance_ratio * p_height

            # min_dist = pc.min_poly_distance(p1, p2)

            # min_dist = min([abs(pre_p1[2] - pre_p2[0]), abs(pre_p2[2] - pre_p1[0])])
            #
            # if min_dist > max_word_dist:
            #     continue

            # if pre_p1[3] - pre_p1[1] > 0:
            #     y1 = pre_p1[1]
            #     y2 = pre_p1[3]
            #
            # else:
            #     y1 = pre_p1[3]
            #     y2 = pre_p1[3]

            # if not any([y1 <= pre_p2[1] <= y2, y1 <= pre_p2[3] <= y2,
            #             all([y1 <= pre_p2[1], y1 <= pre_p2[3], y2 >= pre_p2[1], y2 >= pre_p2[3]])]):
            #     continue

            max_word_dist = max_word_distance_ratio * p_height
            #
            min_dist = pc.min_poly_distance(p1, p2)
            #
            # min_dist = min([abs(pre_p1[2] - pre_p2[0]), abs(pre_p2[2] - pre_p1[0])])
            #
            # if min_dist < max_word_dist:
            #     L2.append(elem)

            if min_dist > max_word_dist:
                continue

            L2.append(elem)

        # del pc

        for elem in group:
            L2.append((elem, elem))

        print("Executing group_pairs_to_nested_list...")

        temp_word_grouped_tuples = group_pairs_to_nested_list(L2)

        word_grouped_tuples += temp_word_grouped_tuples

    del pc

    return word_grouped_tuples
