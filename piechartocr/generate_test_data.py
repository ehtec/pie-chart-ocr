import random
from matplotlib import pyplot as plt
import numpy as np
from nltk.corpus import words
# import uuid
from .helperfunctions import clean_folder_contents, get_root_path
import os
from . import data_helpers


# minimum word count in a caption
MIN_WORD_COUNT = 1

# maximum word count in a caption
MAX_WORD_COUNT = 5

# minimum sector count
MIN_SECTOR_COUNT = 3

# maximum sector count
MAX_SECTOR_COUNT = 8

# minimum sector percentage
MIN_SECTOR_PERCENTAGE = 4.0

# total count of charts generated (for each variant)
TOTAL_CHARTS_COUNT = 500

# data root where results are stored
DATA_ROOT = os.path.join(get_root_path(), 'data')


# add linebreaks to words that will be used as pie chart captions
def linebreak_words(labels):

    if len(labels) < 3:
        return ' '.join(labels)

    elif len(labels) < 5:
        return ' '.join(labels[:2]) + '\n' + ' '.join(labels[2:])

    elif len(labels) < 7:
        return ' '.join(labels[:3]) + '\n' + ' '.join(labels[3:])

    return ValueError("len(labels) is too large")


# split total_value into count different values, each of them is at least min_size
def split_number(count, total_value, min_size=0.0):

    if total_value < count * min_size:
        raise ValueError("total_value < count * min_size")

    variable_total_size = total_value - count * min_size

    sizes = np.random.dirichlet(np.ones(count)) * variable_total_size

    res_sizes = [size + min_size for size in sizes]

    return res_sizes


# generate pie chart from provided data
def pie_chart_generator(labels, data, chart_index, legend=True):

    if legend:
        fig = plt.figure(4, figsize=(12, 13))
        ax = fig.add_subplot(211)
    else:
        fig, ax = plt.subplots(figsize=(20, 13))

    plt.rcParams['font.size'] = 20.0
    # ax = fig.add_subplot(211)
    # ax.set_title('Random title')
    ax.axis("equal")
    # ax2 = fig.add_subplot(212)
    # ax2.axis("off")
    # filename = uuid.uuid4().hex

    foldername = f"Chart_{chart_index + 1}"

    # print("data: {0}".format(data))
    # print("labels: {0}".format(labels))

    annotations = [(label.replace('\n', ' '), value / 100) for label, value in zip(labels, data)]
    # print("annotations: {0}".format(annotations))

    if legend:
        folder_path = os.path.join(DATA_ROOT, 'generated_pie_charts_legend', foldername)
    else:
        folder_path = os.path.join(DATA_ROOT, 'generated_pie_charts_without_legend', foldername)

    if os.path.isdir(folder_path):
        clean_folder_contents(folder_path)
    else:
        os.mkdir(folder_path)

    csvpath = os.path.join(folder_path, 'annotation.csv')

    data_helpers.write_annotations_to_csv(csvpath, annotations)

    if legend:
        ax2 = fig.add_subplot(212)
        ax2.axis("off")
        pie = ax.pie(data, startangle=90, autopct='%1.1f%%', pctdistance=1.3)
        plt.legend(pie[0], labels, labelspacing=1.0, loc="upper left")
        plt.tight_layout()
        output_path = os.path.join(folder_path, 'image.png')
        plt.savefig(output_path)
        plt.close()

    else:
        ax.pie(data, startangle=90, autopct='%1.1f%%', labels=labels, pctdistance=0.7, labeldistance=1.5, radius=0.5)
        plt.tight_layout()
        output_path = os.path.join(folder_path, 'image.png')
        plt.savefig(output_path)
        plt.close()


# generate a random pie chart with only the number of sectors given
def generate_random_pie_chart(num_sectors, chart_index, legend=True):

    word_list = words.words()[100000:]
    all_label_list = []

    for i in range(num_sectors):
        label_list = [random.choice(word_list) for _ in range(random.randint(MIN_WORD_COUNT, MAX_WORD_COUNT))]
        label_string = linebreak_words(label_list)
        all_label_list.append(label_string)

    sum_number = 100.0
    areas = split_number(num_sectors, sum_number, MIN_SECTOR_PERCENTAGE)
    # data_label = [f'{label}' for label, size in zip(all_label_list, areas)]
    pie_chart_generator(labels=all_label_list, data=areas, chart_index=chart_index, legend=legend)


def main():

    path = os.path.join(DATA_ROOT, "generated_pie_charts_without_legend")
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        clean_folder_contents(path)

    path = os.path.join(DATA_ROOT, "generated_pie_charts_legend")
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        clean_folder_contents(path)

    for counter in range(TOTAL_CHARTS_COUNT):
        generate_random_pie_chart(random.randint(MIN_SECTOR_COUNT, MAX_SECTOR_COUNT), counter, legend=False)
        generate_random_pie_chart(random.randint(MIN_SECTOR_COUNT, MAX_SECTOR_COUNT), counter, legend=True)


if __name__ == "__main__":
    main()
