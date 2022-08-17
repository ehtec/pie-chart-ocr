import random
from matplotlib import pyplot as plt
import numpy as np
from nltk.corpus import words
import uuid
import os
import shutil
import logging

plt.title(label="Society Food Preference",
          loc="left",
          fontstyle='italic')


def clean_folder_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.exception(e)


def linebreak_words(labels):

    if len(labels) < 3:
        return ' '.join(labels)

    elif len(labels) < 5:
        return ' '.join(labels[:2]) + '\n' + ' '.join(labels[2:])

    elif len(labels) < 7:
        return ' '.join(labels[:3]) + '\n' + ' '.join(labels[3:])

    return ValueError("len(labels) is too large")


def split_number(count, total_value, min_size=0.0):

    if total_value < count * min_size:
        raise ValueError("total_value < count * min_size")

    variable_total_size = total_value - count * min_size

    sizes = np.random.dirichlet(np.ones(count)) * variable_total_size

    res_sizes = [size + min_size for size in sizes]

    return res_sizes


def pie_chart_generator_legend(labels, data, legend=True):
    fig = plt.figure(4, figsize=(10, 10))
    plt.rcParams['font.size'] = 18.0
    ax = fig.add_subplot(211)
    ax.set_title('Random title')
    ax.axis("equal")
    ax2 = fig.add_subplot(212)
    ax2.axis("off")
    filename = uuid.uuid4().hex
    if legend:
        pie = ax.pie(data, startangle=90)
        plt.legend(pie[0], labels, loc='upper left')
        plt.tight_layout()
        plt.savefig("generated_pie_charts/generated_pie_charts_legend/" + filename[:6] + "_legend.png")
        plt.close()
    else:
        ax.pie(data, startangle=90, autopct='%1.2f%%', pctdistance=1.5)
        plt.tight_layout()
        plt.savefig("generated_pie_charts/generated_pie_charts_legend/" + filename[:6] + ".png")
        plt.close()


def num_of_piecharts(num_sectors):
    word_list = words.words()[100000:]
    all_label_list = []
    for i in range(num_sectors):
        label_list = [random.choice(word_list) for j in range(random.randint(1, 5))]
        label_string = linebreak_words(label_list)
        all_label_list.append(label_string)

    sum_number = 100.0
    areas = split_number(num_sectors, sum_number, 4.0)
    data_label = [f'{label}, {size:0.1f}%' for label, size in zip(all_label_list, areas)]
    pie_chart_generator_legend(labels=data_label, data=areas)


# clean_folder_contents("generated_pie_charts/generated_pie_charts_without_legend/")
# clean_folder_contents("generated_pie_charts/generated_pie_charts_legend/")

def main():
    path = "generated_pie_charts/generated_pie_charts_legend/"
    clean_folder_contents(path)
    count = 15
    for counter in range(count):
        num_of_piecharts(random.randint(3, 8))


if __name__ == "__main__":
    main()
