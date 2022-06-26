import random
from matplotlib import pyplot as plt
import numpy as np
from random import randint
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


def numbers_sum(n, m):
    sizes = np.random.dirichlet(np.ones(n)) * m
    for size in sizes:
        if size < 3.0:
            sizes = numbers_sum(n, m)
    return sizes


def pie_chart_generator_legend(labels, data, legend=True):
    fig = plt.figure(4, figsize=(10, 10))
    plt.rcParams['font.size'] = 18.0
    ax = fig.add_subplot(211)
    ax.set_title("Piechart", loc="left", fontstyle='italic')
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


def num_of_piecharts(number_of_labels):
    word_list = words.words()
    labels_1 = [word_list[randint(1, 100000)] for _ in range(number_of_labels)]
    numbers = number_of_labels
    sum_number = 100
    areas = numbers_sum(numbers, sum_number)
    data_label = [f'{label}, {size:0.1f}%' for label, size in zip(labels_1, areas)]
    pie_chart_generator_legend(labels=data_label, data=areas)


# clean_folder_contents("generated_pie_charts/generated_pie_charts_without_legend/")
# clean_folder_contents("generated_pie_charts/generated_pie_charts_legend/")

path = "generated_pie_charts/generated_pie_charts_legend/"
dir_list = os.listdir(path)
if len(dir_list) != 0:
    clean_folder_contents(path)
num_labels = random.randint(1, 9)
count = 15
logging.info(num_labels)
for i in range(count):
    num_of_piecharts(num_labels)
