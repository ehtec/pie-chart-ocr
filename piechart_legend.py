from matplotlib import pyplot as plt
import numpy as np
from random import randint
from nltk.corpus import words
import uuid

plt.title(label="Society Food Preference",
          loc="left",
          fontstyle='italic')


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
        plt.savefig(filename[:6] + "_legend.png")
        plt.close()
    else:
        ax.pie(data, startangle=90, autopct='%1.2f%%', pctdistance=1.5)
        plt.tight_layout()
        plt.savefig("generated_pie_charts/generated_pie_charts_legend"+filename[:6] + ".png")
        plt.close()


for _ in range(10):
    word_list = words.words()
    labels_1 = [word_list[randint(1, 100000)] for i in range(5)]
    numbers = 5
    sum_number = 100
    areas = numbers_sum(numbers, sum_number)
    data_label = [f'{label}, {size:0.1f}%' for label, size in zip(labels_1, areas)]
    pie_chart_generator_legend(labels=data_label, data=areas)
