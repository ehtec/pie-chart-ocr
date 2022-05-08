from matplotlib import pyplot as plt
from random import randint
from nltk.corpus import words
import numpy as np
import uuid
from matplotlib import font_manager as fm


def fix_labels(mylabels, tooclose=0.1, sepfactor=2):
    vecs = np.zeros((len(mylabels), len(mylabels), 2))
    dists = np.zeros((len(mylabels), len(mylabels)))
    for i in range(0, len(mylabels) - 1):
        for j in range(i + 1, len(mylabels)):
            a = np.array(mylabels[i].get_position())
            b = np.array(mylabels[j].get_position())
            dists[i, j] = np.linalg.norm(a - b)
            vecs[i, j, :] = a - b
            if dists[i, j] < tooclose:
                mylabels[i].set_x(a[0] + sepfactor * vecs[i, j, 0])
                mylabels[i].set_y(a[1] + sepfactor * vecs[i, j, 1])
                mylabels[j].set_x(b[0] - sepfactor * vecs[i, j, 0])
                mylabels[j].set_y(b[1] - sepfactor * vecs[i, j, 1])


for _ in range(10):
    fig, ax1 = plt.subplots()
    ax1.axis('equal')
    n = 5
    m = 100
    sizes = np.random.dirichlet(np.ones(n)) * m
    print(sizes)
    word_list = words.words()
    groups = [word_list[randint(1, 100000)] for i in range(5)]
    print(groups)
    filename = uuid.uuid4().hex
    explode = [0.1, 0.1, 0.1, 0.1, 0.1]
    wedges, labels, autopct = ax1.pie(sizes, explode=explode, labels=groups, autopct='%1.1f%%',
                                      shadow=False, startangle=90)
    proptease = fm.FontProperties()
    proptease.set_size('xx-small')
    #plt.setp(labels, fontproperties=proptease)
    plt.setp(autopct, fontproperties=proptease)
    fix_labels(autopct, sepfactor=3)
    fix_labels(labels, sepfactor=2)
    plt.tight_layout()
    plt.savefig(filename[:6] + ".png")
    plt.close()
