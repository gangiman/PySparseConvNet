import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter


def PR_per_class(P_points,labels=None, C=18):
    if labels is None:
        labels = map(str, range(len(P_points)))
    assert len(labels) == len(P_points)
    Recall=np.arange(1, C + 1, dtype=float) / C
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    plt.grid(True)
    plt.hold(True)
    ax.xthick = np.arange(0.0, 1.1, 0.1)
    reco = Recall
    for _idx, pres in enumerate(P_points):
        plt.plot(reco, pres, label=labels[_idx])
    plt.hold(False)
    plt.xlabel('Recall')
    plt.ylabel('Prescion')
    plt.legend()

# @interact(**{__name: Checkbox(description=__name, value=True) for __name in strings_with_arrays})
# def learning_curve_for_starting_point(**kwargs):
#     fig, ax = plt.subplots(1, 1, figsize=(11,7))
#     plt.grid(True)
#     plt.hold(True)
#     # ax.xthick = np.arange(0.0, 1.1, 0.1)
#     for _name, (reco, pres) in strings_with_arrays.items():
#         if kwargs[_name]:
#             plt.plot(reco, pres, label=_name)
#     plt.hold(False)
#     plt.xlabel('Recall')
#     plt.ylabel('Prescion')
#     plt.legend()


def plot_class_distribution(freq_dict):
    pairs = freq_dict.items()
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    N = len(pairs)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    ax.bar(ind, map(itemgetter(1), pairs), width)
    # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    # ax.set_ylim(0,45)
    ax.set_ylabel('number of samples')
    ax.set_title('number of samples by class')
    # xTickMarks = [fashionista_tags[int(i)] for _, i in sorted_pairs]
    xTickMarks = map(itemgetter(0), pairs)
    ax.set_xticks(ind + width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=90, fontsize=10)
    # plt.show()
    return ax
