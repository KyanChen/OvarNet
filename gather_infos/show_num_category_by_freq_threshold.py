import numpy as np
import matplotlib.pyplot as plt
import json


def show_num_by_freq_threshold(in_path):
    categories = json.load(open(in_path, 'r'))['categories']
    categories_freq = [x[1] for x in categories]
    categories_freq = np.array(categories_freq)
    max_value = np.max(categories_freq) * 0.00005
    # xs = np.linspace(50, max_value, num=1000)  # the label locations
    xs = np.linspace(1e4, 1e6, num=1000)  # the label locations

    ys = []
    for x in xs:
        num_categories = np.sum(categories_freq > x)
        ys.append(num_categories)
    ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.plot(xs, ys, label='num categories', linewidth=2, color='red')
    plt.margins(x=0)
    ax.set_ylabel('num categories')
    ax.set_title('num categories by freq thres')
    # ax.set_xticks(x, labels, rotation=90)
    ax.legend()
    ax.grid()
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    # plt.show()
    plt.savefig("categories_between_1e4_1e6.png", bbox_inches='tight', pad_inches=0)
    # plt.savefig("categories_cut_scale_0_00005.png", bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    in_path = 'infos/all_gather_categories.json'
    show_num_by_freq_threshold(in_path)