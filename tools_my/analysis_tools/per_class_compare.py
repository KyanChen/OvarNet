import numpy as np
import matplotlib.pyplot as plt

pre_fix = '/Users/kyanchen/Code/CLIP_Prompt/'
pred = pre_fix+'mmdet/datasets_my/evaluate_tools/out_pred.txt'
gt = pre_fix+'mmdet/datasets_my/evaluate_tools/output_detailed.txt'

def get_data(lines):
    data = []
    for x in lines:
        x = x.split('|')
        line = [t.strip() for t in x if len(t.strip()) > 0]
        data.append(line)
    return np.array(data)


def compare_recall(pred, gt, margin=0.2):
    lines_pred = open(pred, 'r').readlines()[2:]
    lines_gt = open(gt, 'r').readlines()[2:]
    result_show = [lines_pred[x] + lines_gt[x] for x in range(len(lines_pred))]
    pred_data = get_data(lines_pred)
    gt_data = get_data(lines_gt)
    pred_recall = pred_data[:, 2].astype(np.float)
    gt_recall = gt_data[:, 2].astype(np.float)

    labels = pred_data[:, 0]

    # x = np.arange(len(labels))  # the label locations
    # width = 0.45  # the width of the bars
    # fig, ax = plt.subplots(figsize=(300, 15))
    # rects1 = ax.bar(x - width / 2, pred_recall, width, label='clip')
    # rects2 = ax.bar(x + width / 2, gt_recall, width, label='paper')
    # plt.margins(x=0)
    # ax.set_ylabel('recall')
    # ax.set_title('recall over clip and paper')
    # ax.set_xticks(x, labels, rotation=90)
    # ax.legend()
    # # ax.bar_label(rects1, padding=3)
    # # ax.bar_label(rects2, padding=3)
    # fig.tight_layout()
    # plt.savefig("test.png", bbox_inches='tight', pad_inches=0)

    pred_better = pred_recall > gt_recall + 0.1
    gt_better = gt_recall > pred_recall + 0.5

    labels = np.concatenate([labels[pred_better], labels[gt_better]])
    pred_recall = np.concatenate([pred_recall[pred_better], pred_recall[gt_better]])
    gt_recall = np.concatenate([gt_recall[pred_better], gt_recall[gt_better]])

    x = np.arange(len(labels))  # the label locations
    width = 0.45  # the width of the bars
    fig, ax = plt.subplots(figsize=(300, 15))
    rects1 = ax.bar(x - width / 2, pred_recall, width, label='clip')
    rects2 = ax.bar(x + width / 2, gt_recall, width, label='paper')
    plt.margins(x=0)
    ax.set_ylabel('recall')
    ax.set_title('recall over clip and paper')
    ax.set_xticks(x, labels, rotation=90)
    ax.legend()
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.savefig("test_better.png", bbox_inches='tight', pad_inches=0)


    print(np.array(result_show)[pred_better])
    print(50*'%')
    print(np.array(result_show)[gt_better])


if __name__ == '__main__':
    compare_recall(pred, gt)