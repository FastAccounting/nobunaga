import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from tidecv import TIDE, datasets


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", "-a", type=str, default="examples/instances_val2017.json")
    parser.add_argument("--result", "-r", default="examples/coco_instances_results.json", type=str)
    parser.add_argument("--name", "-n", default="", type=str)
    parser.add_argument("--show", type=bool, default=True)
    parser.add_argument("--normalize", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = arg()

    tide = TIDE()
    gt = datasets.COCO(args.annotation)
    bbox_results = datasets.COCOResult(args.result)
    # tide.evaluate(datasets.COCO(), datasets.COCOResult('path/to/your/results/file'), mode=TIDE.BOX) # Use TIDE.MASK for masks
    run = tide.evaluate(gt, bbox_results, mode=TIDE.BOX, name=args.name)  # Use TIDE.MASK for masks
    tide.summarize()
    tide.plot("./result")

    errors = tide.get_main_per_class_errors()
    error_names = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
    total_table = []
    for idx in sorted(errors[""][error_names[0]].keys()):
        rows_items = [
            gt.classes[idx],
        ]
        rows_items += [run.ap_data.objs[idx].get_ap()]
        rows_items += [errors[""][name][idx] for name in error_names]
        total_table.append(rows_items)

    df = pd.DataFrame(total_table, columns=["Name", "AP"] + error_names)
    maximum_dap = math.ceil(df[error_names].max().max())

    sns.set(font_scale=0.4)
    g = sns.PairGrid(
        df.sort_values("AP", ascending=False),
        x_vars=["AP"] + error_names,
        y_vars=["Name"],
        height=8,
        aspect=0.25,
    )
    g.map(
        sns.stripplot,
        size=8,
        orient="h",
        jitter=False,
        palette="flare_r",
        linewidth=1,
        edgecolor="w",
    )
    g.axes[0, 0].set_xlim(0, 100)
    for idx in range(1, g.axes.shape[1]):
        g.axes[0, idx].set_xlim(0, maximum_dap)
        g.axes[0, idx].xaxis.set_major_locator(MultipleLocator(5))

    titles = ["AP"] + error_names
    for ax, title in zip(g.axes.flat, titles):
        ax.set(title=title)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)
    plt.subplots_adjust(left=0.05, top=0.98)
    plt.savefig("per_accuracy_and_errors.png")

    ret = tide.get_confusion_matrix()
    dat = ret[""].T

    if args.normalize:
        dat = dat / dat.astype(np.float).sum(axis=0)
    cm = pd.DataFrame(data=dat, index=gt.classes.values(), columns=gt.classes.values())
    sns.set(font_scale=0.4)
    fig, axes = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        square=True,
        cbar=True,
        annot=False,
        cmap="Blues",
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
    )
    plt.xlabel("Predict", fontsize=13)
    plt.ylabel("GT", fontsize=13)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig("class_error_confusion_matrix.png")


if __name__ == "__main__":
    main()
