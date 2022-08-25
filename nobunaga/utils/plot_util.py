from collections import OrderedDict
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame

import nobunaga.constants as Const


def plot_pie(
    evaluation: "nobunaga.evaluator.Evaluator",
    colors_main: OrderedDict,
    pie_path: str,
    high_dpi: int,
    low_dpi: int,
    font_size: int,
    image_size: int,
):

    # pie plot for error type breakdown
    error_sizes = evaluation.get_main_error_distribution()
    fig, ax = plt.subplots(1, 1, figsize=(image_size, image_size), dpi=high_dpi)
    patches, outer_text, inner_text = ax.pie(
        error_sizes,
        colors=colors_main.values(),
        labels=Const.MAIN_ERRORS,
        autopct="%1.1f%%",
        startangle=90,
    )
    for text in outer_text + inner_text:
        text.set_text("")
    for i in range(len(colors_main)):
        if error_sizes[i] > 0.05:
            inner_text[i].set_text(list(colors_main.keys())[i])
        inner_text[i].set_fontsize(font_size)
        inner_text[i].set_fontweight("bold")
    ax.axis("equal")
    plt.savefig(pie_path, bbox_inches="tight", dpi=low_dpi)
    plt.close()


def plot_bar(
    is_vertical: bool,
    data: DataFrame,
    colors: OrderedDict,
    bar_path: str,
    title_labels: list,
    max_scale: int,
    high_dpi: int,
    low_dpi: int,
    title_size: int,
    scale_size: int,
):

    fig, ax = plt.subplots(1, 1, figsize=(len(title_labels), 5), dpi=high_dpi)
    sns.barplot(
        data=data,
        x="x",
        y="y",
        ax=ax,
        palette=colors.values(),
    )
    if is_vertical:
        ax.set_ylim(0, max_scale)
    else:
        ax.set_xlim(0, max_scale)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if is_vertical:
        ax.set_xticklabels(title_labels)
    else:
        ax.set_yticklabels(title_labels)

    x_size = title_size if is_vertical else scale_size
    y_size = title_size if not is_vertical else scale_size
    plt.setp(ax.get_xticklabels(), fontsize=x_size)
    plt.setp(ax.get_yticklabels(), fontsize=y_size)
    ax.grid(False)
    sns.despine(left=True, bottom=True, right=True)
    plt.savefig(bar_path, bbox_inches="tight", dpi=low_dpi)
    plt.close()


def plot_matrix(
    confusion_matrix: List[list],
    rows: list,
    columns: list,
    x_title: str,
    y_title: str,
    output_file_path: str,
):
    cm = pd.DataFrame(data=confusion_matrix, index=rows, columns=columns)
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
    plt.xlabel(x_title, fontsize=13)
    plt.ylabel(y_title, fontsize=13)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(output_file_path)
