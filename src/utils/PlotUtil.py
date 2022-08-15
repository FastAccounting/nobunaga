from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

import src.Constants as Const
from src import Evaluator


class PlotUtil:
    @staticmethod
    def plot_pie(
        evaluation: Evaluator,
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

    @staticmethod
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
