import math
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

import src.Constants as Const
from src import Evaluator
from src.utils import PlotUtil, Util


class ImagePrinter:
    def __init__(self, model_name: str, categories: dict, evaluation: Evaluator):
        self.model_name = model_name
        self.categories = categories
        self.evaluation = evaluation
        self.class_error_labels = evaluation.get_class_errors()
        self.location_error_labels = evaluation.get_location_errors()
        self.background_error_labels = evaluation.get_background_errors()
        self.duplicate_error_labels = evaluation.get_duplicate_errors()
        self.miss_error_labels = evaluation.get_miss_errors()
        self.both_error_labels = evaluation.get_both_errors()
        self.index_category_id_relations = {}
        for index, category_id in enumerate(self.categories.keys()):
            self.index_category_id_relations[category_id] = index

    def output_confusion_matrix(self, normalize: bool):
        confusion_matrix = {}
        class_count = len(self.categories)
        # row: predicted classes, col: actual classes
        cm = np.zeros((class_count, class_count), dtype=np.int32)
        for class_error in self.class_error_labels:
            try:
                gt_category_id = class_error.get_max_match_category_id()
                pred_category_id = class_error.get_pred_category_id()
                cm[pred_category_id][gt_category_id] += 1
            except:
                pass

        # output to terminal
        confusion_matrix[self.model_name] = cm
        Util.print_table(
            [
                ["pred/gt"]
                + [category_name for category_id, category_name in self.categories.items()],
            ]
            + [
                [category_name]
                + [str(cnt) for cnt in cm[self.index_category_id_relations.get(category_id, -1)]]
                for category_id, category_name in self.categories.items()
            ],
            title=f"{self.model_name} confusion matrix",
        )
        confusion_matrix = confusion_matrix[self.model_name].T

        if normalize:
            confusion_matrix = confusion_matrix / confusion_matrix.astype(np.float).sum(axis=0)
        category_names = [category_name for category, category_name in self.categories.items()]
        cm = pd.DataFrame(data=confusion_matrix, index=category_names, columns=category_names)
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
        plt.savefig("_class_error_confusion_matrix.png")

    def output_error_type_matrix(self, normalize: bool):
        confusion_matrix = {}
        class_count = len(self.categories)
        error_type_count = len(Const.MAIN_ERRORS)
        # row: ground truth classes, col: error_type
        cm = np.zeros((class_count, error_type_count), dtype=np.int32)
        for error_label in self.class_error_labels:
            category_id = self.index_category_id_relations.get(
                error_label.get_max_match_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self.location_error_labels:
            category_id = self.index_category_id_relations.get(
                error_label.get_max_match_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self.duplicate_error_labels:
            category_id = self.index_category_id_relations.get(
                error_label.get_max_match_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self.background_error_labels:
            category_id = self.index_category_id_relations.get(
                error_label.get_max_match_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self.miss_error_labels:
            category_id = self.index_category_id_relations.get(
                error_label.get_max_match_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self.both_error_labels:
            category_id = self.index_category_id_relations.get(
                error_label.get_max_match_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1

        # output to terminal
        confusion_matrix[self.model_name] = cm
        Util.print_table(
            [
                ["label/error"] + [error_type for error_type in Const.MAIN_ERRORS],
            ]
            + [
                [category_name]
                + [str(cnt) for cnt in cm[self.index_category_id_relations.get(category_id)]]
                for category_id, category_name in self.categories.items()
            ],
            title=f"{self.model_name} error type matrix",
        )
        confusion_matrix = confusion_matrix[self.model_name]

        if normalize:
            confusion_matrix = confusion_matrix / confusion_matrix.astype(np.float).sum(axis=0)
        category_names = [category_name for category, category_name in self.categories.items()]
        cm = pd.DataFrame(data=confusion_matrix, index=category_names, columns=Const.MAIN_ERRORS)
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
        plt.xlabel("Error", fontsize=13)
        plt.ylabel("Label", fontsize=13)
        fig.subplots_adjust(bottom=0.15)
        plt.savefig("_error_type_confusion_matrix.png")

    def output_per_accuracy_and_errors(self):
        total_table = []
        for category_id, category_name in self.categories.items():
            true_positive_count = self.evaluation.get_true_positive_by_category_id(category_id)
            class_error_count = len(self.evaluation.get_class_errors_by_category_id(category_id))
            location_error_count = len(
                self.evaluation.get_location_errors_by_category_id(category_id)
            )
            both_error_count = len(self.evaluation.get_both_errors_by_category_id(category_id))
            duplicate_error_count = len(
                self.evaluation.get_duplicate_errors_by_category_id(category_id)
            )
            background_error_count = len(
                self.evaluation.get_background_errors_by_category_id(category_id)
            )
            miss_error_count = len(self.evaluation.get_miss_errors_by_category_id(category_id))

            # class error
            all_count = (
                class_error_count
                + location_error_count
                + both_error_count
                + duplicate_error_count
                + background_error_count
                + miss_error_count
                + true_positive_count
            )
            if all_count == 0:
                rows_items = [category_name, 0, 0, 0, 0, 0, 0, 0]
            else:
                precision = true_positive_count / all_count
                class_precision = (
                    0
                    if class_error_count + true_positive_count == 0
                    else true_positive_count / (class_error_count + true_positive_count)
                )
                location_precision = (
                    0
                    if location_error_count + true_positive_count == 0
                    else true_positive_count / (location_error_count + true_positive_count)
                )
                both_precision = (
                    0
                    if both_error_count + true_positive_count == 0
                    else true_positive_count / (both_error_count + true_positive_count)
                )
                duplicate_precision = (
                    0
                    if duplicate_error_count + true_positive_count == 0
                    else true_positive_count / (duplicate_error_count + true_positive_count)
                )
                background_precision = (
                    0
                    if background_error_count + true_positive_count == 0
                    else true_positive_count / (background_error_count + true_positive_count)
                )
                miss_precision = (
                    0
                    if miss_error_count + true_positive_count == 0
                    else true_positive_count / (miss_error_count + true_positive_count)
                )
                rows_items = [
                    category_name,
                    precision,
                    class_precision,
                    location_precision,
                    both_precision,
                    duplicate_precision,
                    background_precision,
                    miss_precision,
                ]
            total_table.append(rows_items)

        df = pd.DataFrame(total_table, columns=["Name", "AP"] + Const.MAIN_ERRORS)
        maximum_dap = math.ceil(df[Const.MAIN_ERRORS].max().max())

        sns.set(font_scale=0.4)
        g = sns.PairGrid(
            df, x_vars=["AP"] + Const.MAIN_ERRORS, y_vars=["Name"], height=8, aspect=0.25
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

        titles = ["AP"] + Const.MAIN_ERRORS
        for ax, title in zip(g.axes.flat, titles):
            ax.set(title=title)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True)

        sns.despine(left=True, bottom=True)
        plt.subplots_adjust(left=0.05, top=0.98)
        plt.savefig("_per_accuracy_and_errors.png")

    def output_error_files(self, image_dir: str, error_type: str):
        image_dir = Path(image_dir)
        output_dir = Path(f"./{self.model_name}_error")
        output_dir.mkdir(exist_ok=True)

        error_labels = []
        if error_type == Const.ERROR_TYPE_CLASS:
            error_labels = self.class_error_labels
        elif error_type == Const.ERROR_TYPE_LOCATION:
            error_labels = self.location_error_labels
        elif error_type == Const.ERROR_TYPE_BACKGROUND:
            error_labels = self.background_error_labels
        elif error_type == Const.ERROR_TYPE_MISS:
            error_labels = self.miss_error_labels
        elif error_type == Const.ERROR_TYPE_DUPLICATE:
            error_labels = self.duplicate_error_labels
        elif error_type == Const.ERROR_TYPE_BOTH:
            error_labels = self.both_error_labels

        # class error
        index_dict = {}
        for error_label in tqdm(error_labels, error_type + " error"):
            pred_label = error_label.get_pred_label()
            if error_type == Const.ERROR_TYPE_MISS:
                pred_label = None
            gt_label = error_label.get_gt_match_label()
            if error_type == Const.ERROR_TYPE_CLASS or error_type == Const.ERROR_TYPE_BOTH:
                gt_label = error_label.get_gt_unmatch_label()
            elif error_type == Const.ERROR_TYPE_BACKGROUND:
                gt_label = None
            image_name = error_label.get_image_name()

            pred_bboxes = []
            if not pred_label:
                pred_category_name = ""
                pred_bbox = [0, 0, 0, 0]
                pred_confidence = 0
            else:
                pred_category_name = self.categories.get(pred_label.get_category_id())
                pred_bbox = pred_label.get_bbox()
                pred_confidence = float("{:.2f}".format(pred_label.get_confidence() * 100))
            pred_bboxes.append([pred_category_name] + pred_bbox + [pred_confidence])

            gt_bboxes = []
            if not gt_label:
                gt_category_name = ""
                gt_bbox = [0, 0, 0, 0]
                gt_confidence = 0
            else:
                gt_category_name = self.categories.get(gt_label.get_category_id())
                gt_bbox = gt_label.get_bbox()
                gt_confidence = ""
            gt_bboxes.append([gt_category_name] + gt_bbox + [gt_confidence])
            image_name_path = Path(image_name)

            new_file_path = str(
                output_dir
                / f"{image_name_path.stem}_{error_label.get_error_type()}_{str(index_dict.get(image_name, 1))}{image_name_path.suffix}"
            )
            Util.write_label(
                str(image_dir / image_name), new_file_path, pred_bboxes, gt_bboxes, True
            )
            index_dict[image_name] = index_dict.get(image_name, 1) + 1

    def output_error_summary(self):
        out_dir = "./_result"
        os.makedirs(out_dir, exist_ok=True)

        mpl.rcParams["figure.dpi"] = 150

        # Seaborn color palette
        sns.set_palette("muted", 10)
        current_palette = sns.color_palette()

        # Seaborn style
        sns.set(style="whitegrid")

        colors_main = OrderedDict(
            {
                Const.ERROR_TYPE_CLASS: current_palette[9],
                Const.ERROR_TYPE_LOCATION: current_palette[8],
                Const.ERROR_TYPE_BOTH: current_palette[2],
                Const.ERROR_TYPE_DUPLICATE: current_palette[6],
                Const.ERROR_TYPE_BACKGROUND: current_palette[4],
                Const.ERROR_TYPE_MISS: current_palette[3],
            }
        )

        colors_special = OrderedDict(
            {
                Const.ERROR_FALSE_NEGATIVE: current_palette[0],
                Const.ERROR_FALSE_POSITIVE: current_palette[1],
                Const.ERROR_TRUE_POSITIVE: current_palette[2],
                Const.ERROR_TRUE_NEGATIVE: current_palette[3],
            }
        )
        main_max_scale = max(self.evaluation.get_main_error_distribution())
        special_max_scale = max(self.evaluation.get_special_error_distribution())

        # Do the plotting now
        tmp_dir = "_tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        high_dpi = int(500)
        low_dpi = int(300)

        # get the data frame
        main_errors = pd.DataFrame(
            data={
                "y": Const.MAIN_ERRORS,
                "x": self.evaluation.get_main_error_distribution(),
            }
        )
        special_errors = pd.DataFrame(
            data={
                "x": Const.SPECIAL_ERRORS,
                "y": self.evaluation.get_special_error_distribution(),
            }
        )

        # pie plot for error type breakdown
        image_size = len(Const.MAIN_ERRORS) + len(Const.SPECIAL_ERRORS)
        pie_path = os.path.join(
            tmp_dir, "{}_{}_main_error_pie.png".format(self.model_name, Const.MODE_BBOX)
        )
        PlotUtil.plot_pie(self.evaluation, colors_main, pie_path, high_dpi, low_dpi, 36, image_size)
        main_bar_path = os.path.join(
            tmp_dir, "{}_{}_main_error_bar.png".format(self.model_name, Const.MODE_BBOX)
        )
        PlotUtil.plot_bar(
            False,
            main_errors,
            colors_main,
            main_bar_path,
            Const.MAIN_ERRORS,
            main_max_scale,
            high_dpi,
            low_dpi,
            18,
            14,
        )
        special_bar_path = os.path.join(
            tmp_dir, "{}_{}_special_error_bar.png".format(self.model_name, Const.MODE_BBOX)
        )
        PlotUtil.plot_bar(
            True,
            special_errors,
            colors_special,
            special_bar_path,
            Const.SPECIAL_ERRORS,
            special_max_scale,
            high_dpi,
            low_dpi,
            18,
            14,
        )

        # get each subplot image
        pie_im = cv2.imread(pie_path)
        main_bar_im = cv2.imread(main_bar_path)
        special_bar_im = cv2.imread(special_bar_path)

        # pad the hbar image vertically
        main_bar_im = np.concatenate(
            [
                np.zeros((special_bar_im.shape[0] - main_bar_im.shape[0], main_bar_im.shape[1], 3))
                + 255,
                main_bar_im,
            ],
            axis=0,
        )
        summary_im = np.concatenate([main_bar_im, special_bar_im], axis=1)

        # pad summary_im
        if summary_im.shape[1] < pie_im.shape[1]:
            lpad, rpad = int(np.ceil((pie_im.shape[1] - summary_im.shape[1]) / 2)), int(
                np.floor((pie_im.shape[1] - summary_im.shape[1]) / 2)
            )
            summary_im = np.concatenate(
                [
                    np.zeros((summary_im.shape[0], lpad, 3)) + 255,
                    summary_im,
                    np.zeros((summary_im.shape[0], rpad, 3)) + 255,
                ],
                axis=1,
            )

        # pad pie_im
        else:
            lpad, rpad = int(np.ceil((summary_im.shape[1] - pie_im.shape[1]) / 2)), int(
                np.floor((summary_im.shape[1] - pie_im.shape[1]) / 2)
            )
            pie_im = np.concatenate(
                [
                    np.zeros((pie_im.shape[0], lpad, 3)) + 255,
                    pie_im,
                    np.zeros((pie_im.shape[0], rpad, 3)) + 255,
                ],
                axis=1,
            )

        summary_im = np.concatenate([pie_im, summary_im], axis=0)

        if out_dir is None:
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow((summary_im / 255)[:, :, (2, 1, 0)])
            plt.show()
            plt.close()
        else:
            cv2.imwrite(
                os.path.join(out_dir, "{}_{}_summary.png".format(self.model_name, Const.MODE_BBOX)),
                summary_im,
            )
