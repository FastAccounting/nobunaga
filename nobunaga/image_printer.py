import math
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import nobunaga.constants as Const
from nobunaga.evaluator import Evaluator
from nobunaga.io import (plot_bar, plot_matrix, plot_pie, print_table,
                         write_label)


class ImagePrinter(object):
    def __init__(self, model_name: str, categories: dict, evaluation: Evaluator, image_dir: str):
        self._model_name = model_name
        self._categories = categories
        self._evaluation = evaluation
        self._image_dir = Path(image_dir)
        self._class_error_labels = evaluation.get_class_errors()
        self._location_error_labels = evaluation.get_location_errors()
        self._background_error_labels = evaluation.get_background_errors()
        self._duplicate_error_labels = evaluation.get_duplicate_errors()
        self._miss_error_labels = evaluation.get_miss_errors()
        self._both_error_labels = evaluation.get_both_errors()
        self._index_category_id_relations = {}
        for index, category_id in enumerate(self._categories.keys()):
            self._index_category_id_relations[category_id] = index

        self._out_dir = Path("./_result")
        self._out_dir.mkdir(exist_ok=True)

    def output_confusion_matrix(self, normalize: bool):
        confusion_matrix = {}
        class_count = len(self._categories)
        # row: predicted classes, col: actual classes
        cm = np.zeros((class_count, class_count), dtype=np.int32)
        for class_error in self._class_error_labels:
            try:
                gt_category_id = self._index_category_id_relations.get(
                    class_error.get_max_unmatch_category_id(), -1
                )
                pred_category_id = self._index_category_id_relations.get(
                    class_error.get_pred_category_id(), -1
                )
                cm[pred_category_id][gt_category_id] += 1
            except:
                pass

        # output to terminal
        confusion_matrix[self._model_name] = cm
        print_table(
            [
                ["pred/gt"]
                + [category_name for category_id, category_name in self._categories.items()],
            ]
            + [
                [category_name]
                + [str(cnt) for cnt in cm[self._index_category_id_relations.get(category_id, -1)]]
                for category_id, category_name in self._categories.items()
            ],
            title=f"{self._model_name} confusion matrix",
        )
        confusion_matrix = confusion_matrix[self._model_name].T

        if normalize:
            confusion_matrix = confusion_matrix / confusion_matrix.astype(np.float).sum(axis=0)
        category_names = [category_name for category, category_name in self._categories.items()]
        output_file_path = str(self._out_dir / f"{self._model_name}_class_error_confusion_matrix.png")
        plot_matrix(
            confusion_matrix, category_names, category_names, "Pred", "Gt", output_file_path
        )

    def output_error_type_detail(self, normalize: bool, mode: list = ["confusion_matrix", "strip"]):
        confusion_matrix = {}
        class_count = len(self._categories)
        error_type_count = len(Const.MAIN_ERRORS)
        # row: ground truth classes, col: error_type
        cm = np.zeros((class_count, error_type_count), dtype=np.int32)
        for error_label in self._class_error_labels:
            category_id = self._index_category_id_relations.get(
                error_label.get_pred_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self._location_error_labels:
            category_id = self._index_category_id_relations.get(
                error_label.get_pred_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self._duplicate_error_labels:
            category_id = self._index_category_id_relations.get(
                error_label.get_pred_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self._background_error_labels:
            category_id = self._index_category_id_relations.get(
                error_label.get_pred_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self._miss_error_labels:
            category_id = self._index_category_id_relations.get(
                error_label.get_max_match_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1
        for error_label in self._both_error_labels:
            category_id = self._index_category_id_relations.get(
                error_label.get_pred_category_id(), -1
            )
            error_type_id = Const.MAIN_ERRORS.index(error_label.get_error_type())
            cm[category_id][error_type_id] += 1

        # output to terminal
        confusion_matrix[self._model_name] = cm

        print_table(
            [
                ["label/error"] + [error_type for error_type in Const.MAIN_ERRORS],
            ]
            + [
                [category_name]
                + [str(cnt) for cnt in cm[self._index_category_id_relations.get(category_id)]]
                for category_id, category_name in self._categories.items()
            ],
            title=f"{self._model_name} error type matrix",
        )
        confusion_matrix = confusion_matrix[self._model_name]

        if normalize:
            confusion_matrix = confusion_matrix / confusion_matrix.astype(np.float).sum(axis=0)
        category_names = [category_name for category, category_name in self._categories.items()]
        cm = pd.DataFrame(data=confusion_matrix, index=category_names, columns=Const.MAIN_ERRORS)
        cm.index.name = "Label"

        if "confusion_matrix" in mode:
            output_file_path = str(
                self._out_dir / f"{self._model_name}_error_type_confusion_matrix.png"
            )
            plot_matrix(
                confusion_matrix,
                category_names,
                Const.MAIN_ERRORS,
                "Label",
                "Error",
                output_file_path,
            )

        if "strip" in mode:
            grid_cm = cm.reset_index()
            maximum_dap = math.ceil(cm[Const.MAIN_ERRORS].max().max())
            sns.set(font_scale=0.4)
            g = sns.PairGrid(
                grid_cm,
                x_vars=Const.MAIN_ERRORS,
                y_vars=["Label"],
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

            for idx in range(g.axes.shape[1]):
                g.axes[0, idx].set_xlim(0, maximum_dap)

            for ax, title in zip(g.axes.flat, Const.MAIN_ERRORS):
                ax.set(title=title)
                ax.xaxis.grid(False)
                ax.yaxis.grid(True)
            sns.despine(left=True, bottom=True)
            plt.subplots_adjust(left=0.12, top=0.98)
            plt.savefig(str(self._out_dir / f"{self._model_name}_error_type_strip.png"))

    def output_error_files(self, error_type: str):
        output_dir = Path(f"./{self._model_name}_error")
        output_dir.mkdir(exist_ok=True)

        error_labels = []
        if error_type == Const.ERROR_TYPE_CLASS:
            error_labels = self._class_error_labels
        elif error_type == Const.ERROR_TYPE_LOCATION:
            error_labels = self._location_error_labels
        elif error_type == Const.ERROR_TYPE_BACKGROUND:
            error_labels = self._background_error_labels
        elif error_type == Const.ERROR_TYPE_MISS:
            error_labels = self._miss_error_labels
        elif error_type == Const.ERROR_TYPE_DUPLICATE:
            error_labels = self._duplicate_error_labels
        elif error_type == Const.ERROR_TYPE_BOTH:
            error_labels = self._both_error_labels

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
            elif error_type == Const.ERROR_TYPE_DUPLICATE:
                pred_labels = error_label.get_duplicate_pred_labels()
                pred_labels.append(error_label.get_pred_label())
                pred_label = pred_labels
            image_name = error_label.get_image_name()

            pred_bboxes = []
            if type(pred_label) == list:
                for pred in pred_label:
                    pred_category_name = self._categories.get(pred.get_category_id())
                    pred_bbox = pred.get_bbox()
                    pred_confidence = float("{:.2f}".format(pred.get_confidence() * 100))
                    pred_bboxes.append([pred_category_name] + pred_bbox + [pred_confidence])
            elif not pred_label:
                pred_category_name = ""
                pred_bbox = [0, 0, 0, 0]
                pred_confidence = 0
                pred_bboxes.append([pred_category_name] + pred_bbox + [pred_confidence])
            else:
                pred_category_name = self._categories.get(pred_label.get_category_id())
                pred_bbox = pred_label.get_bbox()
                pred_confidence = float("{:.2f}".format(pred_label.get_confidence() * 100))
                pred_bboxes.append([pred_category_name] + pred_bbox + [pred_confidence])

            gt_bboxes = []
            if not gt_label:
                gt_category_name = ""
                gt_bbox = [0, 0, 0, 0]
                gt_confidence = 0
            else:
                gt_category_name = self._categories.get(gt_label.get_category_id())
                gt_bbox = gt_label.get_bbox()
                gt_confidence = ""
            gt_bboxes.append([gt_category_name] + gt_bbox + [gt_confidence])
            image_name_path = Path(image_name)

            new_file_path = str(
                output_dir
                / f"{image_name_path.stem}_{error_label.get_error_type()}_{str(index_dict.get(image_name, 1))}{image_name_path.suffix}"
            )
            write_label(
                str(self._image_dir / image_name), new_file_path, pred_bboxes, gt_bboxes, True
            )
            index_dict[image_name] = index_dict.get(image_name, 1) + 1

    def output_error_summary(self):

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
                Const.ERROR_FALSE_POSITIVE: current_palette[0],
                Const.ERROR_FALSE_NEGATIVE: current_palette[1],
            }
        )
        main_max_scale = max(self._evaluation.get_main_error_distribution())
        special_max_scale = max(self._evaluation.get_special_error_distribution())

        # Do the plotting now
        tmp_dir = Path("_tmp")
        tmp_dir.mkdir(exist_ok=True)

        high_dpi = int(500)
        low_dpi = int(300)

        # get the data frame
        main_errors = pd.DataFrame(
            data={
                "y": Const.MAIN_ERRORS,
                "x": self._evaluation.get_main_error_distribution(),
            }
        )
        special_errors = pd.DataFrame(
            data={
                "x": Const.SPECIAL_ERRORS,
                "y": self._evaluation.get_special_error_distribution(),
            }
        )

        # pie plot for error type breakdown
        image_size = len(Const.MAIN_ERRORS) + len(Const.SPECIAL_ERRORS)
        pie_path = str(
            tmp_dir / "{}_{}_main_error_pie.png".format(self._model_name, Const.MODE_BBOX)
        )
        plot_pie(self._evaluation, colors_main, pie_path, high_dpi, low_dpi, 36, image_size)
        main_bar_path = str(
            tmp_dir / "{}_{}_main_error_bar.png".format(self._model_name, Const.MODE_BBOX)
        )
        plot_bar(
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
        special_bar_path = str(
            tmp_dir / "{}_{}_special_error_bar.png".format(self._model_name, Const.MODE_BBOX)
        )
        plot_bar(
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

        if self._out_dir is None:
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow((summary_im / 255)[:, :, (2, 1, 0)])
            plt.show()
            plt.close()
        else:
            cv2.imwrite(
                str(self._out_dir / "{}_{}_summary.png".format(self._model_name, Const.MODE_BBOX)),
                summary_im,
            )
