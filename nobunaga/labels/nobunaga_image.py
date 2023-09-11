import numpy as np

import nobunaga.constants as Const
from nobunaga.labels import GtLabel, Label, PredLabel
from nobunaga.utils import calculate_ious


class Image(object):
    def __init__(
        self,
        image: dict,
        categories: dict,
        gts: dict,
        preds: dict,
        iou_threshold: float,
        confidence_threshold: float,
    ):
        self._image = image
        self._categories = categories
        self._gts = gts
        self._preds = preds
        self._iou_threshold = iou_threshold
        self._confidence_threshold = confidence_threshold

        # relation dict between position and category id
        self._pred_category_row_relations = {}
        for index, pred in enumerate(self._preds):
            self._pred_category_row_relations[index] = pred.get("category_id", -1)

        # relation dict between position and category id
        self._gt_category_column_relations = {}
        for index, gt in enumerate(self._gts):
            self._gt_category_column_relations[index] = gt.get("category_id", -1)

        # row: predict, col: gt
        gt_bboxes = [gt[Const.MODE_BBOX] for gt in self._gts]
        gt_error_types = [None for gt in self._gts]
        pred_bboxes = [pred[Const.MODE_BBOX] for pred in self._preds]
        self._iou = calculate_ious(gt_bboxes, pred_bboxes)

        self._is_confident_matrix = np.array(
            [
                [pred.get("score") >= self._confidence_threshold] * len(self._gts)
                for pred in self._preds
            ]
        )

        # matrix has true when pred class and ground truth equals
        self._match_gt_category = [[]]
        if len(self._gts) > 0:
            pred_categories = np.array([pred.get("category_id", "") for pred in self._preds])
            gt_categories = np.array([gt.get("category_id", "") for gt in self._gts])
            self._match_gt_category = pred_categories[:, None] == gt_categories[None, :]

        # get all predicted label
        self._labels = []
        if len(self._iou) == 0:
            return
        match_gt_category_matrix = (
                self._iou * self._is_confident_matrix * self._match_gt_category
        )
        unmatch_gt_category_matrix = (
                self._iou * self._is_confident_matrix * ~self._match_gt_category
        )
        for pred_row_index, match_row in enumerate(match_gt_category_matrix):
            # pred label
            pred_category_id = self._pred_category_row_relations.get(pred_row_index, -1)
            pred_bbox = pred_bboxes[pred_row_index]
            pred_confidence = self._preds[pred_row_index].get("score", -1)
            pred_label = PredLabel(
                self.get_image_id(),
                self.get_image_name(),
                pred_row_index,
                pred_category_id,
                pred_bbox,
                pred_confidence,
            )

            # match gt label
            max_match_gt_category_index = np.argmax(match_row)
            match_gt_category_id = self._gt_category_column_relations.get(
                max_match_gt_category_index, -1
            )
            match_gt_category_bbox = gt_bboxes[max_match_gt_category_index]
            match_gt_category_label = GtLabel(
                self.get_image_id(),
                self.get_image_name(),
                max_match_gt_category_index,
                match_gt_category_id,
                match_gt_category_bbox,
            )

            # unmatch gt label
            unmatch_gt_category_row = unmatch_gt_category_matrix[pred_row_index]
            max_unmatch_gt_category_index = np.argmax(unmatch_gt_category_row)
            unmatch_gt_category_id = self._gt_category_column_relations.get(
                max_unmatch_gt_category_index, -1
            )
            unmatch_gt_category_bbox = gt_bboxes[max_unmatch_gt_category_index]
            unmatch_gt_category_label = GtLabel(
                self.get_image_id(),
                self.get_image_name(),
                max_unmatch_gt_category_index,
                unmatch_gt_category_id,
                unmatch_gt_category_bbox,
            )

            label = Label(
                self.get_image_id(),
                self.get_image_name(),
                pred_label,
                match_gt_category_label,
                unmatch_gt_category_label,
                self._iou_threshold,
                self._confidence_threshold,
            )

            if gt_error_types[max_match_gt_category_index] is None:
                gt_error_types[max_match_gt_category_index] = label.get_error_type()
                self._labels.append(label)
            else:
                is_duplicate_error = False
                for exist_label in self._labels:
                    # duplicate error(more than 2 pred detect same gt)
                    if (
                        exist_label.get_gt_match_label().get_index()
                        == label.get_gt_match_label().get_index()
                        and label.get_max_match_gt_category_iou() >= self._iou_threshold
                        and label.get_max_match_gt_category_iou()
                        > label.get_max_unmatch_gt_category_iou()
                        and exist_label.get_max_match_gt_category_iou() >= self._iou_threshold
                        and exist_label.get_max_match_gt_category_iou()
                        > exist_label.get_max_unmatch_gt_category_iou()
                    ):
                        exist_label.add_duplicate_pred_labels(label.get_pred_label())
                        gt_error_types[max_match_gt_category_index] = Const.ERROR_TYPE_DUPLICATE
                        is_duplicate_error = True
                        break
                if not is_duplicate_error:
                    gt_error_types[max_match_gt_category_index] = label.get_error_type()
                    self._labels.append(label)

        # background error label (detected but no gt exists.)
        iou_matrix = self._iou * self._is_confident_matrix
        for pred_index, iou in enumerate(iou_matrix):
            if (
                iou[np.argmax(iou)] == 0
                and self._preds[pred_index].get("score", -1) > self._confidence_threshold
            ):
                pred_category_id = self._pred_category_row_relations.get(pred_index, -1)
                pred_bbox = pred_bboxes[pred_index]
                pred_confidence = self._preds[pred_index].get("score", -1)
                pred_label = PredLabel(
                    self.get_image_id(),
                    self.get_image_name(),
                    pred_index,
                    pred_category_id,
                    pred_bbox,
                    pred_confidence,
                )
                label = Label(
                    self.get_image_id(),
                    self.get_image_name(),
                    pred_label,
                    None,
                    None,
                    self._iou_threshold,
                    self._confidence_threshold,
                )
                self._labels.append(label)

        for gt_index, gt_error_type in enumerate(gt_error_types):
            if gt_error_type is not None:
                continue
            gt_category_id = self._gt_category_column_relations.get(gt_index, -1)
            gt_label = GtLabel(
                self.get_image_id(),
                self.get_image_name(),
                gt_index,
                gt_category_id,
                gt_bboxes[gt_index],
            )

            label = Label(
                self.get_image_id(),
                self.get_image_name(),
                None,
                gt_label,
                None,
                self._iou_threshold,
                self._confidence_threshold,
            )
            if gt_error_types[gt_index] is None:
                self._labels.append(label)
                gt_error_types[gt_index] = label.get_error_type()

    def get_image_id(self):
        return self._image.get("id", -1)

    def get_image_name(self):
        return self._image.get("file_name", "")

    def get_gts(self):
        return self._gts

    def get_preds(self):
        return self._preds

    def get_labels(self):
        return self._labels

    def get_miss_errors(self):
        errors = [label for label in self._labels if label.is_miss_error()]
        return errors

    def get_background_errors(self):
        errors = [label for label in self._labels if label.is_background_error()]
        return errors

    def get_location_errors(self):
        errors = [label for label in self._labels if label.is_location_error()]
        return errors

    def get_class_errors(self):
        errors = [label for label in self._labels if label.is_class_error()]
        return errors

    def get_duplicate_errors(self):
        errors = [label for label in self._labels if label.is_duplicate_error()]
        return errors

    def get_both_errors(self):
        errors = [label for label in self._labels if label.is_both_error()]
        return errors

    def get_normal_labels(self):
        labels = [label for label in self._labels if label.get_error_type() == ""]
        return labels

    def get_false_positive_count(self):
        count = 0
        for predicted_label in self._labels:
            if predicted_label.is_false_positive():
                count += 1
        return count

    def get_true_positive_count(self):
        count = 0
        for predicted_label in self._labels:
            if predicted_label.is_true_positive():
                count += 1
        return count

    def get_false_negative_count(self):
        count = 0
        for predicted_label in self._labels:
            if predicted_label.is_false_negative():
                count += 1
        return count

    def get_true_positive_count_by_category_id(self, category_id: int):
        count = 0
        for label in self._labels:
            if label.get_pred_category_id() == category_id:
                if label.is_true_positive():
                    count += 1
        return count

    def get_correct_distance(self):
        """
        [Cls, Loc, Both, Dupe, Bkg, Miss, No Error, All Errors]
        """
        correct_distances = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for label in self._labels:
            correct_distances = [
                x + y for x, y in zip(correct_distances, label.get_correct_distance())
            ]
        return correct_distances

    def get_correction_cost(self):
        return self.get_correct_distance()[-1]
