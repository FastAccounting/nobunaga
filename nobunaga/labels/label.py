import math

import nobunaga.constants as Const
from nobunaga.utils import calculate_iou

from .gt_label import GtLabel
from .pred_label import PredLabel


class Label(object):
    def __init__(
        self,
        image_id: int,
        image_name: str,
        pred_label: PredLabel,
        match_gt_category_label: GtLabel,
        unmatch_gt_category_label: GtLabel,
        iou_threshold: float,
        confidence_threshold: float,
    ):
        self._image_id = image_id
        self._image_name = image_name
        self._pred_label = pred_label
        self._duplicate_pred_labels = []
        self._match_gt_category_label = match_gt_category_label
        self._unmatch_gt_category_label = unmatch_gt_category_label
        self._iou_threshold = iou_threshold
        self._confidence_threshold = confidence_threshold

    def add_duplicate_pred_labels(self, label: PredLabel):
        self._duplicate_pred_labels.append(label)

    def get_image_id(self):
        return self._image_id

    def get_image_name(self):
        return self._image_name

    def get_pred_label(self):
        return self._pred_label

    def get_duplicate_pred_labels(self):
        return self._duplicate_pred_labels

    def get_gt_match_label(self):
        return self._match_gt_category_label

    def get_gt_unmatch_label(self):
        return self._unmatch_gt_category_label

    def get_pred_category_id(self):
        if not self._pred_label:
            return -1
        return self._pred_label.get_category_id()

    def get_max_match_gt_category_id(self):
        if not self._match_gt_category_label:
            return -1
        return self._match_gt_category_label.get_category_id()

    def get_max_unmatch_gt_category_id(self):
        if not self._unmatch_gt_category_label:
            return -1
        return self._unmatch_gt_category_label.get_category_id()

    def get_max_match_gt_category_iou(self):
        if not self._pred_label or not self._match_gt_category_label:
            return 0
        return calculate_iou(self._match_gt_category_label.get_bbox(), self._pred_label.get_bbox())

    def get_max_unmatch_gt_category_iou(self):
        if not self._pred_label or not self._unmatch_gt_category_label:
            return 0
        return calculate_iou(
            self._unmatch_gt_category_label.get_bbox(), self._pred_label.get_bbox()
        )

    def is_background_error(self):
        return (
            0
            <= self.get_max_unmatch_gt_category_iou()
            <= self.get_max_match_gt_category_iou()
            < Const.THRESHOLD_MIN_DETECTED
        ) or (
            0
            <= self.get_max_match_gt_category_iou()
            < self.get_max_unmatch_gt_category_iou()
            < Const.THRESHOLD_MIN_DETECTED
        ) or (
               self._pred_label is not None
            and self._match_gt_category_label is None
            and self._unmatch_gt_category_label is None
        )

    def is_class_error(self):
        return (
            self.get_pred_category_id() != self.get_max_unmatch_gt_category_id()
            and self.get_max_unmatch_gt_category_iou() > self.get_max_match_gt_category_iou()
            and self.get_max_unmatch_gt_category_iou() >= self._iou_threshold
        )

    def is_location_error(self):
        return (
            self._iou_threshold
            > self.get_max_match_gt_category_iou()
            > Const.THRESHOLD_MIN_DETECTED
            and self.get_max_match_gt_category_iou() >= self.get_max_unmatch_gt_category_iou()
        )

    def is_both_error(self):
        return (
            self.get_pred_category_id() != self.get_max_unmatch_gt_category_id()
            and self.get_max_unmatch_gt_category_iou() > self.get_max_match_gt_category_iou()
            and (
                self._iou_threshold
                > self.get_max_unmatch_gt_category_iou()
                > Const.THRESHOLD_MIN_DETECTED
            )
        )

    def is_miss_error(self):
        return (
            self._pred_label is None
            and self._match_gt_category_label is not None
            and self._unmatch_gt_category_label is None
        )

    def is_duplicate_error(self):
        return len(self._duplicate_pred_labels) > 0

    def is_true_positive(self):
        return (
            self.get_max_match_gt_category_iou() >= self._iou_threshold
            and self.get_max_match_gt_category_iou() > self.get_max_unmatch_gt_category_iou()
        )

    def is_false_positive(self):
        return (
            self.is_background_error()
            or self.is_class_error()
            or self.is_both_error()
            or self.is_location_error()
            or self.is_duplicate_error()
        )

    def is_false_negative(self):
        return self.is_miss_error()

    def get_error_type(self):
        if self.is_class_error():
            return Const.ERROR_TYPE_CLASS
        elif self.is_miss_error():
            return Const.ERROR_TYPE_MISS
        elif self.is_location_error():
            return Const.ERROR_TYPE_LOCATION
        elif self.is_duplicate_error():
            return Const.ERROR_TYPE_DUPLICATE
        elif self.is_class_error():
            return Const.ERROR_TYPE_CLASS
        elif self.is_both_error():
            return Const.ERROR_TYPE_BOTH
        elif self.is_background_error():
            return Const.ERROR_TYPE_BACKGROUND
        else:
            return ""

    def get_correct_distance(self):
        """
        [Cls, Loc, Both, Dupe, Bkg, Miss, No Error, All Errors]
        """
        match_distance = (
            -math.log(self.get_max_match_gt_category_iou(), 10)
            if self.get_max_match_gt_category_iou() > 0
            else 0
        )
        unmatch_distance = 1
        if self.is_class_error():
            return [
                unmatch_distance,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                unmatch_distance,
            ]
        elif self.is_location_error():
            return [
                0.0,
                match_distance,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                match_distance
            ]
        elif self.is_both_error():
            return [
                0.0,
                0.0,
                unmatch_distance,
                0.0,
                0.0,
                0.0,
                0.0,
                unmatch_distance,
            ]
        elif self.is_duplicate_error():
            max_iou = 0.0
            distance = 0.0
            for duplicate_pred_label in self._duplicate_pred_labels:
                iou = calculate_iou(
                    self._match_gt_category_label.get_bbox(), duplicate_pred_label.get_bbox()
                )
                if iou >= max_iou:
                    max_iou = iou
                    if iou > 0.0:
                        distance += -math.log(iou, 10)
            return [
                0.0,
                0.0,
                0.0,
                distance,
                0.0,
                0.0,
                0.0,
                distance
            ]
        elif self.is_background_error():
            distance = -math.log(self._iou_threshold, 10)
            return [
                0.0,
                0.0,
                0.0,
                0.0,
                distance,
                0.0,
                0.0,
                distance
            ]
        elif self.is_miss_error():
            distance = -math.log(self._iou_threshold, 10)
            return [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                distance,
                0.0,
                distance
            ]
        else:
            return [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                match_distance,
                0.0
            ]
