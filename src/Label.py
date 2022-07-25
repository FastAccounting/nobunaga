import src.Constants as Const
from src.utils.Util import *

from .GtLabel import *
from .PredLabel import *


class Label:
    def __init__(
        self,
        image_id: int,
        image_name: str,
        pred_label: PredLabel,
        gt_match_label: GtLabel,
        gt_unmatch_label: GtLabel,
        iou_threshold: float,
        confidence_threshold: float,
    ):

        self.image_id = image_id
        self.image_name = image_name
        self.pred_label = pred_label
        self.gt_match_label = gt_match_label
        self.gt_unmatch_label = gt_unmatch_label
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def get_image_id(self):
        return self.image_id

    def get_image_name(self):
        return self.image_name

    def get_pred_label(self):
        return self.pred_label

    def get_gt_match_label(self):
        return self.gt_match_label

    def get_gt_unmatch_label(self):
        return self.gt_unmatch_label

    def get_pred_category_id(self):
        if not self.pred_label:
            return -1
        return self.pred_label.get_category_id()

    def get_max_match_category_id(self):
        if not self.gt_match_label:
            return -1
        return self.gt_match_label.get_category_id()

    def get_max_unmatch_category_id(self):
        if not self.gt_unmatch_label:
            return -1
        return self.gt_unmatch_label.get_category_id()

    def get_max_match_iou(self):
        if not self.pred_label or not self.gt_match_label:
            return 0
        return Util.calculate_iou(self.gt_match_label.get_bbox(), self.pred_label.get_bbox())

    def get_max_unmatch_iou(self):
        if not self.pred_label or not self.gt_unmatch_label:
            return 0
        return Util.calculate_iou(self.gt_unmatch_label.get_bbox(), self.pred_label.get_bbox())

    def is_background_error(self):
        return (
            0
            <= self.get_max_unmatch_iou()
            < self.get_max_match_iou()
            < Const.THRESHOLD_MIN_DETECTED
            and not self.is_duplicate_error()
            and not self.is_miss_error()
        )

    def is_class_error(self):
        return (
            self.get_pred_category_id() != self.get_max_unmatch_category_id()
            and self.get_max_unmatch_iou() > self.get_max_match_iou()
            and self.get_max_unmatch_iou() > self.iou_threshold
            and not self.is_duplicate_error()
            and not self.is_miss_error()
        )

    def is_location_error(self):
        return (
            self.iou_threshold > self.get_max_match_iou() > Const.THRESHOLD_MIN_DETECTED
            and self.get_max_match_iou() > self.get_max_unmatch_iou()
            and not self.is_duplicate_error()
            and not self.is_miss_error()
        )

    def is_both_error(self):
        return (
            self.get_pred_category_id() != self.get_max_unmatch_category_id()
            and self.get_max_unmatch_iou() > self.get_max_match_iou()
            and self.iou_threshold > self.get_max_unmatch_iou() > Const.THRESHOLD_MIN_DETECTED
            and not self.is_duplicate_error()
            and not self.is_miss_error()
        )

    def is_miss_error(self):
        return (
            self.pred_label is None
            and self.gt_match_label is not None
            and self.gt_unmatch_label is None
        )

    def is_duplicate_error(self):
        return (
            self.pred_label is not None
            and self.gt_match_label is not None
            and self.gt_unmatch_label is None
        )

    def is_true_positive(self):
        return self.get_max_match_iou() > self.iou_threshold

    def is_false_positive(self):
        return self.get_max_unmatch_iou() > self.iou_threshold

    def is_false_negative(self):
        return self.iou_threshold > self.get_max_match_iou()

    def is_true_negative(self):
        return self.iou_threshold > self.get_max_unmatch_iou()

    def get_error_type(self):
        if self.is_class_error():
            return Const.ERROR_TYPE_CLASS
        elif self.is_location_error():
            return Const.ERROR_TYPE_LOCATION
        elif self.is_background_error():
            return Const.ERROR_TYPE_BACKGROUND
        elif self.is_both_error():
            return Const.ERROR_TYPE_BOTH
        elif self.is_miss_error():
            return Const.ERROR_TYPE_MISS
        elif self.is_duplicate_error():
            return Const.ERROR_TYPE_DUPLICATE
        else:
            return ""
