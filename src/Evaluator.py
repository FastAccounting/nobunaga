from .PredJson import *
from .GtJson import *
from .Image import *


class Evaluator:
    def __init__(self, gt: GtJson, pred: PredJson, iou_threshold: float, confidence_threshold: float):
        self.gt = gt
        self.pred = pred
        self.images = []
        for image_id, gt_annotation in self.gt.get_annotations().items():
            image = self.gt.get_image_by_image_id(image_id)
            pred_annotation = self.pred.get_annotation_by_image_id(image_id)
            image = Image(image,
                          self.gt.get_categories(),
                          gt_annotation,
                          pred_annotation,
                          iou_threshold,
                          confidence_threshold
                          )
            self.images.append(image)

    def get_images(self):
        return self.images

    def get_all_labels(self):
        labels = []
        for image in self.images:
            for label in image.get_labels():
                labels.append(label)
        return labels

    def get_true_positive_by_category_id(self, category_id: int):
        true_positive_count = 0
        for image in self.images:
            true_positive_count += image.get_true_positive_count_by_category_id(category_id)
        return true_positive_count

    def get_gt(self):
        return self.gt

    def get_pred(self):
        return self.pred

    def get_class_errors(self):
        error_labels = []
        for image in self.images:
            if image.get_class_errors():
                error_labels.extend(image.get_class_errors())
        return error_labels

    def get_location_errors(self):
        error_labels = []
        for image in self.images:
            if image.get_location_errors():
                error_labels.extend(image.get_location_errors())
        return error_labels

    def get_miss_errors(self):
        error_labels = []
        for image in self.images:
            if image.get_miss_errors():
                error_labels.extend(image.get_miss_errors())
        return error_labels

    def get_background_errors(self):
        error_labels = []
        for image in self.images:
            if image.get_background_errors():
                error_labels.extend(image.get_background_errors())
        return error_labels

    def get_duplicate_errors(self):
        error_labels = []
        for image in self.images:
            if image.get_duplicate_errors():
                error_labels.extend(image.get_duplicate_errors())
        return error_labels

    def get_both_errors(self):
        error_labels = []
        for image in self.images:
            if image.get_both_errors():
                error_labels.extend(image.get_both_errors())
        return error_labels

    def get_class_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self.images:
            for label in image.get_class_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_location_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self.images:
            for label in image.get_location_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_miss_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self.images:
            for label in image.get_miss_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_background_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self.images:
            for label in image.get_background_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_duplicate_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self.images:
            for label in image.get_duplicate_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_both_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self.images:
            for label in image.get_both_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels
