from nobunaga.io import GtJson, PredJson
from nobunaga.labels import Image


class Evaluator(object):
    def __init__(
        self, gt: GtJson, pred: PredJson, iou_threshold: float, confidence_threshold: float
    ):
        self._gt = gt
        self._pred = pred
        self._images = []
        self._iou_threshold = iou_threshold
        self._confidence_threshold = confidence_threshold
        for image_id, gt_annotation in self._gt.get_annotations().items():
            image = self._gt.get_image_by_image_id(image_id)
            pred_annotation = self._pred.get_annotation_by_image_id(image_id)
            image = Image(
                image,
                self._gt.get_categories(),
                gt_annotation,
                pred_annotation,
                iou_threshold,
                confidence_threshold,
            )
            self._images.append(image)

    def get_images(self):
        return self._images

    def get_image_by_image_id(self, image_id: int):
        for image in self._images:
            if image.get_image_id() == image_id:
                return image
        return None

    def get_all_labels(self):
        labels = []
        for image in self._images:
            for label in image.get_labels():
                labels.append(label)
        return labels

    def get_true_positive_by_category_id(self, category_id: int):
        true_positive_count = 0
        for image in self._images:
            true_positive_count += image.get_true_positive_count_by_category_id(category_id)
        return true_positive_count

    def get_gt(self):
        return self._gt

    def get_pred(self):
        return self._pred

    def get_confidence_threshold(self):
        return self._confidence_threshold

    def get_iou_threshold(self):
        return self._iou_threshold

    def get_confidence_threshold(self):
        return self._confidence_threshold

    def get_main_error_distribution(self):
        class_error = len(self.get_class_errors())
        location_error = len(self.get_location_errors())
        both_error = len(self.get_both_errors())
        duplicate_error = len(self.get_duplicate_errors())
        background_error = len(self.get_background_errors())
        miss_error = len(self.get_miss_errors())
        error_distribution = [
            class_error,
            location_error,
            both_error,
            duplicate_error,
            background_error,
            miss_error,
        ]
        return error_distribution

    def get_special_error_distribution(self):
        false_positive = self.get_false_positive_count()
        false_negative = self.get_false_negative_count()
        error_rate = [
            false_positive,
            false_negative,
        ]
        return error_rate

    def get_false_positive_count(self):
        false_positive_count = 0
        for image in self._images:
            false_positive_count = false_positive_count + image.get_false_positive_count()
        return false_positive_count

    def get_true_positive_count(self):
        true_positive_count = 0
        for image in self._images:
            true_positive_count = true_positive_count + image.get_true_positive_count()
        return true_positive_count

    def get_false_negative_count(self):
        false_negative_count = 0
        for image in self._images:
            false_negative_count = false_negative_count + image.get_false_negative_count()
        return false_negative_count

    def get_class_errors(self):
        error_labels = []
        for image in self._images:
            if image.get_class_errors():
                error_labels.extend(image.get_class_errors())
        return error_labels

    def get_location_errors(self):
        error_labels = []
        for image in self._images:
            if image.get_location_errors():
                error_labels.extend(image.get_location_errors())
        return error_labels

    def get_miss_errors(self):
        error_labels = []
        for image in self._images:
            if image.get_miss_errors():
                error_labels.extend(image.get_miss_errors())
        return error_labels

    def get_background_errors(self):
        error_labels = []
        for image in self._images:
            if image.get_background_errors():
                error_labels.extend(image.get_background_errors())
        return error_labels

    def get_duplicate_errors(self):
        error_labels = []
        for image in self._images:
            if image.get_duplicate_errors():
                error_labels.extend(image.get_duplicate_errors())
        return error_labels

    def get_both_errors(self):
        error_labels = []
        for image in self._images:
            if image.get_both_errors():
                error_labels.extend(image.get_both_errors())
        return error_labels

    def get_class_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self._images:
            for label in image.get_class_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_location_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self._images:
            for label in image.get_location_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_miss_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self._images:
            for label in image.get_miss_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_background_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self._images:
            for label in image.get_background_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_duplicate_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self._images:
            for label in image.get_duplicate_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels

    def get_both_errors_by_category_id(self, category_id: int):
        error_labels = []
        for image in self._images:
            for label in image.get_both_errors():
                if label.get_pred_category_id() == category_id:
                    error_labels.append(label)
        return error_labels
