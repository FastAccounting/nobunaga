from .Label import *
from .Util import *


class Image:
    def __init__(
        self,
        image: dict,
        categories: dict,
        gts: dict,
        preds: dict,
        iou_threshold: float,
        confidence_threshold: float,
    ):
        self.image = image
        self.categories = categories
        self.gts = gts
        self.preds = preds
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        # relation dict between position and category id
        self.pred_category_row_relations = {}
        for index, pred in enumerate(self.preds):
            self.pred_category_row_relations[index] = pred.get("category_id", -1)

        # relation dict between position and category id
        self.gt_category_column_relations = {}
        for index, gt in enumerate(self.gts):
            self.gt_category_column_relations[index] = gt.get("category_id", -1)

        # row: predict, col: gt
        gt_bboxes = [gt[Const.MODE_BBOX] for gt in self.gts]
        pred_bboxes = [pred[Const.MODE_BBOX] for pred in self.preds]
        self.iou = Util.calculate_ious(gt_bboxes, pred_bboxes)

        self.confidence_matrix = np.array(
            [[pred.get("score") > self.confidence_threshold] * len(self.gts) for pred in self.preds]
        )

        # matrix has true when pred class and ground truth equals
        self.class_matching = [[]]
        if len(self.gts) > 0:
            pred_classes = np.array([pred.get("category_id", "") for pred in self.preds])
            gt_classes = np.array([gt.get("category_id", "") for gt in self.gts])
            self.class_matching = pred_classes[:, None] == gt_classes[None, :]

        # get all predicted label
        self.labels = []
        if len(self.iou) == 0:
            return
        class_match_matrix = self.iou * self.confidence_matrix * self.class_matching
        class_unmatch_matrix = self.iou * self.confidence_matrix * ~self.class_matching
        for gt_index, match_row in enumerate(class_match_matrix):

            # pred label
            pred_category_id = self.pred_category_row_relations.get(gt_index, -1)
            pred_bbox = pred_bboxes[gt_index]
            pred_confidence = self.preds[gt_index].get("score", -1)
            pred_label = PredLabel(
                self.get_image_id(),
                self.get_image_name(),
                gt_index,
                pred_category_id,
                pred_bbox,
                pred_confidence,
            )

            # match gt label
            max_match_gt_index = np.argmax(match_row)
            match_category_id = self.gt_category_column_relations.get(max_match_gt_index, -1)
            match_bbox = gt_bboxes[max_match_gt_index]
            match_gt_label = GtLabel(
                self.get_image_id(),
                self.get_image_name(),
                max_match_gt_index,
                match_category_id,
                match_bbox,
            )

            # unmatch gt label
            unmatch_row = class_unmatch_matrix[gt_index]
            max_unmatch_gt_index = np.argmax(unmatch_row)
            unmatch_category_id = self.gt_category_column_relations.get(max_unmatch_gt_index, -1)
            unmatch_bbox = gt_bboxes[max_unmatch_gt_index]
            unmatch_gt_label = GtLabel(
                self.get_image_id(),
                self.get_image_name(),
                max_unmatch_gt_index,
                unmatch_category_id,
                unmatch_bbox,
            )

            self.labels.append(
                Label(
                    self.get_image_id(),
                    self.get_image_name(),
                    pred_label,
                    match_gt_label,
                    unmatch_gt_label,
                    self.iou_threshold,
                    self.confidence_threshold,
                )
            )

        # miss error label
        transpose_class_match_matrix = self.iou.T * self.class_matching.T
        for transpose_pred_index, gt in enumerate(transpose_class_match_matrix):
            iou = transpose_class_match_matrix[transpose_pred_index][np.argmax(gt, 0)]
            if iou < self.iou_threshold * self.confidence_threshold:
                # record miss category to Label class, it is illegal usage
                gt_category_id = self.gt_category_column_relations.get(transpose_pred_index, -1)
                gt_label = GtLabel(
                    self.get_image_id(),
                    self.get_image_name(),
                    transpose_pred_index,
                    gt_category_id,
                    gt_bboxes[transpose_pred_index],
                )
                self.labels.append(
                    Label(
                        self.get_image_id(),
                        self.get_image_name(),
                        None,
                        gt_label,
                        None,
                        self.iou_threshold,
                        self.confidence_threshold,
                    )
                )

        # duplicate error label
        for transpose_gt_index, class_match_row in enumerate(transpose_class_match_matrix):
            duplicates = []
            for transpose_pred_index, class_match_iou in enumerate(class_match_row):
                if class_match_iou > Const.THRESHOLD_MIN_DETECTED:
                    duplicates.append(transpose_pred_index)
            if len(duplicates) > 1:
                for transpose_pred_index in duplicates:
                    # prediction
                    pred_category_id = self.pred_category_row_relations.get(
                        transpose_pred_index, -1
                    )
                    pred_confidence = self.preds[transpose_pred_index].get("score", -1)
                    pred_label = PredLabel(
                        self.get_image_id(),
                        self.get_image_name(),
                        transpose_pred_index,
                        pred_category_id,
                        pred_bboxes[transpose_pred_index],
                        pred_confidence,
                    )
                    # ground truth
                    match_category_id = self.gt_category_column_relations.get(
                        transpose_gt_index, -1
                    )
                    gt_label = GtLabel(
                        self.get_image_id(),
                        self.get_image_name(),
                        transpose_gt_index,
                        match_category_id,
                        gt_bboxes[transpose_gt_index],
                    )
                    self.labels.append(
                        Label(
                            self.get_image_id(),
                            self.get_image_name(),
                            pred_label,
                            gt_label,
                            None,
                            self.iou_threshold,
                            self.confidence_threshold,
                        )
                    )

    def get_image_id(self):
        return self.image.get("id", -1)

    def get_image_name(self):
        return self.image.get("file_name", "")

    def get_gts(self):
        return self.gts

    def get_preds(self):
        return self.preds

    def get_labels(self):
        return self.labels

    def get_miss_errors(self):
        errors = [label for label in self.labels if label.is_miss_error()]
        return errors

    def get_background_errors(self):
        errors = [label for label in self.labels if label.is_background_error()]
        return errors

    def get_location_errors(self):
        errors = [label for label in self.labels if label.is_location_error()]
        return errors

    def get_class_errors(self):
        errors = [label for label in self.labels if label.is_class_error()]
        return errors

    def get_duplicate_errors(self):
        errors = [label for label in self.labels if label.is_duplicate_error()]
        return errors

    def get_both_errors(self):
        errors = [label for label in self.labels if label.is_both_error()]
        return errors

    def get_normal_labels(self):
        labels = [label for label in self.labels if label.get_error_type() == ""]
        return labels

    def get_false_positive_count(self):
        count = 0
        for predicted_label in self.labels:
            if predicted_label.is_false_positive():
                count += 1
        return count

    def get_true_positive_count(self):
        count = 0
        for predicted_label in self.labels:
            if predicted_label.is_true_positive():
                count += 1
        return count

    def get_true_positive_count_by_category_id(self, category_id: int):
        count = 0
        for label in self.labels:
            if label.get_pred_category_id() == category_id:
                if label.is_true_positive():
                    count += 1
        return count
