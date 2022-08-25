import numpy as np


def calculate_ious(gt_bboxes: list, pred_bboxes: list):
    iou_matrix = []
    for pred_bbox in pred_bboxes:
        iou_row = []
        for gt_bbox in gt_bboxes:
            iou = calculate_iou(gt_bbox, pred_bbox)
            iou_row.append(iou)
        iou_matrix.append(iou_row)
    return np.array(iou_matrix)


def calculate_iou(gt_bbox: list, pred_bbox: list):
    pred_left = pred_bbox[0]
    pred_width = pred_bbox[2]
    pred_right = pred_left + pred_width
    pred_top = pred_bbox[1]
    pred_height = pred_bbox[3]
    pred_bottom = pred_top + pred_height
    gt_left = gt_bbox[0]
    gt_width = gt_bbox[2]
    gt_right = gt_left + gt_width
    gt_top = gt_bbox[1]
    gt_height = gt_bbox[3]
    gt_bottom = gt_top + gt_height
    intersection_left = 0
    intersection_right = 0
    intersection_top = 0
    intersection_bottom = 0
    if pred_left <= gt_left <= pred_right:
        intersection_left = gt_left
    elif gt_left <= pred_left <= gt_right:
        intersection_left = pred_left
    if pred_left <= gt_right <= pred_right:
        intersection_right = gt_right
    elif gt_left <= pred_right <= gt_right:
        intersection_right = pred_right
    if pred_top <= gt_top <= pred_bottom:
        intersection_top = gt_top
    elif gt_top <= pred_top <= gt_bottom:
        intersection_top = pred_top
    if pred_top <= gt_bottom <= pred_bottom:
        intersection_bottom = gt_bottom
    elif gt_top <= pred_bottom <= gt_bottom:
        intersection_bottom = pred_bottom
    intersection = (intersection_right - intersection_left) * (
        intersection_bottom - intersection_top
    )
    union = (
        (pred_right - pred_left) * (pred_bottom - pred_top)
        + (gt_right - gt_left) * (gt_bottom - gt_top)
        - intersection
    )
    iou = 0.0 if union <= 0 else float(intersection / union)
    return iou
