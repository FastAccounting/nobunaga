import os
import platform

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Util:
    @staticmethod
    def print_table(rows: list, title: str = None):
        print()
        # Get all rows to have the same number of columns
        max_cols = max([len(row) for row in rows])
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # Compute the text width of each column
        col_widths = [
            max([len(rows[i][col_idx]) for i in range(len(rows))])
            for col_idx in range(len(rows[0]))
        ]

        divider = "--" + ("---".join(["-" * w for w in col_widths])) + "-"
        thick_divider = divider.replace("-", "=")

        if title:
            left_pad = (len(divider) - len(title)) // 2
            print(("{:>%ds}" % (left_pad + len(title))).format(title))

        print(thick_divider)
        for row in rows:
            # Print each row while padding to each column's text width
            print(
                "  "
                + "   ".join(
                    [
                        ("{:>%ds}" % col_widths[col_idx]).format(row[col_idx])
                        for col_idx in range(len(row))
                    ]
                )
                + "  "
            )
            if row == rows[0]:
                print(divider)
        print(thick_divider)
        print()

    @staticmethod
    def calculate_ious(gt_bboxes: list, pred_bboxes: list):
        iou_matrix = []
        for pred_bbox in pred_bboxes:
            iou_row = []
            for gt_bbox in gt_bboxes:
                iou = Util.calculate_iou(gt_bbox, pred_bbox)
                iou_row.append(iou)
            iou_matrix.append(iou_row)
        return np.array(iou_matrix)

    @staticmethod
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

    @staticmethod
    def get_file_name(file_path):
        try:
            return (
                os.path.splitext(os.path.basename(file_path))[0]
                + os.path.splitext(os.path.basename(file_path))[1]
            )
        except Exception as e:
            print(e)
            return ""

    @staticmethod
    def get_file_name_only(file_path):
        try:
            return os.path.splitext(os.path.basename(file_path))[0]
        except Exception as e:
            print(e)
            return ""

    @staticmethod
    def get_directory(file_path):
        try:
            return os.path.dirname(file_path)
        except Exception as e:
            print(e)
            return ""

    @staticmethod
    def write_label(
        image_path: str, new_file_path: str, pred_bboxes: list, gt_bboxes: list, is_mac: bool
    ):
        pred_image = Image.open(image_path)
        if pred_image.mode != "RGB":
            pred_image = pred_image.convert("RGB")
        score_format = ": {:.1f}"
        width = 2
        alpha = 0.5
        text_size = 16
        fill = False
        font = Util.get_font(text_size)

        if len(pred_bboxes) > 0:
            for bbox in pred_bboxes:
                name = bbox[0]
                pred_left = bbox[1]
                pred_width = bbox[3]
                pred_right = pred_left + pred_width
                pred_top = bbox[2]
                pred_height = bbox[4]
                pred_bottom = pred_top + pred_height
                score = bbox[5]
                color = (0, 255, 0)
                display_str = name + score_format.format(score)

                pred_image = _draw_single_box(
                    image=pred_image,
                    xmin=pred_left,
                    ymin=pred_top,
                    xmax=pred_right,
                    ymax=pred_bottom,
                    color=color,
                    display_str=display_str,
                    font=font,
                    width=width,
                    alpha=alpha,
                    fill=fill,
                )

        pred_draw = ImageDraw.Draw(pred_image)
        pred_draw.text((pred_image.width / 2, 10), "Pred", font=font, fill="#000000")

        gt_image = Image.open(image_path)
        if gt_image.mode != "RGB":
            gt_image = gt_image.convert("RGB")

        if len(gt_bboxes) > 0:
            for bbox in gt_bboxes:
                name = bbox[0]
                gt_left = bbox[1]
                gt_width = bbox[3]
                gt_right = gt_left + gt_width
                gt_top = bbox[2]
                gt_height = bbox[4]
                gt_bottom = gt_top + gt_height
                score = bbox[5]
                color = (0, 255, 0)
                display_str = name

                gt_image = _draw_single_box(
                    image=gt_image,
                    xmin=gt_left,
                    ymin=gt_top,
                    xmax=gt_right,
                    ymax=gt_bottom,
                    color=color,
                    display_str=display_str,
                    font=font,
                    width=width,
                    alpha=alpha,
                    fill=fill,
                )

        gt_draw = ImageDraw.Draw(gt_image)
        gt_draw.text((gt_image.width / 2, 10), "GT", font=font, fill="#000000")

        dst = Image.new("RGB", (pred_image.width + gt_image.width, pred_image.height))
        dst.paste(pred_image, (0, 0))
        dst.paste(gt_image, (pred_image.width, 0))
        dst.save(new_file_path)
        return new_file_path

    @staticmethod
    def get_font(text_size: int):
        font_path = "/Library/Fonts/Arial Unicode.ttf"
        if platform.system() == "Linux":
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        font = ImageFont.truetype(font_path, text_size)
        return font


def _draw_single_box(
    image,
    xmin,
    ymin,
    xmax,
    ymax,
    color=(0, 255, 0),
    display_str=None,
    font=None,
    width=2,
    alpha=0.5,
    fill=False,
):

    draw = ImageDraw.Draw(image, mode="RGBA")
    left, right, top, bottom = xmin, xmax, ymin, ymax
    alpha_color = color + (int(255 * alpha),)
    draw.rectangle(
        [(left, top), (right, bottom)],
        outline=color,
        fill=alpha_color if fill else None,
        width=width,
    )

    if display_str:
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)

        box_height = ymax - ymin
        if text_height < box_height / 3:
            text_position = bottom
        else:
            text_position = top

        draw.rectangle(
            xy=[
                (left + width, text_position - text_height - 2 * margin - width),
                (left + text_width + width, text_position - width),
            ],
            fill=alpha_color,
        )
        draw.text(
            (left + margin + width, text_position - text_height - margin - width),
            display_str,
            fill="black",
            font=font,
        )

    return image
