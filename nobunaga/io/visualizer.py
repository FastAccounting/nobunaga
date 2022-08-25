import platform

import numpy as np
import PIL
from PIL import ImageDraw, ImageFont


def write_label(
    image_path: str, new_file_path: str, pred_bboxes: list, gt_bboxes: list, is_mac: bool
):
    pred_image = PIL.Image.open(image_path)
    if pred_image.mode != "RGB":
        pred_image = pred_image.convert("RGB")
    score_format = ": {:.1f}"
    width = 2
    alpha = 0.5
    text_size = 16
    fill = False
    font = get_font(text_size)

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

    gt_image = PIL.Image.open(image_path)
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

    dst = PIL.Image.new("RGB", (pred_image.width + gt_image.width, pred_image.height))
    dst.paste(pred_image, (0, 0))
    dst.paste(gt_image, (pred_image.width, 0))
    dst.save(new_file_path)
    return new_file_path


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
