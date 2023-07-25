import math
import platform

import numpy as np
import PIL
from PIL import ImageDraw, ImageFont


def write_label(image_path: str, new_file_path: str, bboxes: dict, col_size: int):
    images = []
    image_height = 0
    image_width = 0
    for title, bbox_list in bboxes.items():
        image = PIL.Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        score_format = ": {:.1f}"
        text_size = 16
        font = get_font(text_size)

        if len(bbox_list) > 0:
            for bbox in bbox_list:
                name = bbox[0]
                left = bbox[1]
                width = bbox[3]
                right = left + width
                top = bbox[2]
                height = bbox[4]
                bottom = top + height
                score = bbox[5]
                color = (0, 255, 0)
                if len(str(score)) > 0:
                    display_str = name + score_format.format(score)
                else:
                    display_str = name
                image = _draw_single_box(
                    image=image,
                    xmin=int(left),
                    ymin=int(top),
                    xmax=int(right),
                    ymax=int(bottom),
                    color=color,
                    display_str=display_str,
                    font=font,
                    width=2,
                    alpha=0.5,
                    fill=False,
                )

        draw = ImageDraw.Draw(image)
        draw.text((image.width / 2, 10), title, font=font, fill="#000000")
        image_height = image.height
        image_width = image.width
        images.append(image)

    row_count = math.ceil(len(bboxes) / col_size)
    merged_image = PIL.Image.new("RGB", (image_width * col_size, image_height * row_count))
    col_index = 0
    row_index = 0
    for image in images:
        merged_image.paste(image, (image_width * col_index, image_height * row_index))
        if col_index >= col_size - 1:
            col_index = 0
            row_index += 1
        else:
            col_index += 1
    merged_image.save(new_file_path)
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
