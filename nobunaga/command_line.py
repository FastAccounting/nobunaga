import argparse
import os.path

import nobunaga.constants as Const
from nobunaga.evaluator import Evaluator
from nobunaga.image_printer import ImagePrinter
from nobunaga.io import GtJson, PredJson


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", "-g", type=str, default="test/jsons/gt_coco.json", required=False)
    parser.add_argument("--pred", "-p", type=str, default="test/jsons/pred_coco.json", required=False)
    parser.add_argument("--image_dir", "-d", type=str, default="test/images/", required=False)
    parser.add_argument("--iou_threshold", "-i", type=float, default=0.5)
    parser.add_argument("--confidence_threshold", "-c", type=float, default=0.7)
    parser.add_argument("--model_name", "-m", type=str, default="")
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--output_image", "-o", default=True)
    args = parser.parse_args()

    error_args_name = ""
    error_args_value = ""
    if not os.path.exists(args.gt):
        error_args_name = "gt"
        error_args_value = args.gt
    if not os.path.exists(args.pred):
        error_args_name = "pred"
        error_args_value = args.pred
    if not os.path.exists(args.image_dir):
        error_args_name = "image_dir"
        error_args_value = args.image_dir
    if error_args_value != "":
        print("'{error_args_name}' : '{error_args}' does not exist.".format(
            error_args_name=error_args_name,
            error_args=error_args_value
        ))
        exit()

    if not os.path.isdir(args.image_dir):
        error_args_name = "image_dir"
        error_args_value = args.image_dir
    if error_args_value != "":
        print("'{error_args_name}' : '{error_args}' have to be directory.".format(
            error_args_name=error_args_name,
            error_args=error_args_value
        ))
        exit()

    for arg_name, value in vars(args).items():
        print(f"{arg_name.ljust(21)}: {value}")
    return args


def main():
    args = arg()

    # read coco json file
    gt = GtJson(args.gt)
    pred = PredJson(args.pred)
    evaluation = Evaluator(gt, pred, args.iou_threshold, args.confidence_threshold)
    categories = gt.get_categories()

    printer = ImagePrinter(args.model_name, categories, evaluation, args.image_dir)
    printer.output_error_summary()
    printer.output_error_type_detail(args.normalize, mode=["confusion_matrix", "strip"])
    printer.output_correction_distance_csv_per_file()
    printer.output_correction_distance_csv_per_label()
    printer.output_confusion_matrix(args.normalize)

    # if you set argument -o you can output error images.
    if args.output_image:
        printer.output_correction_distance_files()
        for error_type in Const.MAIN_ERRORS:
            printer.output_error_files(error_type)


if __name__ == "__main__":
    main()
