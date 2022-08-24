import argparse

import nobunaga.constants as Const
#from nobunaga import Evaluator, GtJson, ImagePrinter, PredJson
from .evaluator import Evaluator
from .gt_json import GtJson
from .image_printer import ImagePrinter
from .pred_json import PredJson


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", "-g", type=str, default="", required=True)
    parser.add_argument("--pred", "-p", type=str, default="", required=True)
    parser.add_argument("--image_dir", "-d", type=str, default="", required=True)
    parser.add_argument("--iou_threshold", "-i", type=float, default=0.5)
    parser.add_argument("--confidence_threshold", "-c", type=float, default=0.7)
    parser.add_argument("--model_name", "-m", type=str, default="")
    parser.add_argument("--normalize", type=bool, default=False)
    args = parser.parse_args()
    print(args)

    return args


def main():
    args = arg()

    # read coco json file
    gt = GtJson(args.gt)
    pred = PredJson(args.pred)
    evaluation = Evaluator(gt, pred, args.iou_threshold, args.confidence_threshold)
    categories = gt.get_categories()

    printer = ImagePrinter(args.model_name, categories, evaluation)
    printer.output_error_summary()
    printer.output_error_type_detail(args.normalize, mode=["confusion_matrix", "strip"])
    printer.output_confusion_matrix(args.normalize)
    for error_type in Const.MAIN_ERRORS:
        printer.output_error_files(args.image_dir, error_type)


if __name__ == "__main__":
    main()
