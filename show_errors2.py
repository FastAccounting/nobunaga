import argparse
import src.Constants as Const
from src import Evaluator, GtJson, ImagePrinter, PredJson


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", "-g", type=str, default="")
    parser.add_argument("--pred", "-p", type=str, default="")
    parser.add_argument("--image_dir", "-d", type=str, default="")
    parser.add_argument("--iou_threshold", "-i", type=float, default=0.5)
    parser.add_argument("--confidence_threshold", "-c", type=float, default=0.7)
    parser.add_argument("--model_name", "-m", type=str, default="")
    parser.add_argument("--normalize", type=bool, default=False)
    args = parser.parse_args()
    print(args)
    if len(args.gt) == 0:
        print()
        print("gt(coco json path) is required.")
        print()
        exit(0)
    if len(args.pred) == 0:
        print()
        print("pred(coco json path) is required.")
        print()
        exit(0)
    if len(args.image_dir) == 0:
        print()
        print("image_dir is required.")
        print()
        exit(0)
    return args


def main():
    args = arg()

    # read coco json file
    gt = GtJson(args.gt)
    pred = PredJson(args.pred)
    evaluation = Evaluator(gt, pred, args.iou_threshold, args.confidence_threshold)
    categories = gt.get_categories()

    printer = ImagePrinter(args.model_name, categories, evaluation)
    printer.output_error_type_matrix(args.normalize, args.model_name)
    # printer.output_per_accuracy_and_errors()
    printer.output_confusion_matrix(args.normalize, args.model_name)
    printer.output_error_files(args.image_dir, Const.ERROR_TYPE_CLASS)
    printer.output_error_files(args.image_dir, Const.ERROR_TYPE_LOCATION)
    printer.output_error_files(args.image_dir, Const.ERROR_TYPE_DUPLICATE)
    printer.output_error_files(args.image_dir, Const.ERROR_TYPE_BACKGROUND)
    printer.output_error_files(args.image_dir, Const.ERROR_TYPE_MISS)
    printer.output_error_files(args.image_dir, Const.ERROR_TYPE_BOTH)


if __name__ == "__main__":
    main()
