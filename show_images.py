import sys
import argparse
from src import Evaluator, PredJson, GtJson, ImagePrinter


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', '-g', type=str, default='')
    parser.add_argument('--pred', '-p', type=str, default='')
    parser.add_argument('--image_dir', '-d', type=str, default='')
    parser.add_argument('--iou_threshold', '-i', type=float, default=0.5)
    parser.add_argument('--confidence_threshold', '-c', type=float, default=0.7)
    parser.add_argument('--model_name', '-m', type=str, default='')
    parser.add_argument('--normalize', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    if len(args.gt) == 0:
        print()
        print('gt(coco json path) is required.')
        print()
        exit(0)
    if len(args.pred) == 0:
        print()
        print('pred(coco json path) is required.')
        print()
        exit(0)
    if len(args.image_dir) == 0:
        print()
        print('image_dir is required.')
        print()
        exit(0)
    return args


def main():
    args = arg()

    # read coco json file
    gt = GtJson(args.gt)
    pred = PredJson(args.pred)
    evaluation = Evaluator(gt, pred, args.iou_threshold, args.confidence_threshold)

    printer = ImagePrinter(args.model_name, gt.get_categories(), evaluation)
    printer.output_labeled_images(args.image_dir)


if __name__ == '__main__':
    main()
