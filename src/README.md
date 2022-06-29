# Modified TIDE

## Installation
```shell
pip install git+https://github.com/FastAccounting/nobunaga
```

# Usage


## show error analysis
The error analysis tool oda nobunaga (object detection analysis) has been extended to output errors in each category.
See `show_errors2.py` to use the tool.



```bash
  python3 show_errors2.py \
    -p coco_instances_results.json \
    -g instances_val.json \
    -i 0.5 \
    -c 0.7 \
    -d path/to/image_dir

  or

  python3 show_errors2.py \
    --pred coco_instances_results.json \
    --gt instances_val.json \
    --iou_threshold 0.5 \
    --confidence_threshold 0.7 \
    --image_dir path/to/image_dir

```

You can get below files
```
error distribution
- error_type_confusion_matrix.png  

class error matrix
- class_error_confusion_matrix.png  

error file list
- error_file_names.csv

error files with error label
- _error/
```

## show comparison betweeen ground truth and prediction
```bash
python3 show_images.py \
  -p coco_instances_results.json \
  -g instances_val.json \
  -i 0.5 \
  -c 0.7 \
  -d path/to/image_dir

or

python3 show_images.py \
  --pred coco_instances_results.json \
  --gt instances_val.json \
  --iou_threshold 0.5 \
  --confidence_threshold 0.7 \
  --image_dir path/to/image_dir

```

You can get below files
```
all labeled files
- _normal/
```
