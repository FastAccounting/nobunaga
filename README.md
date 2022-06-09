# Modified TIDE

## Installation
```shell
pip install git+https://github.com/Swall0w/tide
```

# Usage
The error analysis tool TIDE has been extended to output errors in each category.
See `show_errors.py` to use the tool.
In addition to the typical TIDE results, each class's confusion matrix and error analysis are output.


```bash
  python show_errors.py --annotation examples/instances_val2017.json --result examples/coco_instances_results.json
```

![Confusion Matrix](examples/coco_result.png)

![Per-class information](examples/per_class_info.png)
