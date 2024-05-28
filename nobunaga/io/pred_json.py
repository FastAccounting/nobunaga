import json


class PredJson(object):
    def __init__(self, file_path: str):
        with open(file_path, "r") as json_file:
            pred_json = json.load(json_file)
        self._annotations = {}

        pred_json_list = []
        if type(pred_json) == list:
            pred_json_list = pred_json
        elif type(pred_json) == dict:
            for data_type, annotations in pred_json.items():
                if data_type == "annotations":
                    for annotation in annotations:
                        image_id = annotation.get("image_id")
                        for anno in annotation.get("segments_info", []):
                            anno["image_id"] = image_id
                            pred_json_list.append(anno)

        for annotation in pred_json_list:
            image_id = annotation.get("image_id")
            if self._annotations.get(image_id, "") == "":
                self._annotations[image_id] = []
            self._annotations[image_id].append(annotation)

    def get_annotations(self):
        return self._annotations

    def get_annotation_by_image_id(self, image_id: int):
        return self._annotations.get(image_id, [])
