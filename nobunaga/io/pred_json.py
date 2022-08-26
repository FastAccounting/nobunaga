import json


class PredJson(object):
    def __init__(self, file_path: str):
        with open(file_path, "r") as json_file:
            cocojson = json.load(json_file)
        self._annotations = {}

        for annotation in cocojson:
            if self._annotations.get(annotation.get("image_id", ""), "") == "":
                self._annotations[annotation.get("image_id", "")] = []
            self._annotations[annotation.get("image_id", "")].append(annotation)

    def get_annotations(self):
        return self._annotations

    def get_annotation_by_image_id(self, image_id: int):
        return self._annotations.get(image_id, [])
