import json


class PredJson:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as json_file:
            cocojson = json.load(json_file)
        self.annotations = {}

        for annotation in cocojson:
            if self.annotations.get(annotation.get('image_id', ''), '') == '':
                self.annotations[annotation.get('image_id', '')] = []
            self.annotations[annotation.get('image_id', '')].append(annotation)

    def get_annotations(self):
        return self.annotations

    def get_annotation_by_image_id(self, image_id: int):
        return self.annotations.get(image_id, [])
