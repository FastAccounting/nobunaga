import json


class GtJson:

    def __init__(self, file_path: str):
        with open(file_path, 'r') as json_file:
            cocojson = json.load(json_file)
        self.images = {}
        self.annotations = {}
        self.categories = {}

        for image in cocojson['images']:
            self.images[image.get('id', -1)] = image

        for annotation in cocojson['annotations']:
            if self.annotations.get(annotation.get('image_id', ''), '') == '':
                self.annotations[annotation.get('image_id', '')] = []
            self.annotations[annotation.get('image_id', '')].append(annotation)

        for category in cocojson['categories'] if 'categories' in cocojson else []:
            self.categories[category.get('id', -1)] = category.get('name', '')

    def get_image_dict(self):
        image_dict = {}
        for image_id, image in self.images.items():
            image_dict[image_id] = {
                'images': image,
                'annotations': self.annotations.get(image_id, []),
                'categories': self.categories,
            }
        return image_dict

    def get_images(self):
        return self.images

    def get_annotations(self):
        return self.annotations

    def get_categories(self):
        return self.categories

    def get_image_by_image_id(self, image_id: int):
        return self.images.get(image_id, {})

    def get_annotation_by_image_id(self, image_id: int):
        return self.annotations.get(image_id, [])

    def get_category_by_image_id(self, image_id: int):
        return self.categories.get(image_id, {})

    def get_image_file_name(self, image_id: int):
        return self.images.get(image_id, '').get('file_name', '')

    def get_category_name(self, category_id: int):
        return self.categories.get(category_id, '').get('name', '')

