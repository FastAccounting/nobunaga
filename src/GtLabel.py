class GtLabel:
    def __init__(self, image_id: int, image_name: str, index: int, category_id: int, bbox: list):

        self.image_id = image_id
        self.image_name = image_name
        self.index = index
        self.category_id = category_id
        self.bbox = bbox

    def get_image_id(self):
        return self.image_id

    def get_image_name(self):
        return self.image_name

    def get_index(self):
        return self.index

    def get_category_id(self):
        return self.category_id

    def get_bbox(self):
        return self.bbox
