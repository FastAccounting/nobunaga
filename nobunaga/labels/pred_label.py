class PredLabel(object):
    def __init__(
        self,
        image_id: int,
        image_name: str,
        index: int,
        category_id: int,
        bbox: list,
        confidence: float,
    ):
        self._image_id = image_id
        self._image_name = image_name
        self._index = index
        self._category_id = category_id
        self._bbox = bbox
        self._confidence = confidence

    def get_image_id(self):
        return self._image_id

    def get_image_name(self):
        return self._image_name

    def get_index(self):
        return self._index

    def get_category_id(self):
        return self._category_id

    def get_bbox(self):
        return self._bbox

    def get_confidence(self):
        return self._confidence
