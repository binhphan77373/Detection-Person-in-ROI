from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image, classes=[0]):
        results = self.model.predict(source=image, classes=classes)
        return results[0].boxes.xyxy.cpu()
