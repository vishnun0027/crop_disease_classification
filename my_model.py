from transformers import AutoImageProcessor
import torch
from transformers import AutoModelForImageClassification
from PIL import Image

class cropDiseaseModel:
    def __init__(self):
        self.checkpoint = "vishnun0027/Crop_Disease_model_1"
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.model = AutoModelForImageClassification.from_pretrained(self.checkpoint)

    def predict(self, image_path):
        image = Image.open(image_path)
        inputs = self.image_processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        disease = self.model.config.id2label[predicted_label]
        return disease
    