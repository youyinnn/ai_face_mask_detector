import Training
import Models
import Application


class MaskDetection():

    def __init__(self, model_path):
        self.model = Training.load_model(Models.Base_CNN, model_path)

    def predict(self, imgs):
        return self.model(imgs)
