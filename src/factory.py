from model import UNetDenoiseModel
from train import Trainer
from infer import InferenceEngine

class DenoiseFactory:
    @staticmethod
    def get_model():
        return UNetDenoiseModel()

    @staticmethod
    def get_trainer(model):
        return Trainer(model)

    @staticmethod
    def get_inference_engine(model):
        return InferenceEngine(model)
