import torch

class InferenceEngine:
    def __init__(self, model):
        self.model = model.eval()

    def denoise(self, audio_tensor):
        with torch.no_grad():
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            output = self.model(audio_tensor)
            return output.squeeze().numpy()
