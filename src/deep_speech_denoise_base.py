from factory import DenoiseFactory
from utils import load_audio, save_audio

if __name__ == "__main__":
    model = DenoiseFactory.get_model()
    denoiser = DenoiseFactory.get_inference_engine(model)

    noisy_audio = load_audio("sample_noisy.wav")
    denoised_audio = denoiser.denoise(noisy_audio)
    save_audio("denoised_output.wav", denoised_audio)
