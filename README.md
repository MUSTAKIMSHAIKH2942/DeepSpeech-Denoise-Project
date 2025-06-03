
# 🧠 DeepSpeech Denoise Project

A deep learning-based speech denoising pipeline using a U-Net architecture. Modular code with Factory Pattern to dynamically create model, trainer, and inference components.

## 🏗️ Architecture

- `model.py`: U-Net based 1D ConvNet
- `factory.py`: Factory Design Pattern to manage object instantiation
- `train.py`: Training logic
- `infer.py`: Denoising inference
- `utils.py`: Audio helpers

## 🚀 How to Use

```bash
# Train (you'll need your dataloader logic)
python src/train.py

# Inference
python src/deep_speech_denoise_base.py
🔧 Setup

pip install torch torchaudio
🛠️ Why Factory Pattern?
Easily swap models or inference engines

Encapsulates complexity

Promotes extensibility

📁 Structure

DeepSpeechDenoiseProject/
├── notebooks/
├── src/
├── docs/
├── models/
├── README.md
└── .gitignore


---

### `.gitignore`

```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
*.swp
.env
*.wav
models/
