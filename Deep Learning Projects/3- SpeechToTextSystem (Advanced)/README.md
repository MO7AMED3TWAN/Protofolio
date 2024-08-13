# 🌟 MTC-AIC

Welcome to the **MTC-AIC** repository! This project is dedicated to developing a system that manages and enhances the workflow for Arabic Speech-to-Text (ASR) using the DeepSpeech2 architecture. This system was built for the MTC AIC2 competition, adhering to strict guidelines that prohibited the use of any transformers or pre-trained models.

## 📜 Project Overview

In this project, we developed an end-to-end ASR model specifically tailored for Arabic, training it from scratch. The model was trained for nearly 20 days using a custom dataset comprising 50 recorded audio samples. Despite the challenges posed by the limited and noisy data, our objective was to achieve high accuracy in transcribing Arabic speech. This project documents the development process, challenges encountered, and the outcomes achieved in creating a competitive ASR system under these constraints.

## 📑 Table of Contents
- [🧠 Model Structure](#-model-structure)
- [⚙️ Installation](#%EF%B8%8F-installation)
- [🚀 Usage](#-usage)
- [📂 Files and Directories](#-files-and-directories)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [👤 Author](#-author)

## 🧠 Model Structure

DeepSpeech2 is a set of speech recognition models based on [Baidu DeepSpeech2](https://arxiv.org/abs/1512.02595). The architecture is summarized as follows:

![DeepSpeech2 architecture](./res/ds2.png)

The preprocessing part converts a raw audio waveform signal into a log-spectrogram of size (*N_timesteps*, *N_frequency_features*). *N_timesteps* depends on the original audio file’s duration, while *N_frequency_features* can be assigned in the model’s configuration file as the `num_audio_features` parameter.

The Deep Neural Network (DNN) component generates a probability distribution *P_t(c)* over vocabulary characters *c* for each time step *t*.

DeepSpeech2 is trained using CTC loss and consists of:
- Two convolutional layers:
  1. 32 channels, kernel size [11, 41], stride [2, 2]
  2. 32 channels, kernel size [11, 21], stride [1, 2]
- Five bidirectional GRU layers (size: 800)
- One fully connected layer (size: 1600)
- One projection layer (size: number of characters + 1 for CTC blank symbol, 29)

The model utilizes spectrograms with 160 frequency features (i.e., without frequency filtering).

## ⚙️ Installation

To set up the MTC-AIC project locally, follow these steps:

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/MO7AMED3TWAN/MTC-AIC.git
   cd MTC-AIC
   ```

2. **Create a virtual environment:**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download necessary models:**  
   Ensure that the DeepSpeech2 model weights are present in the `Models/` directory.

## 🚀 Usage

To use the MTC-AIC system:

1. **Run the main script:**  
   - Open `main.ipynb` in Jupyter Notebook or Jupyter Lab.
   - Follow the instructions in the notebook to train the model, perform inference, and evaluate performance on the test data.

2. **Customize the pipeline:**  
   - Modify the source code in `src/` to adapt the model training or inference process to your specific needs.

## 📂 Files and Directories

- **`main.ipynb`**: Main workflow notebook for the ASR system.
- **`Audio_Preprocessing.ipynb`**: Notebook for audio preprocessing.
- **`TranscriptCleaning&EDA.ipynb`**: Notebook for text correction.
- **`src/`**: Source code for the project.
- **`Models/`**: Trained DeepSpeech2 model weights and configuration files.
- **`Data/`**: Arabic audio dataset used for training (not provided due to MTC-AIC2 competition copyright restrictions).
- **`res/`**: Documentation images or PDFs.
- **`requirements.txt`**: Dependency file.

## 🤝 Contributing

We welcome contributions to the MTC-AIC project! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed explanation of your changes.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

**Mohamed Atwan**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/your-linkedin-url)

For any inquiries or further information, feel free to reach out via LinkedIn.