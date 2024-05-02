# Physics-Informed LSTM
Code for the publication *"Physics-Informed Long-Short Term Memory Neural Network Performance on Holloman High-Speed Test Track Sled Study"* from the proceedings of the ASME 2022 Fluids Engineering Division Summer Meeting.

Paper available [here](https://www.researchgate.net/publication/364073635_Physics-Informed_Long-Short_Term_Memory_Neural_Network_Performance_on_Holloman_High-Speed_Test_Track_Sled_Study)

# Dataset
The raw CSV files (21GBs) and the split .npy files for the training (562MBs) and validation dataset (563MBs) can be provided when inquired.

# Installation
1. Install pyenv and use it to install Python 3.10, ```pyenv install 3.10```
2. Install poetry
3. Run ```poetry install```
  - If poetry install seems stuck you might have to run ```export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring``` in your shell due to a bug (https://github.com/python-poetry/poetry/issues/8623)
4. To verify your torch installation is using CUDA, run ```poetry run python dev_test_installation.py```

# Running
1. Set-up the experiment details/configuration in
> physics_lstm / config / paper.yaml

2. Run the experiment
```
cd physics_lstm
poetry run python lstm_pytorch.py
```