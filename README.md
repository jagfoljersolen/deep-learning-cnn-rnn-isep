# deep-learning-cnn-rnn-isep

CNN image classification (Fruits-360) and RNN time series forecasting (Jena Climate) in TensorFlow/Keras.

[Report (PDF)](report/APPROF_ADF_M1A_1252373_1252365.pdf)

## Structure

```
CNN/                 fruit classification
RNN/                 temperature forecasting
requirements.txt
```

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## CNN — Fruits-360

- Input: 100x100x3, rescale 1/255, `ImageDataGenerator`, batch size 32
- Split: 109,865 train / 27,356 val / 45,724 test (260 classes)
- Best custom model (Large CNN): Conv2D(64→128→256, 3x3, same) + BatchNorm + MaxPool2x2 blocks → Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(260, softmax)
- Optimizer: Adam, loss: categorical cross-entropy, early stopping (patience 5) + checkpointing

| Experiment | Test Acc. | Macro P | Macro R | Macro F1 |
|---|---|---|---|---|
| Baseline CNN | 0.94 | 0.94 | 0.94 | 0.94 |
| CNN + Augmentation | 0.95 | 0.96 | 0.95 | 0.95 |
| Small CNN (16/32/64) | 0.97 | 0.97 | 0.96 | 0.96 |
| Large CNN (64/128/256) | 0.97 | 0.97 | 0.97 | 0.97 |
| AveragePooling CNN | 0.94 | 0.95 | 0.94 | 0.94 |
| Stride/Padding CNN | 0.96 | 0.96 | 0.96 | 0.96 |
| MobileNetV2 (transfer) | 0.97 | 0.98 | 0.98 | 0.98 |

## RNN — Jena Climate

- Features: T, p, Tdew, rh, VPact, VPdef, wv, wd (cyclic-encoded wd + time)
- Split: 70/15/15 chronological, min-max scaled
- Sequence: 72-step (12h) / 144-step (24h) / 288-step (48h) input windows, 144-step (24h) horizon
- Units: 64, batch size 256, Adam, loss: MSE, early stopping (patience 5, max 30 epochs)
- Metric: RMSLE

| Model | Window | RMSLE | Epochs |
|---|---|---|---|
| LSTM | 24h | 0.0902 | 8 |
| LSTM | 48h | 0.0904 | 8 |
| SimpleRNN | 12h | 0.0907 | 19 |
| LSTM | 12h | 0.0909 | 11 |
| SimpleRNN | 24h | 0.0913 | 12 |
| SimpleRNN | 48h | 0.0932 | 8 |

Best model: LSTM(64) → Dense(1), 24h window, RMSLE 0.0902.

## Requirements

TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib (see `requirements.txt`).