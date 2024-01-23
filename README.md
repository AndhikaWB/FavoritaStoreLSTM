# Favorita Store LSTM
Sales prediction based on real data released by Favorita grocery store (check on [Kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)). The model were trained using multiple `fit`, where each fit contains data from specific store (up to 10 stores).

There's also custom sampling layer used in this model, which make use of bicubic interpolation rather than the default repeat (nearest interpolation) behavior from Keras `UpSampling1D` layer. The source code can be seen directly on the notebook (also for other custom layers).

Current model architecture:
``` python
model = keras.models.Sequential([
    Input(shape = (days_past, len(x_col))),
    RandomApply(
        TrainingOnly(AveragePooling1D(pool_size = 3, padding = 'same')),
        rate = 0.4, seed = 1234, name = 'avg_pooling_1'
    ),
    RandomApply(
        TrainingOnly(ImprovedUpSampling1D(size = 3, interpolation = 'bicubic')),
        rate = 0.4, seed = 1234, name = 'up_sampling_1'
    ),

    # --------------------

    LSTM(units = 256, return_sequences = True, name = 'lstm_1'),
    LSTM(units = 128, return_sequences = True, name = 'lstm_2'),
    LSTM(units = 256, return_sequences = True, name = 'lstm_3'),

    TimeDistributed(Dense(units = 128), name = 'dense_1'),
    PReLU(name = 'prelu_1'),
    Dropout(rate = 0.1, name = 'dropout_1'),

    TimeDistributed(Dense(units = 64), name = 'dense_2'),
    PReLU(name = 'prelu_2'),
    Dropout(rate = 0.1, name = 'dropout_2'),

    TimeDistributed(Dense(units = len(y_col)), name = 'dense_3')
])
```

## Todo
- Use different model for different store rather than training a generalized one
- Implement LTTB, and compare prediction result when average pooling vs LTTB
- Figure out how to make the model recognize holiday spike, or if not possible, use univariate LSTM or simpler algorithm (e.g. ARIMA)

## Screenshots
<details>
    <summary>Bicubic Upsampling</summary>
    <img src="misc/screenshot/screenshot_1.png">
</details>

<details>
    <summary>Sample Prediction</summary>
    <img src="misc/screenshot/screenshot_2.png">
</details>