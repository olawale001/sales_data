from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from google.colab import drive


drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/sales_data.csv', encoding='latin1')
data.columns = data.columns.str.strip()

print(data.columns)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[["QUANTITYORDERED"]].values)

def create_sequence(data, time_steps=10):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)    

time_steps = 10
x, y = create_sequence(data_scaled, time_steps)
x_train, x_test = x[:-100], x[-100]
y_train, y_test = y[:-100], y[-100]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(50),
    Dense(1)
])
model.summary()
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x_train, y_train, epochs= 20, batch_size=32)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

plt.plot(data.index[-100:], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Sales')
plt.plot(data.index[-100:], predictions, label="Predicted Sales", linestyles='dashed')
plt.legend()
plt.show()