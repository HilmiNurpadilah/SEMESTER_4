import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

# Load data
data = pd.read_csv('prevalensi_balita_stunting_kabupatenkota.csv')

# Encode nama kabupaten/kota
label_encoder = LabelEncoder()
data['kode_kabupaten_kota'] = label_encoder.fit_transform(data['nama_kabupaten_kota'])

# Scale input
scaler = MinMaxScaler()
data[['tahun', 'kode_kabupaten_kota']] = scaler.fit_transform(data[['tahun', 'kode_kabupaten_kota']])

# Scale target (prevalensi)
output_scaler = MinMaxScaler()
data['prevalensi_balita_stunting'] = output_scaler.fit_transform(data[['prevalensi_balita_stunting']])

# Build model ANN
model = Sequential([
    Input(shape=(2,)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
X = data[['tahun', 'kode_kabupaten_kota']].values
y = data['prevalensi_balita_stunting'].values
model.fit(X, y, epochs=200, batch_size=16, verbose=1)

# Evaluate
y_true = output_scaler.inverse_transform(y.reshape(-1, 1))
y_pred = output_scaler.inverse_transform(model.predict(X))
mae = mean_absolute_error(y_true, y_pred)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Predict 2024 & 2025 untuk kabupaten Bandung (kode 3204 sesuai dataset)
bandung_kode = label_encoder.transform(['KABUPATEN BANDUNG'])[0]
pred_input = np.array([[2024, bandung_kode], [2025, bandung_kode]])
pred_scaled = scaler.transform(pred_input)
pred_output = model.predict(pred_scaled)
pred_final = output_scaler.inverse_transform(pred_output)

df_pred = pd.DataFrame(pred_final, columns=["Prediksi Kasus"])
df_pred["Tahun"] = [2024, 2025]
print(df_pred)

# Tampilkan grafik
plt.figure()
plt.plot(data['tahun'], output_scaler.inverse_transform(data['prevalensi_balita_stunting']), label='Data Aktual', marker='o')
plt.plot(df_pred["Tahun"], df_pred["Prediksi Kasus"], label='Prediksi', marker='o', linestyle='--')
plt.title("Prediksi Kasus Stunting untuk Kabupaten Bandung")
plt.xlabel("Tahun")
plt.ylabel("Prediksi Kasus (%)")
plt.ylim(0, 50)
plt.legend()
plt.grid()
plt.show()

# Simpan model
model.save("stunting_balita_ann.h5")
print("✅ Model disimpan sebagai 'stunting_balita_ann.h5'")
