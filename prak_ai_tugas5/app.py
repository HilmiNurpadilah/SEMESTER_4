from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)

# Load data dan encoder
df = pd.read_csv("prevalensi_balita_stunting_kabupatenkota.csv")
le = LabelEncoder()
df["kode_kabupaten_kota"] = le.fit_transform(df["nama_kabupaten_kota"])

X = df[["tahun", "kode_kabupaten_kota"]]
y = df["prevalensi_balita_stunting"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

output_scaler = MinMaxScaler()
y_scaled = output_scaler.fit_transform(y.values.reshape(-1, 1))

# Load model
model = load_model("stunting_balita_ann.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    kabupaten_list = sorted(df["nama_kabupaten_kota"].unique())
    plot_url = None

    if request.method == "POST":
        tahun = int(request.form["tahun"])
        kabupaten = request.form["kabupaten"]
        kode = le.transform([kabupaten])[0]

        input_data = np.array([[tahun, kode]])
        scaled_input = scaler.transform(input_data)
        pred_scaled = model.predict(scaled_input)
        hasil = round(output_scaler.inverse_transform(pred_scaled)[0][0], 2)

        # Plotting the graph
        plt.figure()
        plt.plot(df['tahun'], output_scaler.inverse_transform(y_scaled), label='Data Aktual', marker='o')
        plt.scatter([tahun], [hasil], color='red', label='Prediksi', zorder=5)
        plt.title(f"Prediksi Kasus Stunting untuk {kabupaten}")
        plt.xlabel("Tahun")
        plt.ylabel("Prediksi Kasus (%)")
        plt.ylim(0, 50)
        plt.legend()
        plt.grid()

        # Save plot to a string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close()

    return render_template("index.html", kabupaten_list=kabupaten_list, hasil=hasil, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
