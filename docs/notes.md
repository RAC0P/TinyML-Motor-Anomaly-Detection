📘 Project Notes – TinyML Motor Anomaly Detection
🛠 Hardware Setup

We used a 12V DC motor with the following sensors mounted on it:

A3144 Hall Effect Sensor → Measures motor RPM (rotational speed).

MPU6050 (Accelerometer + Gyroscope) → Captures vibration and motion data.

DS18B20 Digital Temperature Sensor → Measures motor surface temperature.

All sensors were interfaced with an ESP32 development board.
The motor was powered with a stable 12V DC supply.

🔄 Data Collection Workflow

Connected sensors to ESP32 and streamed values over serial.

Logged raw sensor readings into .csv files.

Organized datasets into:

dataset_summary.csv → Key statistics of the collected dataset.

motor_Rh.csv → Raw time-series sensor values.

autoencoder_results.csv → Model inference results.

🧠 Model Training & Conversion

Training

Data was preprocessed and normalized.

An autoencoder was trained using TensorFlow in Python.

Training script: src/training/train_autoencoder.py.

Model Conversion

The trained model was exported to TensorFlow Lite.

A helper script src/training/tflite_to_header.py converted the .tflite model into a .h C header file (C array format).

Deployment

The header file (motor_autoencoder_tflite.h) was added into the ESP32 inference code (src/esp32_inference/autoencoder_inference.ino).

The ESP32 runs the model fully on-device.

📊 Results & Validation

The autoencoder learned the normal operating pattern of the motor.

When anomalies (unusual vibration, heat, or RPM changes) were introduced, the model flagged them.

Plots (in figs/) show:

Motor anomaly detection (motor_anomaly_plot.png)

Correlation heatmap (correlation_heatmap.png)

MSE anomaly distribution (mse_histogram.png)

Quantitative results are summarized in autoencoder_results.csv.

⚡ Challenges & Solutions

Memory Limitations: ESP32 has limited RAM; we optimized model size by reducing neurons and quantizing the model.

Sensor Noise: Added smoothing/normalization during preprocessing.

Real-Time Constraints: Balanced sampling rate to avoid buffer overflow while keeping detection accurate.

📌 Future Improvements

Add support for edge logging (ESP32 saving anomalies to SD card).

Explore classification models for fault type detection (bearing fault, overheating, etc.).

Integrate with MQTT or BLE for IoT applications.