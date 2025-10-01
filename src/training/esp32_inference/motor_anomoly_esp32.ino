
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include "motor_autoencoder_tflite.h"

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define FEATURE_COUNT 8
#define TENSOR_ARENA_SIZE (10 * 1024)
const float MEANS[FEATURE_COUNT] = {0.07845951, 0.01078413, 1.0710341, -1.1145038, 0.1545188, 0.589188, 39.126278, 4159.0782};
const float STDS[FEATURE_COUNT]  = {0.41568548, 0.39861843, 0.21081677, 8.445867, 9.9282167, 7.5683938, 0.3665173, 344.50074};

float ANOMALY_THRESHOLD = 0.8f;

#define MS2_TO_G (1.0f / 9.80665f)
#define RAD_TO_DEG 57.2957795f
#ifndef PULSES_PER_REV
#define PULSES_PER_REV 1
#endif
#define ENABLE_AUTO_CALIBRATION 1
#define CALIBRATION_SAMPLES 200
#define DEBUG_PRINT 0

Eloquent::TF::Sequential<2, TENSOR_ARENA_SIZE> tf;

Adafruit_MPU6050 mpu;
OneWire oneWire(4);
DallasTemperature temp_sensor(&oneWire);
volatile unsigned long hall_count = 0;


#define SDA_PIN 21
#define SCL_PIN 22
#define HALL_PIN 32

void IRAM_ATTR hallISR() { hall_count++; }

unsigned long last_inference_time = 0;
const unsigned long INFERENCE_INTERVAL_MS = 500; // 0.5s
#if ENABLE_AUTO_CALIBRATION
bool calibrated = false;
int calib_n = 0;
float calib_mean = 0.0f;
float calib_m2 = 0.0f;
#endif

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!mpu.begin()) Serial.println("MPU6050 NOT FOUND");
  temp_sensor.begin();
  pinMode(HALL_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN), hallISR, RISING);

  tf.setNumInputs(FEATURE_COUNT);
  tf.setNumOutputs(FEATURE_COUNT);
  tf.resolver.AddFullyConnected();
  tf.resolver.AddRelu();

  if (!tf.begin(motor_autoencoder_tflite).isOk()) {
    Serial.println("TFLite init failed:");
    Serial.println(tf.exception.toString());
    while (1);
  }

  Serial.println("Setup complete!");
}

void loop() {
  const unsigned long now = millis();
  if (now - last_inference_time < INFERENCE_INTERVAL_MS) return;
  last_inference_time = now;

  // Read sensors
  sensors_event_t accel, gyro, tmp_evt;
  mpu.getEvent(&accel, &gyro, &tmp_evt);

  temp_sensor.requestTemperatures();
  float tempC = temp_sensor.getTempCByIndex(0);
  if (tempC < -40 || tempC > 125 || tempC == -127) {
    
    tempC = MEANS[6];
  }

  noInterrupts();
  unsigned long hall_snapshot = hall_count;
  hall_count = 0;
  interrupts();


  float ax_g = accel.acceleration.x * MS2_TO_G;
  float ay_g = accel.acceleration.y * MS2_TO_G;
  float az_g = accel.acceleration.z * MS2_TO_G;
  float gx_dps = gyro.gyro.x * RAD_TO_DEG;
  float gy_dps = gyro.gyro.y * RAD_TO_DEG;
  float gz_dps = gyro.gyro.z * RAD_TO_DEG;

  float rpm = (hall_snapshot * 60000.0f) / (PULSES_PER_REV * INFERENCE_INTERVAL_MS);

  float raw[FEATURE_COUNT] = { ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, tempC, rpm };

  float input_features[FEATURE_COUNT];
  for (int i = 0; i < FEATURE_COUNT; i++) {
    input_features[i] = (raw[i] - MEANS[i]) / STDS[i];
    // Optional: clamp extreme outliers that destabilize MSE
    if (input_features[i] > 6) input_features[i] = 6;
    if (input_features[i] < -6) input_features[i] = -6;
  }

  if (!tf.predict(input_features).isOk()) {
    Serial.println("Invoke failed");
    Serial.println(tf.exception.toString());
    return;
  }

  float mse = 0;
  for (int i = 0; i < FEATURE_COUNT; i++) {
    float diff = input_features[i] - tf.output(i);
    mse += diff * diff;
  }
  mse /= FEATURE_COUNT;

#if DEBUG_PRINT
  Serial.print("raw: ");
  for (int i = 0; i < FEATURE_COUNT; i++) { Serial.print(raw[i], 4); Serial.print(i < FEATURE_COUNT - 1 ? "," : " "); }
  Serial.print(" | z: ");
  for (int i = 0; i < FEATURE_COUNT; i++) { Serial.print(input_features[i], 3); Serial.print(i < FEATURE_COUNT - 1 ? "," : " "); }
  Serial.print(" | out: ");
  for (int i = 0; i < FEATURE_COUNT; i++) { Serial.print(tf.output(i), 3); Serial.print(i < FEATURE_COUNT - 1 ? "," : " "); }
  Serial.println();
#endif

#if ENABLE_AUTO_CALIBRATION
  if (!calibrated) {
    calib_n++;
    float delta = mse - calib_mean;
    calib_mean += delta / calib_n;
    calib_m2 += delta * (mse - calib_mean);

    if (calib_n >= CALIBRATION_SAMPLES) {
      float variance = (calib_n > 1) ? (calib_m2 / (calib_n - 1)) : 0.0f;
      float stddev = sqrtf(variance);
      ANOMALY_THRESHOLD = calib_mean + 3.0f * stddev;
      calibrated = true;
      Serial.print("Calibration done. Mean MSE="); Serial.print(calib_mean, 6);
      Serial.print(" Std="); Serial.print(stddev, 6);
      Serial.print(" -> Threshold="); Serial.println(ANOMALY_THRESHOLD, 6);
    } else {
      Serial.print("Calibrating ["); Serial.print(calib_n); Serial.print("/"); Serial.print(CALIBRATION_SAMPLES); Serial.print("] MSE="); Serial.println(mse, 6);
    }
    return; 
  }
#endif

  Serial.print("MSE: "); Serial.print(mse, 6);
  if (mse > ANOMALY_THRESHOLD) Serial.println(" -> !!! ANOMALY DETECTED !!!");
  else Serial.println(" -> Normal");
}