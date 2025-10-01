# Save as tflite_to_header.py and run: python tflite_to_header.py

motor_predictive = "motor_autoencoder.tflite"

with open(motor_predictive, "rb") as f:
    data = f.read()

with open("motor_autoencoder_tflite.h", "w") as f:
    f.write("const unsigned char motor_autoencoder_tflite[] = {")
    f.write(",".join(str(b) for b in data))
    f.write("};\nunsigned int motor_autoencoder_tflite_len = %d;\n" % len(data))