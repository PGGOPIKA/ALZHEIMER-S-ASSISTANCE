# classify_gru.py

import sys
import json
import numpy as np
import tensorflow as tf

# Load the saved GRU model
model = tf.keras.models.load_model("gru_model.h5")

# Read JSON input from stdin (sent from your Node.js server)
raw_input = sys.stdin.read()
sensor_data = json.loads(raw_input)

# Expected input keys: accX, accY, accZ, gyroX, gyroY, gyroZ, heartRate, spo2, bodyTemp, gsr
input_vector = np.array([
    sensor_data['accX'], sensor_data['accY'], sensor_data['accZ'],
    sensor_data['gyroX'], sensor_data['gyroY'], sensor_data['gyroZ'],
    sensor_data['heartRate'], sensor_data['spo2'],
    sensor_data['bodyTemp'], sensor_data['gsr']
])

# Repeat the vector to simulate a sequence of 20 time steps
sequence = np.tile(input_vector, (20, 1))  # shape: (20, 10)
sequence = sequence.reshape(1, 20, 10)     # shape: (1, 20, 10)

# Run prediction
activity_pred, emotion_pred, sleep_pred = model.predict(sequence)

# Format predictions into structured JSON
result = {
    "activity": [
        {
            "time": "now",
            "walking": float(activity_pred[0][0]),
            "resting": float(activity_pred[0][1]),
            "falling": float(activity_pred[0][2]),
            "exercising": float(activity_pred[0][3])
        }
    ],
    "emotion": {
        "relaxed": float(emotion_pred[0][0]),
        "stressed": float(emotion_pred[0][1]),
        "motivated": float(emotion_pred[0][2]),
        "anxious": float(emotion_pred[0][3])
    },
    "sleep": {
        "deep": float(sleep_pred[0][0]),
        "light": float(sleep_pred[0][1]),
        "awake": float(sleep_pred[0][2])
    }
}

# Print JSON to stdout (Node.js backend will read this)
print(json.dumps(result))
