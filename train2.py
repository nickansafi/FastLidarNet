#Requirement Library
import os
from sklearn.utils import shuffle
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

from multi_res import (
    ConditionalBatchNorm, MultiResModel, calibrate_bn, export_tflite
)

# Check GPU availability
gpu_available = tf.test.is_gpu_available()
print('GPU AVAILABLE:', gpu_available)
typestore = get_typestore(Stores.ROS1_NOETIC)
typestore.register(get_types_from_msg(
    "float32 steering_angle\nfloat32 steering_angle_velocity\n"
    "float32 speed\nfloat32 acceleration\nfloat32 jerk",
    "ackermann_msgs/msg/AckermannDrive"))
typestore.register(get_types_from_msg(
    "std_msgs/Header header\nackermann_msgs/AckermannDrive drive",
    "ackermann_msgs/msg/AckermannDriveStamped"))

#========================================================
# Functions
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    if x_max == x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return np.mean(loss)

#========================================================
# Global Config
#========================================================
lidar = []
servo = []
speed = []
test_lidar = []
test_servo = []
test_speed = []

model_name = 'TLN2test'
dataset_path = [
    './Dataset/out.bag',
    './Dataset/f2.bag',
    './Dataset/f4.bag',
]
loss_figure_path = './Figures/loss_curve.png'

scales = [1.0]
down_sample_param = 1  # load at full resolution
lr = 5e-5
loss_function = 'huber'
batch_size = 64
num_epochs = 20
hz = 40

max_speed = 0
min_speed = 0

#========================================================
# Get Dataset
#========================================================
for pth in dataset_path:
    if not os.path.exists(pth):
        print(f"out.bag doesn't exist in {pth}")
        exit(0)
    good_bag = Reader(pth)

    lidar_data = []
    servo_data = []
    speed_data = []

    good_bag.open()
    for connection, t, rawdata in good_bag.messages():
        msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
        topic = connection.topic
        if topic == 'Lidar':
            ranges = msg.ranges[::down_sample_param]
            lidar_data.append(ranges)
        if topic == 'Ackermann':
            data = msg.drive.steering_angle
            s_data = msg.drive.speed
            servo_data.append(data)
            if s_data > max_speed:
                max_speed = s_data
            speed_data.append(s_data)
    good_bag.close()

    lidar_data = np.array(lidar_data)
    servo_data = np.array(servo_data)
    speed_data = np.array(speed_data)

    shuffled_data = shuffle(
        np.concatenate((servo_data[:, np.newaxis], speed_data[:, np.newaxis]), axis=1),
        random_state=62)
    shuffled_lidar_data = shuffle(lidar_data, random_state=62)

    train_ratio = 0.85
    train_samples = int(train_ratio * len(shuffled_lidar_data))
    x_train_bag = shuffled_lidar_data[:train_samples]
    x_test_bag = shuffled_lidar_data[train_samples:]

    y_train_bag = shuffled_data[:train_samples]
    y_test_bag = shuffled_data[train_samples:]

    lidar.extend(x_train_bag)
    servo.extend(y_train_bag[:, 0])
    speed.extend(y_train_bag[:, 1])

    test_lidar.extend(x_test_bag)
    test_servo.extend(y_test_bag[:, 0])
    test_speed.extend(y_test_bag[:, 1])

    print(f'\nData in {pth}:')
    print(f'Shape of Train Data --- Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
    print(f'Shape of Test Data --- Lidar: {len(test_lidar)}, Servo: {len(test_servo)}, Speed: {len(test_speed)}')

total_number_samples = len(lidar)
print(f'Overall Samples = {total_number_samples}')

lidar = np.expand_dims(np.asarray(lidar), -1)
servo = np.asarray(servo)
speed = np.asarray(speed)
speed = linear_map(speed, min_speed, max_speed, 0, 1)
test_lidar = np.expand_dims(np.asarray(test_lidar), -1)
test_servo = np.asarray(test_servo)
test_speed = np.asarray(test_speed)
test_speed = linear_map(test_speed, min_speed, max_speed, 0, 1)

print(f'Min_speed: {min_speed}')
print(f'Max_speed: {max_speed}')
print(f'Loaded {len(lidar)} Training samples ---- {(len(lidar)/total_number_samples)*100:0.2f}% of overall')
print(f'Loaded {len(test_lidar)} Testing samples ---- {(len(test_lidar)/total_number_samples)*100:0.2f}% of overall\n')

assert len(lidar) == len(servo) == len(speed)
assert len(test_lidar) == len(test_servo) == len(test_speed)

#======================================================
# Split Dataset
#======================================================
print('Splitting Data into Train/Test')
train_targets = np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1)
test_data = np.concatenate((test_servo[:, np.newaxis], test_speed[:, np.newaxis]), axis=1)

print(f'Train Data(lidar): {lidar.shape}')
print(f'Train Data(servo, speed): {servo.shape}, {speed.shape}')
print(f'Test Data(lidar): {test_lidar.shape}')
print(f'Test Data(servo, speed): {test_servo.shape}, {test_speed.shape}')

#======================================================
# DNN Arch
#======================================================
num_groups = len(scales)
embed_dim = 16

scan_input = tf.keras.layers.Input(shape=(None, 1), name='scan')
group_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='group')

res_embed = tf.keras.layers.Embedding(num_groups, embed_dim, name='embed')(group_input)

x = tf.keras.layers.Conv1D(24, 10, strides=4, activation='relu')(scan_input)
x = ConditionalBatchNorm(num_groups)([x, group_input])
x = tf.keras.layers.Conv1D(36, 8, strides=4, activation='relu')(x)
x = ConditionalBatchNorm(num_groups)([x, group_input])
x = tf.keras.layers.Conv1D(48, 4, strides=2, activation='relu')(x)
x = ConditionalBatchNorm(num_groups)([x, group_input])
x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
x = ConditionalBatchNorm(num_groups)([x, group_input])
x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
x = ConditionalBatchNorm(num_groups)([x, group_input])

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Concatenate()([x, res_embed])
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dense(50, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(2, activation='tanh')(x)

model = MultiResModel(scales, inputs=[scan_input, group_input], outputs=output)

#======================================================
# Model Compilation and Fit
#======================================================
optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss=loss_function)
print(model.summary())

start_time = time.time()
history = model.fit(
    lidar, train_targets,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(test_lidar, test_data))

print(f'=============>{int(time.time() - start_time)} seconds<=============')

# Plot training and validation losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
os.makedirs('./Figures', exist_ok=True)
plt.savefig(loss_figure_path)
plt.close()

#======================================================
# BN Calibration
#======================================================
print("Calibrating BN statistics...")
calib_ds = tf.data.Dataset.from_tensor_slices((lidar, train_targets)).batch(batch_size)
calibrate_bn(model, calib_ds, scales)

#======================================================
# Save Model
#======================================================
model.save('./Benchmark/f1tenth_benchmarks/zarrar/' + model_name + '.keras')
print(f"{model_name}.keras saved.")

base_resolution = lidar.shape[1]
export_tflite(model, base_resolution, scales, './Benchmark/f1tenth_benchmarks/zarrar/' + model_name + '.tflite')

base_resolution = lidar.shape[1]
def evaluate_tflite(model_path, test_lidar_full, test_data, scales):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scan_idx = input_details[0]['index']

    period = 1.0 / hz

    for group_idx, scale in enumerate(scales):
        factor = int(round(1.0 / scale))
        preds = []
        times = []

        for i in range(len(test_lidar_full)):
            scan = test_lidar_full[i]
            if factor > 1:
                scan = tf.nn.avg_pool1d(
                    scan[np.newaxis], ksize=factor, strides=factor, padding='VALID'
                ).numpy()[0]
            scan_in = scan[np.newaxis].astype(np.float32)

            interpreter.resize_tensor_input(scan_idx, scan_in.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(scan_idx, scan_in)

            ts = time.time()
            interpreter.invoke()
            dur = time.time() - ts
            times.append(dur * 1e6)

            if dur > period:
                print(f"{dur:.3f}: took {int(dur * 1e6)} us - deadline miss.")

            out = interpreter.get_tensor(output_details[0]['index'])
            preds.append(out[0])

        preds = np.array(preds)
        arr = np.array(times)
        arr = arr[arr < np.percentile(arr, 99)]
        res = int(test_lidar_full.shape[1] * scale)
        hl = huber_loss(test_data, preds)

        print(f"\nScale {scale:.2f} (resolution {res}):")
        print(f"  Huber Loss: {hl:.4f}")
        print(f"  Avg Inference Time: {np.mean(arr):.0f} us")
        print(f"  Max Inference Time: {np.max(arr):.0f} us")

    # Plot inference times for full resolution
    times = []
    for i in range(len(test_lidar_full)):
        scan_in = test_lidar_full[i][np.newaxis].astype(np.float32)
        interpreter.resize_tensor_input(scan_idx, scan_in.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(scan_idx, scan_in)
        ts = time.time()
        interpreter.invoke()
        times.append((time.time() - ts) * 1e6)
    arr = np.array(times)
    arr = arr[arr < np.percentile(arr, 99)]
    plt.figure()
    plt.plot(arr)
    plt.xlabel('Inference Iteration')
    plt.ylabel('Inference Time (microseconds)')
    plt.title('Inference Time per Iteration')
    plt.savefig('./Figures/inference_times.png')
    plt.close()

print("\n==========================================")
print("TFLite Model Evaluation")
print("==========================================")
evaluate_tflite('./Benchmark/f1tenth_benchmarks/zarrar/' + model_name + '.tflite', test_lidar, test_data, scales)

print('\nEnd')