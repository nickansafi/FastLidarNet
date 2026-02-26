#Requirement Library
import os
import glob
import sqlite3
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from tensorflow.keras.optimizers import Adam

from multi_res import (
    ConditionalBatchNorm, MultiResModel, calibrate_bn, export_tflite,
    InferGroupFromShape
)

print('GPU AVAILABLE:', len(tf.config.list_physical_devices('GPU')) > 0)

typestore = get_typestore(Stores.ROS2_HUMBLE)
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

def read_db3(pth, typestore):
    """Read lidar, servo, speed from a single .db3 file."""
    lidar_data = []
    servo_data = []
    speed_data = []
    max_speed = 0
    min_speed = float('inf')

    conn = sqlite3.connect(pth)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT topics.name, topics.type, messages.data "
        "FROM messages JOIN topics ON messages.topic_id = topics.id")
    for topic, dtype, rawdata in cursor:
        try:
            msg = typestore.deserialize_cdr(rawdata, dtype)
        except:
            continue
        if topic in ['Lidar', 'scan', '/scan', '/Lidar']:
            ranges = msg.ranges[:1080]
            if len(ranges) != 1080:
                continue
            lidar_data.append(ranges)
        elif topic in ['Ackermann', 'drive', '/drive', '/Ackermann']:
            servo_data.append(msg.drive.steering_angle)
            s = msg.drive.speed
            if s > max_speed:
                max_speed = s
            if s < min_speed:
                min_speed = s
            speed_data.append(s)
    conn.close()

    if len(set([len(lidar_data), len(servo_data), len(speed_data)])) != 1:
        print(f"  Skipping {pth}: length mismatch "
              f"(lidar={len(lidar_data)}, servo={len(servo_data)}, speed={len(speed_data)})")
        return None, None, None, 0, float('inf')

    return (np.array(lidar_data), np.array(servo_data), np.array(speed_data),
            max_speed, min_speed)

def load_paths(paths, typestore, train_ratio=0.85):
    """Load and split data from a list of .db3 paths."""
    all_lidar, all_servo, all_speed = [], [], []
    test_lidar, test_servo, test_speed = [], [], []
    g_max, g_min = 0, float('inf')

    for pth in paths:
        if not os.path.exists(pth):
            print(f"  {pth} not found, skipping")
            continue
        print(f'  Reading {pth}...')
        lidar_data, servo_data, speed_data, mx, mn = read_db3(pth, typestore)
        if lidar_data is None:
            continue
        g_max = max(g_max, mx)
        g_min = min(g_min, mn)

        idx = np.arange(len(lidar_data))
        np.random.seed(62)
        np.random.shuffle(idx)
        lidar_data = lidar_data[idx]
        servo_data = servo_data[idx]
        speed_data = speed_data[idx]

        split = int(train_ratio * len(lidar_data))
        all_lidar.extend(lidar_data[:split])
        all_servo.extend(servo_data[:split])
        all_speed.extend(speed_data[:split])
        test_lidar.extend(lidar_data[split:])
        test_servo.extend(servo_data[split:])
        test_speed.extend(speed_data[split:])

        print(f'    Train: {split}, Test: {len(lidar_data) - split}')

    if g_min == float('inf'):
        g_min = 0

    return (np.array(all_lidar), np.array(all_servo), np.array(all_speed),
            np.array(test_lidar), np.array(test_servo), np.array(test_speed),
            g_min, g_max)

#========================================================
# Config
#========================================================
model_name = 'TLNtest3'
scales = [1.0]
lr = 5e-5
batch_size = 256
pretrain_epochs = 5
finetune_epochs = 50
hz = 40

# Current (fine-tune) paths
current_paths = [
    './Dataset/out/out.db3',
    './Dataset/f2/f2.db3',
    './Dataset/f4/f4.db3',
]

# All .db3 files in ./Dataset
all_db3 = sorted(glob.glob('./Dataset/**/*.db3', recursive=True))

# Pretrain paths = everything except current
current_abs = set(os.path.abspath(p) for p in current_paths)
pretrain_paths = [p for p in all_db3 if os.path.abspath(p) not in current_abs]

print(f"Pretrain paths ({len(pretrain_paths)}):")
for p in pretrain_paths:
    print(f"  {p}")
print(f"Finetune paths ({len(current_paths)}):")
for p in current_paths:
    print(f"  {p}")

#========================================================
# Load Both Datasets
#========================================================
print("\n=== Loading pretrain data ===")
(pre_lidar, pre_servo, pre_speed,
 pre_test_lidar, pre_test_servo, pre_test_speed,
 pre_min, pre_max) = load_paths(pretrain_paths, typestore)

print("\n=== Loading finetune data ===")
(ft_lidar, ft_servo, ft_speed,
 ft_test_lidar, ft_test_servo, ft_test_speed,
 ft_min, ft_max) = load_paths(current_paths, typestore)

# Global speed range
global_min = min(pre_min, ft_min) if len(pre_lidar) > 0 else ft_min
global_max = max(pre_max, ft_max) if len(pre_lidar) > 0 else ft_max
print(f"\nGlobal speed range: [{global_min}, {global_max}]")

def prepare(lidar, servo, speed, test_lidar, test_servo, test_speed):
    lidar = np.expand_dims(lidar, -1)
    servo = np.array(servo)
    speed = linear_map(np.array(speed), global_min, global_max, 0, 1)
    test_lidar = np.expand_dims(test_lidar, -1)
    test_servo = np.array(test_servo)
    test_speed = linear_map(np.array(test_speed), global_min, global_max, 0, 1)
    targets = np.stack([servo, speed], axis=1)
    test_targets = np.stack([test_servo, test_speed], axis=1)
    return lidar, targets, test_lidar, test_targets

pre_x, pre_y, pre_tx, pre_ty = prepare(
    pre_lidar, pre_servo, pre_speed,
    pre_test_lidar, pre_test_servo, pre_test_speed)

ft_x, ft_y, ft_tx, ft_ty = prepare(
    ft_lidar, ft_servo, ft_speed,
    ft_test_lidar, ft_test_servo, ft_test_speed)

print(f"Pretrain samples: {len(pre_x)} train, {len(pre_tx)} test")
print(f"Finetune samples: {len(ft_x)} train, {len(ft_tx)} test")

#========================================================
# Build Model
#========================================================
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
print(model.summary())

#========================================================
# Stage 1: Pretrain on everything except current paths
#========================================================
start_time = time.time()

if len(pre_x) > 0:
    print("\n=== Stage 1: Pretraining ===")
    model.compile(optimizer=Adam(lr), loss='huber')

    pre_train_ds = tf.data.Dataset.from_tensor_slices((pre_x, pre_y)) \
        .shuffle(len(pre_x)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    pre_val_ds = tf.data.Dataset.from_tensor_slices((pre_tx, pre_ty)) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    history_pre = model.fit(
        pre_train_ds,
        epochs=pretrain_epochs,
        validation_data=pre_val_ds if len(pre_tx) > 0 else None)
    print(f"Pretrain done in {int(time.time() - start_time)}s")
else:
    print("\nNo pretrain data found, skipping Stage 1")

#========================================================
# Stage 2: Finetune on current paths with LR schedule
#========================================================
print("\n=== Stage 2: Finetuning ===")
model.compile(optimizer=Adam(lr), loss='huber')

ft_train_ds = tf.data.Dataset.from_tensor_slices((ft_x, ft_y)) \
    .shuffle(len(ft_x)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ft_val_ds = tf.data.Dataset.from_tensor_slices((ft_tx, ft_ty)) \
    .batch(batch_size).prefetch(tf.data.AUTOTUNE)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5,
    min_delta=1e-4, min_lr=1e-6, verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-4, patience=15,
    restore_best_weights=True)

history_ft = model.fit(
    ft_train_ds,
    epochs=finetune_epochs,
    validation_data=ft_val_ds,
    callbacks=[reduce_lr, early_stop])

print(f'\nTotal training time: {int(time.time() - start_time)}s')

#========================================================
# Loss Plot
#========================================================
os.makedirs('./Figures', exist_ok=True)
plt.figure()
plt.plot(history_ft.history['loss'], label='Train')
plt.plot(history_ft.history['val_loss'], label='Val')
plt.title('Finetune Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('./Figures/loss_curve.png')
plt.close()

#========================================================
# Keras Model Evaluation
#========================================================
print("\n==========================================")
print("Keras Model Evaluation")
print("==========================================")

for i, scale in enumerate(scales):
    factor = int(round(1.0 / scale))
    if factor > 1:
        tx = tf.nn.avg_pool1d(
            tf.constant(ft_tx, dtype=tf.float32),
            ksize=factor, strides=factor, padding='VALID').numpy()
    else:
        tx = ft_tx
    group = np.full(len(tx), i, dtype=np.int32)
    preds = model.predict([tx, group])
    hl = huber_loss(ft_ty, preds)
    res = int(1080 * scale)
    print(f"Scale {scale:.2f} (res {res}): Huber Loss = {hl:.4f}")

#========================================================
# BN Calibration
#========================================================
print("\nCalibrating BN statistics...")
calib_ds = tf.data.Dataset.from_tensor_slices((ft_x, ft_y)).batch(batch_size)
calibrate_bn(model, calib_ds, scales)

#========================================================
# Save
#========================================================
os.makedirs('./Models', exist_ok=True)
model.save('./Models/' + model_name + '.keras')
print(f"{model_name}.keras saved.")

base_resolution = ft_x.shape[1]
export_tflite(model, base_resolution, scales, './Models/' + model_name + '.tflite')

#========================================================
# TFLite Evaluation
#========================================================
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

print("\n==========================================")
print("TFLite Model Evaluation")
print("==========================================")
evaluate_tflite('./Models/' + model_name + '.tflite', ft_tx, ft_ty, scales)

print('\nDone.')