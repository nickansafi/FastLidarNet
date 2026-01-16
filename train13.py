#Requirement Library
import os
import sqlite3
from sklearn.utils import shuffle
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.losses import huber
from tensorflow.keras.optimizers import Adam, AdamW
from scipy.ndimage import uniform_filter1d

from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types

import glob
from custom import *

# Register AckermannDriveStamped message type
ackermann_msg_def = """
std_msgs/Header header
AckermannDrive drive

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: ackermann_msgs/AckermannDrive
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""

register_types(get_types_from_msg(ackermann_msg_def, 'ackermann_msgs/msg/AckermannDriveStamped'))

# Check GPU availability
gpu_available = tf.test.is_gpu_available()
print('GPU AVAILABLE:', gpu_available)

#========================================================
# Functions
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    """Linear mapping function."""
    if x_max == x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    mean_loss = np.mean(loss)
    return mean_loss

def downsample_average(array, factor):
    """Downsample by averaging using uniform filter."""
    if factor == 1:
        return array
    smoothed = uniform_filter1d(array, size=factor, mode='nearest')
    return smoothed[::factor]

def read_db3_file(db3_path):
    """Read messages directly from a .db3 file."""
    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {row[0]: {'name': row[1], 'type': row[2]} for row in cursor.fetchall()}
    
    cursor.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp")
    messages = cursor.fetchall()
    
    conn.close()
    return topics, messages


def load_dataset(paths, down_sample_params, train_ratio=0.85):
    """Load and process data from given paths."""
    lidar = {}
    servo = {}
    speed = {}
    test_lidar = {}
    test_servo = {}
    test_speed = {}
    
    max_speed = 0
    min_speed = float('inf')
    
    for pth in paths:
        if not os.path.exists(pth):
            print(f"{pth} doesn't exist")
            continue

        print(f'\nReading {pth}...')
        
        topics, messages = read_db3_file(pth)
        
        lidar_topic_id = None
        ackermann_topic_id = None
        
        for topic_id, topic_info in topics.items():
            topic_name = topic_info['name']
            if topic_name in ['Lidar', 'scan', '/scan', '/Lidar']:
                lidar_topic_id = topic_id
            elif topic_name in ['Ackermann', 'drive', '/drive', '/Ackermann']:
                ackermann_topic_id = topic_id
        
        if lidar_topic_id is None or ackermann_topic_id is None:
            print(f"Warning: Could not find required topics in {pth}")
            print(f"Available topics: {[(tid, info['name']) for tid, info in topics.items()]}")
            continue
        
        lidar_data = []
        servo_data = []
        speed_data = []
        
        for topic_id, timestamp, data in messages:
            try:
                if topic_id == lidar_topic_id:
                    msg = deserialize_cdr(data, topics[topic_id]['type'])
                    lidar_data.append(msg.ranges)
                elif topic_id == ackermann_topic_id:
                    msg = deserialize_cdr(data, topics[topic_id]['type'])
                    servo_data.append(msg.drive.steering_angle)
                    s_data = msg.drive.speed
                    if s_data > max_speed:
                        max_speed = s_data
                    if s_data < min_speed:
                        min_speed = s_data
                    speed_data.append(s_data)
            except Exception as e:
                print(f"Error deserializing message: {e}")
                continue
        
        if len(lidar_data) != len(servo_data) or len(lidar_data) != len(speed_data):
            print(f"Warning: Data length mismatch in {pth} - skipping")
            continue
        
        servo_data = np.array(servo_data)
        speed_data = np.array(speed_data)
        
        indices = np.arange(len(lidar_data))
        np.random.seed(62)
        np.random.shuffle(indices)
        
        lidar_data = [lidar_data[i] for i in indices]
        servo_data = servo_data[indices]
        speed_data = speed_data[indices]
        
        train_samples = int(train_ratio * len(lidar_data))
        x_train_bag = lidar_data[:train_samples]
        x_test_bag = lidar_data[train_samples:] if train_ratio < 1.0 else []
        y_train_servo = servo_data[:train_samples]
        y_train_speed = speed_data[:train_samples]
        y_test_servo = servo_data[train_samples:] if train_ratio < 1.0 else np.array([])
        y_test_speed = speed_data[train_samples:] if train_ratio < 1.0 else np.array([])
        
        for param in down_sample_params:
            for i in range(len(x_train_bag)):
                scan_downsampled = downsample_average(np.array(x_train_bag[i]), param)
                resolution = len(scan_downsampled)
                
                if resolution not in lidar:
                    lidar[resolution] = []
                    servo[resolution] = []
                    speed[resolution] = []
                    test_lidar[resolution] = []
                    test_servo[resolution] = []
                    test_speed[resolution] = []
                
                lidar[resolution].append(scan_downsampled)
                servo[resolution].append(y_train_servo[i])
                speed[resolution].append(y_train_speed[i])
            
            if train_ratio < 1.0:
                for i in range(len(x_test_bag)):
                    scan_downsampled = downsample_average(np.array(x_test_bag[i]), param)
                    resolution = len(scan_downsampled)
                    
                    if resolution not in test_lidar:
                        test_lidar[resolution] = []
                        test_servo[resolution] = []
                        test_speed[resolution] = []
                    
                    test_lidar[resolution].append(scan_downsampled)
                    test_servo[resolution].append(y_test_servo[i])
                    test_speed[resolution].append(y_test_speed[i])
        
        print(f'Train: {len(x_train_bag)}, Test: {len(x_test_bag)}')
    
    # Handle case where no valid speed data was found
    if min_speed == float('inf'):
        min_speed = 0
    
    return lidar, servo, speed, test_lidar, test_servo, test_speed, min_speed, max_speed


def create_tf_dataset(lidar, servo, speed, min_speed, max_speed, batch_size):
    """Convert loaded data to tf.data.Dataset with proper shuffling."""
    resolutions = list(lidar.keys())
    
    if not resolutions:
        return None, []
    
    lidar_processed = {}
    servo_processed = {}
    speed_processed = {}
    
    total_samples = 0
    for resolution in resolutions:
        lidar_processed[resolution] = np.asarray(lidar[resolution])
        lidar_processed[resolution] = np.expand_dims(lidar_processed[resolution], axis=-1)
        servo_processed[resolution] = np.asarray(servo[resolution])
        speed_processed[resolution] = np.asarray(speed[resolution])
        speed_processed[resolution] = linear_map(speed_processed[resolution], min_speed, max_speed, 0, 1)
        total_samples += len(lidar_processed[resolution])
    
    datasets = []
    for resolution in resolutions:
        resolution_keys = np.full((len(lidar_processed[resolution]),), resolution, dtype=np.int32)
        data = tf.data.Dataset.from_tensor_slices((
            (lidar_processed[resolution], resolution_keys),
            np.stack([servo_processed[resolution], speed_processed[resolution]], axis=1)
        ))
        data = data.batch(batch_size)
        datasets.append(data)
    
    combined = datasets[0]
    for data in datasets[1:]:
        combined = combined.concatenate(data)
    
    # THIS IS THE ONLY NEW LINE
    combined = combined.shuffle(buffer_size=100, reshuffle_each_iteration=True)
    
    return combined, resolutions


#========================================================
# Global Config
#========================================================
down_sample_params = [1, 2, 4]
model_name = 'TLN'
loss_figure_path = './Figures/loss_curve.png'
batch_size = 256
train_ratio = 0.85

#========================================================
# Load Both Datasets
#========================================================
group1_paths = glob.glob('./Dataset3/Dataset/group1/**/*.db3', recursive=True)
group2_paths = glob.glob('./Dataset3/Dataset/group2/**/*.db3', recursive=True)

print("=== Loading Group 1 (large dataset) ===")
lidar1, servo1, speed1, test_lidar1, test_servo1, test_speed1, min_speed1, max_speed1 = \
    load_dataset(group1_paths, down_sample_params, train_ratio)

print("\n=== Loading Group 2 (curated dataset) ===")
lidar2, servo2, speed2, test_lidar2, test_servo2, test_speed2, min_speed2, max_speed2 = \
    load_dataset(group2_paths, down_sample_params, train_ratio)

# Use global min/max for consistent normalization
global_min_speed = min(min_speed1, min_speed2)
global_max_speed = max(max_speed1, max_speed2)

print(f"\nGlobal speed range: [{global_min_speed}, {global_max_speed}]")

# Create TF datasets
train_dataset_1, resolutions = create_tf_dataset(
    lidar1, servo1, speed1, global_min_speed, global_max_speed, batch_size
)
val_dataset_1, _ = create_tf_dataset(
    test_lidar1, test_servo1, test_speed1, global_min_speed, global_max_speed, batch_size
)

train_dataset_2, _ = create_tf_dataset(
    lidar2, servo2, speed2, global_min_speed, global_max_speed, batch_size
)
val_dataset_2, _ = create_tf_dataset(
    test_lidar2, test_servo2, test_speed2, global_min_speed, global_max_speed, batch_size
)

# Print dataset info
total_samples_1 = sum(len(lidar1[res]) for res in lidar1.keys()) if lidar1 else 0
total_samples_2 = sum(len(lidar2[res]) for res in lidar2.keys()) if lidar2 else 0
print(f"\nGroup 1 - Total training samples: {total_samples_1}")
print(f"Group 2 - Total training samples: {total_samples_2}")
print(f"Resolutions: {resolutions}")

if not resolutions:
    print("No data was loaded!")
    exit(1)

#======================================================
# DNN Arch
#======================================================
scan_input = tf.keras.layers.Input(shape=(None, 1), name='scan_input')
resolution_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='resolution_input')

x = tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu')(scan_input)
x = ConditionalBatchNormGather(resolutions)([x, resolution_input])

x = tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu')(x)
x = ConditionalBatchNormGather(resolutions)([x, resolution_input])

x = tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu')(x)
x = ConditionalBatchNormGather(resolutions)([x, resolution_input])

x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = ConditionalBatchNormGather(resolutions)([x, resolution_input])

x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = ConditionalBatchNormGather(resolutions)([x, resolution_input])

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
output = tf.keras.layers.Dense(2, activation='tanh', name="output")(x)

model = tf.keras.Model(inputs=[scan_input, resolution_input], outputs=output)
print(model.summary())

#======================================================
# Model Fit
#======================================================
start_time = time.time()

#------------------------------------------------------
# Stage 1: Brief pre-training (initialization only)
#------------------------------------------------------
print("\n=== Stage 1: Pre-training on Group 1 (short) ===")

optimizer_1 = AdamW(learning_rate=3e-4, weight_decay=1e-4)
model.compile(optimizer=optimizer_1, loss='huber', jit_compile=False)

# No early stopping - just run fixed short duration
history_1 = model.fit(
    train_dataset_1,
    epochs=5,  # Just a few epochs for initialization
    validation_data=val_dataset_1
)

model.save('./Models/' + model_name + '_stage1.keras')

#------------------------------------------------------
# Stage 2: Main training on curated dataset
#------------------------------------------------------
print("\n=== Stage 2: Main training on Group 2 ===")

optimizer_2 = AdamW(learning_rate=3e-4, weight_decay=1e-4)  # Same LR, not lower
model.compile(optimizer=optimizer_2, loss='huber', jit_compile=False)

reduce_lr_2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_delta=1e-4,
    min_lr=1e-6,
    verbose=1
)

early_stop_2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,
    patience=15,
    restore_best_weights=True
)

history_2 = model.fit(
    train_dataset_2,
    epochs=100,  # Full training here
    validation_data=val_dataset_2,
    callbacks=[reduce_lr_2, early_stop_2]
)

print(f'\n=============>{int(time.time() - start_time)} seconds<=============')

#======================================================
# Plot Loss Curves
#======================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Stage 1 plot
ax1.plot(history_1.history['loss'], label='Train')
if 'val_loss' in history_1.history:
    ax1.plot(history_1.history['val_loss'], label='Val')
ax1.set_title('Stage 1: Pre-training Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()

# Stage 2 plot
ax2.plot(history_2.history['loss'], label='Train')
if 'val_loss' in history_2.history:
    ax2.plot(history_2.history['val_loss'], label='Val')
ax2.set_title('Stage 2: Fine-tuning Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig(loss_figure_path)
plt.close()

#======================================================
# Save Final Model
#======================================================
model.save('./Models/' + model_name + '_final.keras')
print(f"{model_name}_final.keras saved.")