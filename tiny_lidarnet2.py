import time
import numpy as np
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf


class TinyLidarNet(BasePlanner):
    def __init__(self, test_id, scale, pre, model_path):
        super().__init__("TinyLidarNet", test_id)
        self.scale = scale
        self.factor = int(round(1.0 / scale))
        self.name = 'TinyLidarNet'

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Compute default input length and allocate with valid shape
        full_len = 1080
        self.input_len = full_len // self.factor
        self.interpreter.resize_tensor_input(
            self.input_details[0]['index'], [1, self.input_len, 1])
        self.interpreter.allocate_tensors()

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def render_waypoints(self, *args, **kwargs):
        pass

    def plan(self, obs):
        scans = np.array(obs['scan'], dtype=np.float32)

        # Downsample via average pooling
        if self.factor > 1:
            scans = tf.nn.avg_pool1d(
                scans[np.newaxis, :, np.newaxis],
                ksize=self.factor, strides=self.factor, padding='VALID'
            ).numpy()[0, :, 0]

        scans[scans > 10] = 10
        scan_in = scans[np.newaxis, :, np.newaxis].astype(np.float32)

        # Resize only if scan length changed
        if scan_in.shape[1] != self.input_len:
            self.input_len = scan_in.shape[1]
            self.interpreter.resize_tensor_input(
                self.input_details[0]['index'], list(scan_in.shape))
            self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_details[0]['index'], scan_in)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        steer = output[0, 0]
        speed = self.linear_map(output[0, 1], 0, 1, 1, 8)
        return np.array([steer, speed])