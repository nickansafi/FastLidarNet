import time
import numpy as np
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import numpy as np
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf
from custom import ConditionalBatchNorm, ZeroMultiply
import keras

from scipy.ndimage import uniform_filter1d

class TinyLidarNet(BasePlanner):
    def __init__(self, test_id, skip_n, pre, model_path):
        super().__init__("TinyLidarNet", test_id)
        self.pre = pre
        self.skip_n = skip_n
        self.model_path = model_path
        self.name = 'TinyLidarNet'
        self.model = tf.keras.models.load_model(model_path, custom_objects={
    "ConditionalBatchNorm": ConditionalBatchNorm,
    "ZeroMultiply": ZeroMultiply
})
        self.infer = tf.function(lambda x, res: self.model((x,res),training=False))
        self.scan_buffer = np.zeros((2, 20))
        self.bn_index = 0

        self.temp_scan = []

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def render_waypoints(self, *args, **kwargs):
        pass
        
    def transform_obs(self, scan):
        self.scan_buffer
        scan = scan[:1080]
        scan = scan[::54]
        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.scan_buffer.shape[0]):
                self.scan_buffer[i, :] = scan
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        scans = np.reshape(self.scan_buffer, (-1))
        return scans

    
    def plan(self, obs):
        def downsample_average(array, factor):
            """Downsample by averaging using uniform filter."""
            if factor == 1:
                return array
            smoothed = uniform_filter1d(array, size=factor, mode='nearest')
            return smoothed[::factor]
        scans = obs['scan']

        scans = downsample_average(scans, self.skip_n)

        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        
        scans = np.array(scans)
        scans[scans>10] = 10

        scans = np.expand_dims(scans, axis=-1).astype(np.float32)
        scans = np.expand_dims(scans, axis=0)
        
        start_time = time.time()
        inf_time = time.time() - start_time
        inf_time = inf_time*1000
        output = self.infer(scans, np.array([scans.shape[1]], dtype=np.int32))

        steer = output[0,0]
        speed = output[0,1]
        min_speed = 1
        max_speed = 8
        speed = self.linear_map(speed, 0, 1, min_speed, max_speed) 
        action = np.array([steer, speed])

        return action