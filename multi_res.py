import tensorflow as tf
import keras
import numpy as np


@keras.saving.register_keras_serializable(package="Custom")
class ConditionalBatchNorm(keras.layers.Layer):
    """Per-group BN with separate affine params and running stats."""

    def __init__(self, num_groups, momentum=0.99, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.momentum = momentum
        self.epsilon = epsilon
        self._calibrating = False

    def build(self, input_shape):
        C = input_shape[0][-1]
        self.gamma = self.add_weight('gamma', (self.num_groups, C),
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight('beta', (self.num_groups, C),
                                    initializer='zeros', trainable=True)
        self.moving_mean = self.add_weight('moving_mean', (self.num_groups, C),
                                           initializer='zeros', trainable=False)
        self.moving_var = self.add_weight('moving_var', (self.num_groups, C),
                                          initializer='ones', trainable=False)
        self.calib_count = self.add_weight('calib_count', (self.num_groups,),
                                           initializer='zeros', trainable=False)
        super().build(input_shape)

    def call(self, inputs, training=None):
        scan, group_index = inputs
        idx = tf.cast(tf.reshape(group_index, [-1])[0], tf.int32)
        gamma_i = tf.gather(self.gamma, idx)
        beta_i = tf.gather(self.beta, idx)

        if training:
            mean = tf.reduce_mean(scan, axis=[0, 1])
            var = tf.reduce_mean(tf.square(scan - mean), axis=[0, 1])
            one_hot = tf.one_hot(idx, self.num_groups, dtype=tf.float32)
            mask = tf.reshape(one_hot, [self.num_groups, 1])

            if self._calibrating:
                count = tf.gather(self.calib_count, idx) + 1.0
                self.calib_count.assign(self.calib_count + one_hot)
                old_mean = tf.gather(self.moving_mean, idx)
                old_var = tf.gather(self.moving_var, idx)
                new_mean = old_mean + (mean - old_mean) / count
                new_var = old_var + (var - old_var) / count
                self.moving_mean.assign(
                    self.moving_mean * (1.0 - mask) +
                    tf.expand_dims(new_mean, 0) * mask)
                self.moving_var.assign(
                    self.moving_var * (1.0 - mask) +
                    tf.expand_dims(new_var, 0) * mask)
            else:
                update = mask * (1.0 - self.momentum)
                self.moving_mean.assign(
                    self.moving_mean * (1.0 - update) +
                    tf.expand_dims(mean, 0) * update)
                self.moving_var.assign(
                    self.moving_var * (1.0 - update) +
                    tf.expand_dims(var, 0) * update)
        else:
            mean = tf.gather(self.moving_mean, idx)
            var = tf.gather(self.moving_var, idx)

        return (scan - mean) * tf.math.rsqrt(var + self.epsilon) * gamma_i + beta_i

    def get_config(self):
        return {**super().get_config(),
                'num_groups': self.num_groups,
                'momentum': self.momentum,
                'epsilon': self.epsilon}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiResModel(tf.keras.Model):
    def __init__(self, scales, **kwargs):
        super().__init__(**kwargs)
        self.scales = scales
        self._factors = [int(round(1.0 / s)) for s in scales]

    def _downsample(self, scans, factor):
        if factor <= 1:
            return scans
        return tf.nn.avg_pool1d(scans, ksize=factor, strides=factor, padding='VALID')

    def train_step(self, data):
        scans, targets = data
        n = float(len(self.scales))
        with tf.GradientTape() as tape:
            total_loss = 0.0
            for i, factor in enumerate(self._factors):
                ds = self._downsample(scans, factor)
                group = tf.fill([tf.shape(ds)[0]], i)
                preds = self([ds, group], training=True)
                total_loss += tf.reduce_mean(tf.keras.losses.huber(targets, preds))
            avg_loss = total_loss / n
        grads = tape.gradient(avg_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": avg_loss}

    def test_step(self, data):
        scans, targets = data
        n = float(len(self.scales))
        total_loss = 0.0
        for i, factor in enumerate(self._factors):
            ds = self._downsample(scans, factor)
            group = tf.fill([tf.shape(ds)[0]], i)
            preds = self([ds, group], training=False)
            total_loss += tf.reduce_mean(tf.keras.losses.huber(targets, preds))
        return {"loss": total_loss / n}


def calibrate_bn(model, dataset, scales, num_batches=None):
    """Post-training BN recalibration with cumulative averaging."""
    factors = [int(round(1.0 / s)) for s in scales]
    bn_layers = [l for l in model.layers if isinstance(l, ConditionalBatchNorm)]

    for layer in bn_layers:
        layer._calibrating = True
        layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))
        layer.moving_var.assign(tf.zeros_like(layer.moving_var))
        layer.calib_count.assign(tf.zeros_like(layer.calib_count))

    was_eager = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)

    count = 0
    for scans, _ in dataset:
        for i, factor in enumerate(factors):
            ds = scans if factor <= 1 else tf.nn.avg_pool1d(
                scans, ksize=factor, strides=factor, padding='VALID')
            group = tf.fill([tf.shape(ds)[0]], i)
            model([ds, group], training=True)
        count += 1
        if num_batches and count >= num_batches:
            break

    tf.config.run_functions_eagerly(was_eager)
    for layer in bn_layers:
        layer._calibrating = False


@keras.saving.register_keras_serializable(package="Custom")
class InferGroupFromShape(keras.layers.Layer):
    """Derives group index from scan length. No extra input needed."""

    def __init__(self, expected_lengths, **kwargs):
        super().__init__(**kwargs)
        self.expected_lengths = expected_lengths

    def build(self, input_shape):
        self.lengths_tensor = tf.constant(self.expected_lengths, dtype=tf.int32)
        super().build(input_shape)

    def call(self, scan):
        length = tf.shape(scan)[1]
        return tf.argmin(tf.abs(self.lengths_tensor - length), output_type=tf.int32)

    def get_config(self):
        return {**super().get_config(), 'expected_lengths': self.expected_lengths}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def export_tflite(model, base_resolution, scales, output_path):
    expected_lengths = [int(base_resolution * s) for s in scales]

    infer_group = InferGroupFromShape(expected_lengths)

    @tf.function(input_signature=[
        tf.TensorSpec([1, None, 1], tf.float32, name='scan'),
    ])
    def serve(scan):
        group = infer_group(scan)
        group_batched = tf.expand_dims(group, 0)
        return model([scan, group_batched], training=False)

    concrete = serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    tflite_bytes = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_bytes)
    print(f"Saved {output_path}")