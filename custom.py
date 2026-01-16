import keras
from keras import ops
import tensorflow as tf

# === Muon ResourceVariable patches (MUST be at top) ===
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

# Patch 'ndim'
@property
def _ndim(self):
    return len(self.shape)
ResourceVariable.ndim = _ndim

# Patch 'path'
_var_id_to_path = {}

def build_var_paths(model):
    """Build variable paths by traversing model layers."""
    _var_id_to_path.clear()
    for layer in model._flatten_layers():
        layer_path = layer.name
        for var in layer._trainable_variables + layer._non_trainable_variables:
            var_name = var.name.rsplit('/', 1)[-1]
            if ':' in var_name:
                var_name = var_name.rsplit(':', 1)[0]
            full_path = f"{layer_path}/{var_name}"
            _var_id_to_path[id(var)] = full_path

@property
def _path(self):
    if id(self) in _var_id_to_path:
        return _var_id_to_path[id(self)]
    name = self.name
    if ':' in name:
        return name.rsplit(':', 1)[0]
    return name
ResourceVariable.path = _path

@keras.saving.register_keras_serializable(package="Custom")
class ConditionalBatchNorm(keras.layers.Layer):
    """Applies different batch normalizations based on scan resolution groups.
    Groups resolutions by rounding down to nearest 270 multiple."""
    
    def __init__(self, resolutions, group_size=270, **kwargs):
        super(ConditionalBatchNorm, self).__init__(**kwargs)
        self.group_size = group_size
        
        # Group resolutions by rounding down to nearest multiple
        self.resolution_to_group = {}
        unique_groups = set()
        
        for res in resolutions:
            group = (res // group_size) * group_size
            self.resolution_to_group[res] = group
            unique_groups.add(group)
        
        self.groups = sorted(list(unique_groups))
        self.bn_layers = {
            group: keras.layers.BatchNormalization(name=f'bn_group_{group}')
            for group in self.groups
        }
    
    def build(self, input_shape):
        scan_shape = input_shape[0]
        for bn in self.bn_layers.values():
            bn.build(scan_shape)
        super(ConditionalBatchNorm, self).build(input_shape)
    
    def call(self, inputs, training=None):
        import tensorflow as tf
        
        scan, resolution = inputs
        
        # Handle batched resolution input - take first element
        resolution_flat = tf.reshape(resolution, [-1])
        resolution_scalar = resolution_flat[0]
        
        # Map resolution to its group
        resolution_scalar = tf.cast(resolution_scalar, tf.int32)
        group_scalar = (resolution_scalar // self.group_size) * self.group_size
        
        # Find the index of the matching group
        groups_tensor = tf.constant(self.groups, dtype=tf.int32)
        matches = tf.equal(groups_tensor, group_scalar)
        index = tf.argmax(tf.cast(matches, tf.int32), output_type=tf.int32)
        
        # Create branch functions
        def make_branch_fn(idx):
            group = self.groups[idx]
            return lambda: self.bn_layers[group](scan, training=training)
        
        branch_fns = {i: make_branch_fn(i) for i in range(len(self.groups))}
        
        # Use switch_case with tensor index
        output = tf.switch_case(index, branch_fns)
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def compute_output_spec(self, *args, **kwargs):
        import tensorflow as tf
        scan_input = args[0][0] if isinstance(args[0], (list, tuple)) else args[0]
        return keras.KerasTensor(shape=scan_input.shape, dtype=scan_input.dtype)
    
    def get_config(self):
        config = super(ConditionalBatchNorm, self).get_config()
        config.update({
            'resolutions': list(self.resolution_to_group.keys()),
            'group_size': self.group_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

@keras.saving.register_keras_serializable()
class ZeroMultiply(keras.layers.Layer):
    """Multiplies input by zero and outputs shape (batch, 2) for dummy connection."""
    def __init__(self, output_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
    
    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        return ops.zeros((batch_size, self.output_dim), dtype="float32")
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super().get_config()
        config.update({'output_dim': self.output_dim})
        return config
    
@keras.saving.register_keras_serializable(package="Custom")
class MuonConv1D(keras.layers.Layer):
    """Conv1D with 2D weight storage for Muon optimizer compatibility.
    
    Stores kernel as (kernel_size * in_channels, out_channels) so Muon
    sees a 2D matrix. Reshapes to (kernel_size, in_channels, out_channels)
    for the actual convolution.
    """
    
    def __init__(self, filters, kernel_size, strides=1, padding='valid', 
                 activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        in_channels = input_shape[-1]
        
        # Store as 2D: (kernel_size * in_channels, filters)
        self.flat_kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size * in_channels, self.filters),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer='zeros',
                trainable=True
            )
        
        self._in_channels = in_channels
        super().build(input_shape)
    
    def call(self, inputs):
        # Reshape to Conv1D format: (kernel_size, in_channels, filters)
        kernel = tf.reshape(
            self.flat_kernel, 
            (self.kernel_size, self._in_channels, self.filters)
        )
        
        output = tf.nn.conv1d(
            inputs,
            filters=kernel,
            stride=self.strides,
            padding=self.padding.upper()
        )
        
        if self.use_bias:
            output = output + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config
    
@keras.saving.register_keras_serializable(package="Custom")
class ConditionalBatchNormGather(keras.layers.Layer):
    """XLA-compatible conditional BN using stacked parameters + gather."""
    
    def __init__(self, resolutions, group_size=270, momentum=0.99, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.momentum = momentum
        self.epsilon = epsilon
        
        unique_groups = set()
        for res in resolutions:
            group = (res // group_size) * group_size
            unique_groups.add(group)
        
        self.groups = sorted(list(unique_groups))
        self.num_groups = len(self.groups)
        self.resolutions = resolutions
    
    def build(self, input_shape):
        num_features = input_shape[0][-1]
        
        # Use keyword arguments only
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.num_groups, num_features),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(self.num_groups, num_features),
            initializer='zeros',
            trainable=True
        )
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=(self.num_groups, num_features),
            initializer='zeros',
            trainable=False
        )
        self.moving_var = self.add_weight(
            name='moving_var',
            shape=(self.num_groups, num_features),
            initializer='ones',
            trainable=False
        )
        super().build(input_shape)
    
    def _get_group_index(self, resolution):
        """Convert resolution to index into parameter tensors."""
        res_scalar = tf.cast(tf.reshape(resolution, [-1])[0], tf.int32)
        group_val = (res_scalar // self.group_size) * self.group_size
        groups_t = tf.constant(self.groups, dtype=tf.int32)
        idx = tf.argmax(tf.cast(tf.equal(groups_t, group_val), tf.int32), output_type=tf.int32)
        return idx
    
    def call(self, inputs, training=None):
        scan, resolution = inputs
        
        idx = self._get_group_index(resolution)
        
        gamma_i = tf.gather(self.gamma, idx)
        beta_i = tf.gather(self.beta, idx)
        
        if training:
            mean = tf.reduce_mean(scan, axis=[0, 1])
            var = tf.reduce_mean(tf.square(scan - mean), axis=[0, 1])
            
            one_hot = tf.one_hot(idx, self.num_groups, dtype=tf.float32)
            mask = tf.reshape(one_hot, [self.num_groups, 1])
            
            mean_expanded = tf.expand_dims(mean, 0)
            var_expanded = tf.expand_dims(var, 0)
            
            update_factor = mask * (1.0 - self.momentum)
            
            new_moving_mean = self.moving_mean * (1.0 - update_factor) + mean_expanded * update_factor
            new_moving_var = self.moving_var * (1.0 - update_factor) + var_expanded * update_factor
            
            self.moving_mean.assign(new_moving_mean)
            self.moving_var.assign(new_moving_var)
        else:
            mean = tf.gather(self.moving_mean, idx)
            var = tf.gather(self.moving_var, idx)
        
        normalized = (scan - mean) * tf.math.rsqrt(var + self.epsilon)
        return normalized * gamma_i + beta_i
    
    def get_config(self):
        return {
            **super().get_config(),
            'resolutions': self.resolutions,
            'group_size': self.group_size,
            'momentum': self.momentum,
            'epsilon': self.epsilon
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)