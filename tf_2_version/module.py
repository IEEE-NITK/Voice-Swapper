import tensorflow as tf 

def gated_linear_layer(inputs, gates, name = None):

    activation = tf.math.multiply(x = inputs, y = tf.sigmoid(gates), name = name)

    return activation

def instance_norm_layer(
    inputs=None, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tf.keras.layers.LayerNormalization(
        epsilon = epsilon)(inputs)
    # yet to add activation layer
    return instance_norm_layer

def conv1d_layer(
    inputs=None, 
    filters=None, 
    kernel_size=None, 
    strides = 1, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.keras.layers.Conv1D(
        filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)(inputs)

    return conv_layer

def conv2d_layer(
    inputs=None, 
    filters=None, 
    kernel_size=None, 
    strides=None, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.keras.layers.Conv2D(
        filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)(inputs)

    return conv_layer

def residual1d_block(
    inputs=None, 
    filters = 1024, 
    kernel_size = 3, 
    strides = 1,
    name_prefix = 'residule_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs = h1_glu, filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')
    
    h3 = inputs + h2_norm

    return h3

def downsample1d_block(
    inputs=None, 
    filters=None, 
    kernel_size=None, 
    strides=None,
    name_prefix = 'downsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def downsample2d_block(
    inputs=None, 
    filters=None, 
    kernel_size=None, 
    strides=None,
    name_prefix = 'downsample2d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def upsample1d_block(
    inputs=None, 
    filters=None, 
    kernel_size=None, 
    strides=None,
    shuffle_size = 2,
    name_prefix = 'upsample1d_block_'):
    
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def pixel_shuffler(inputs=None, shuffle_size = 2, name = None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs

def generator_gatedcnn(inputs=None, reuse = False, scope_name = 'generator_gatedcnn'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')
    with tf.name_scope(scope_name):
        model = tf.keras.Sequential([
                    conv1d_layer(filters=128, kernel_size=15, strides=1, activation=None, name='h1_conv', input_shape=inputs.shape),
                    conv1d_layer(filters=128, kernel_size=15, strides=1, activation=None, name='h1_conv_gates'),
                    tf.keras.layers.Lambda(lambda x: x[0]* tf.keras.activations.sigmoid(x[1]), output_shape=lambda s: s[0],name='h1_glu'),
                    # gated_linear_layer(name='h1_glu'),
                    downsample1d_block(filters=256, kernel_size=5, strides=2, name_prefix='downsample1d_block1_'),
                    downsample1d_block(filters=512, kernel_size=5, strides=2, name_prefix='downsample1d_block2_'),
                    residual1d_block(filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block1_'),
                    residual1d_block(filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block2_'),
                    residual1d_block(filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block3_'),
                    residual1d_block(filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block4_'),
                    residual1d_block(filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block5_'),
                    residual1d_block(filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block6_'),
                    upsample1d_block(filters=1024, kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample1d_block1_'),
                    upsample1d_block(filters=512, kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample1d_block2_'),
                    conv1d_layer(filters=24, kernel_size=15, strides=1, activation=None, name='o1_conv'),
                    tf.keras.layers.Permute((2, 1), name='output_transpose')
                ])
        if reuse:
            with tf.keras.backend.learning_phase_scope(0):
                # Discriminator would be reused in CycleGAN
                return model.predict(inputs)
        return model.trainable_variables,model.predict(inputs)

        # h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv')
        # h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        # h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # # Downsample
        # d1 = downsample1d_block(inputs = h1_glu, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_')
        # d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_')

        # # Residual blocks
        # r1 = residual1d_block(inputs = d2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        # r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        # r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        # r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        # r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        # r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        # # Upsample
        # u1 = upsample1d_block(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        # u2 = upsample1d_block(inputs = u1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        # # Output
        # o1 = conv1d_layer(inputs = u2, filters = 24, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv')
        # o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')
            
def discriminator(inputs=None, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.compat.v1.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        disc = tf.keras.Sequential([
        conv2d_layer(input_shape = inputs.shape, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv')
         ,conv2d_layer( filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv_gates'),
                tf.keras.layers.Lambda(lambda x: x[0]* tf.keras.activations.sigmoid(x[1]), output_shape=lambda s: s[0],name='h1_glu')

        # Downsample
        ,downsample2d_block(  filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block1_')
        , downsample2d_block(  filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block2_')
        , downsample2d_block(  filters = 1024, kernel_size = [6, 3], strides = [1, 2], name_prefix = 'downsample2d_block3_')

        # Output
        , tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid)])

        return disc.predict(inputs)

