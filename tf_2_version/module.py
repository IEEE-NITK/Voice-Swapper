import tensorflow as tf 
import tensorflow_addons as tfa


class Generator:
    def __init__(self) -> None:
        self.layers=[]
        self.prev_model = {}
    
    # get tensor
    def pixel_shuffler(self,inputs=None, shuffle_size = 2, name = None):
        
        shape = inputs.get_shape().as_list()
        if len(shape)==4:
            none,n,w,c = shape
        else:
            n,w,c=shape
       
        oc = c // shuffle_size
        ow = w * shuffle_size

        outputs = tf.reshape(inputs, shape = (n,ow, oc), name = name)

        return outputs

    # returns tensor
    def gated_linear_layer(self,inputs=None, gates=None, name = None):

        activation = tf.math.multiply(x = inputs, y = tf.sigmoid(gates), name = name)
        #returns element wise multiplication
        
        return activation

    # returns instance norm layer
    def instance_norm_layer(self,
        inputs=None, 
        epsilon = 1e-06, 
        activation_fn = None, 
        name = None):

        # instance_norm_layer = tf.keras.layers.LayerNormalization(
            # epsilon = epsilon)
        instance_norm_layer = tfa.layers.InstanceNormalization(axis=1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.layers.append(instance_norm_layer)
        # yet to add activation layer
        return instance_norm_layer

    # returns conv layer
    def conv1d_layer(self,
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
                name = name)
        self.layers.append(conv_layer)
        return conv_layer

    # returns tensor
    def residual1d_block(self,
        inputs=None, 
        filters = 1024, 
        kernel_size = 3, 
        strides = 1,
        name_prefix = 'residule_block_'):

        h1 = self.conv1d_layer(filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')(inputs)
        h1_norm = self.instance_norm_layer(activation_fn = None, name = name_prefix + 'h1_norm')(h1)

        h1_gates = self.conv1d_layer(filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')(inputs)
        h1_norm_gates = self.instance_norm_layer(activation_fn = None, name = name_prefix + 'h1_norm_gates')(h1_gates)

        h1_glu = self.gated_linear_layer(h1_norm,h1_norm_gates)

        h2= self.conv1d_layer(filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')(h1_glu)
        h2_norm = self.instance_norm_layer(activation_fn = None, name = name_prefix + 'h2_norm')(h2)
        
        h3 = tf.keras.layers.Add()([inputs , h2_norm])

        return h3

    # returns tensor
    def downsample1d_block(self,
        inputs=None, 
        filters=None, 
        kernel_size=None, 
        strides=None,
        name_prefix = 'downsample1d_block_'):

        h1 = self.conv1d_layer(filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')(inputs)
        h1_norm = self.instance_norm_layer(activation_fn = None, name = name_prefix + 'h1_norm')(h1)

        h1_gates = self.conv1d_layer(filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')(inputs)
        h1_norm_gates = self.instance_norm_layer(activation_fn = None, name = name_prefix + 'h1_norm_gates')(h1_gates)
        
        h1_glu = self.gated_linear_layer(inputs=h1_norm,gates=h1_norm_gates)

        return h1_glu

    # returns tensor
    def upsample1d_block(self,                
        inputs=None, 
        filters=None, 
        kernel_size=None, 
        strides=None,
        shuffle_size = 2,
        name_prefix = 'upsample1d_block_'):
        
        h1 = self.conv1d_layer(filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')(inputs)
        h1_shuffle = self.pixel_shuffler(inputs=h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
        h1_norm = self.instance_norm_layer(activation_fn = None, name = name_prefix + 'h1_norm')(h1_shuffle)

        h1_gates= self.conv1d_layer(filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')(inputs)
        h1_shuffle_gates = self.pixel_shuffler(inputs=h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
        h1_norm_gates = self.instance_norm_layer(activation_fn = None, name = name_prefix + 'h1_norm_gates')(h1_shuffle_gates)

        h1_glu = self.gated_linear_layer(inputs=h1_norm,gates=h1_norm_gates)

        return h1_glu

    def reuse_model(self,name):
        return self.prev_model[name]


    # returns model
    def generator_gatedcnn(self,input_shape=[10,24,30], reuse = False, scope_name = 'generator_gatedcnn'):
        if reuse:
            return self.reuse_model(scope_name)

        # inputs has shape [batch_size, num_features, time]
        # we need to convert it to [batch_size, time, num_features] for 1D convolution
        input_layer = tf.keras.layers.Input(shape=input_shape[1:],batch_size=input_shape[0],dtype=tf.dtypes.float64)
        permuted_input = tf.keras.layers.Permute((2, 1), name = 'input_transpose')(input_layer)
        h1=self.conv1d_layer(filters=128, kernel_size=15, strides=1,name='h1_conv')(permuted_input)
        h1_gates = self.conv1d_layer(filters=128, kernel_size=15, strides=1,name='h1_conv_gates')(h1)
        h1_glu = self.gated_linear_layer(inputs=h1,gates=h1_gates,name='h1_glu')
        
        d1 = self.downsample1d_block(inputs=h1_glu,filters=256, kernel_size=5, strides=2, name_prefix='downsample1d_block1_')
        d2 = self.downsample1d_block(inputs=d1,filters=512, kernel_size=5, strides=2, name_prefix='downsample1d_block2_')
        
        r1 = self.residual1d_block(inputs = d2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = self.residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = self.residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = self.residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = self.residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = self.residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        u1 = self.upsample1d_block(inputs=r6,filters=512, kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample1d_block1_')
        u2 = self.upsample1d_block(inputs=u1,filters=256, kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample1d_block2_')
        
        o1 = self.conv1d_layer(filters=24, kernel_size=15, strides=1,name='o1_conv')(u2)
        o2 = tf.keras.layers.Permute((2, 1), name = 'output_transpose')(o1)
        model = tf.keras.models.Model(inputs=input_layer,outputs=o2)
        
        self.prev_model[scope_name] = model
        return model

class Discriminator:
    def __init__(self) -> None:
        self.layers = []
        self.prev_model = {}

    # returns conv layer
    def conv2d_layer(self,
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
                name = name)
        self.layers.append(conv_layer)
        return conv_layer

    # returns instance norm layer
    def instance_norm_layer(self,
        inputs=None, 
        epsilon = 1e-06, 
        activation_fn = None, 
        name = None):

        instance_norm_layer = tf.keras.layers.LayerNormalization(
            epsilon = epsilon)
        self.layers.append(instance_norm_layer)
        # yet to add activation layer
        return instance_norm_layer

     # returns tensor
    
    def gated_linear_layer(self,inputs=None, gates=None, name = None):

        activation = tf.math.multiply(x = inputs, y = tf.sigmoid(gates), name = name)
        #returns element wise multiplication
        
        return activation

    # returns tensor
    def downsample2d_block(self,
        inputs=None, 
        filters=None, 
        kernel_size=None, 
        strides=None,
        name_prefix = 'downsample2d_block_'):

        h1 = self.conv2d_layer( filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')(inputs)
        h1_norm = self.instance_norm_layer( activation_fn = None, name = name_prefix + 'h1_norm')(h1)
        h1_gates = self.conv2d_layer( filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')(inputs)
        h1_norm_gates = self.instance_norm_layer( activation_fn = None, name = name_prefix + 'h1_norm_gates')(h1_gates)
        h1_glu = self.gated_linear_layer(inputs=h1_norm,gates=h1_norm_gates)

        return h1_glu

    def reuse_model(self,name):
        return self.prev_model[name]

    def build_discriminator(self,input_shape=[10,24,30],inputs=None,reuse = False, scope_name = 'discriminator'):
        if reuse:
            return self.reuse_model(scope_name)
        # inputs has shape [batch_size, num_features, time]
        # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
        if inputs is not None:
            inputs = tf.expand_dims(inputs, -1)
            input_layer = tf.keras.layers.Input(shape=inputs.shape[1:],batch_size=inputs.shape[0])
        else:
            input_layer = tf.keras.layers.Input(shape=[*input_shape[1:],1],batch_size=input_shape[0])

        h1 = self.conv2d_layer(filters = 128, kernel_size = (3, 3), strides = (1, 2), activation = None, name = 'h1_conv')(input_layer)
        h1_gates = self.conv2d_layer( filters = 128, kernel_size = (3, 3), strides = (1, 2), activation = None, name = 'h1_conv_gates')(input_layer)
        h1_glu = self.gated_linear_layer(inputs=h1,gates=h1_gates)

        # Downsample
        h2 = self.downsample2d_block(inputs=h1_glu,filters = 256, kernel_size = (3, 3), strides = (2, 2), name_prefix = 'downsample2d_block1_')
        h3 = self.downsample2d_block(inputs=h2,filters = 512, kernel_size = (3, 3), strides = (2, 2), name_prefix = 'downsample2d_block2_')
        h4 = self.downsample2d_block(inputs=h3,filters = 1024, kernel_size = (6, 3), strides = (1, 2), name_prefix = 'downsample2d_block3_')

        # Output
        o1 = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid)(h4)
        
        model = tf.keras.models.Model(inputs=input_layer,outputs=o1)
        self.prev_model[scope_name] = model

        return model

if __name__=="__main__":
    example_input = tf.keras.Input(shape=[24,128],batch_size=10)
    generator = Generator()
    output = generator.generator_gatedcnn(input_shape=example_input.shape)
    output.summary()
    print("Successfully built generator")
    
    fake = output(example_input)

    disc = Discriminator()
    model = disc.build_discriminator(input_shape=example_input.shape,inputs=fake)
    model.summary()
    print("Discriminator output shape:",model(fake).get_shape().as_list())
    # tf.input in tensorflow 2.0
