import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from random import randint
import os
import sys
sys.path.append('../../')
import ai8xTF

ai8xTF.set_device (85 , False , 10 )

# create a class for logging screen to file
# make sure to change verbose=2 in fit and evaluate
class Logger():
    """
    stdout logger
    """

    def __init__(self, filename):
        """
        init
        """
        self.terminal = sys.stdout
        self.filename = filename
        self.log = None

    def write(self, message):
        """
        write to file
        """
        with open(self.filename, "a") as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        """
        this flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.
        """
        pass  # pylint: disable=unnecessary-pass

def clamp(x, min=-128,max=127):
    return np.array(tf.clip_by_value(x, min, max))

# following piece it to init seed to make repeated results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(10)
tf.random.set_seed(7)

logdir = 'saved_model'

if not os.path.isdir(logdir):
    os.makedirs(logdir)

# Log stdout to file
sys.stdout = Logger(os.path.join(logdir, 'result.log'))

# Init input samples
test_input = np.array([[-0.7,-0.6,-0.5,-0.4,-0.3],[-0.2,-0.1,0,0.1,0.2],[0.3,0.4,0.5,0.6,0.7]])
print ('Test Input shape', test_input.shape)
test_input = clamp(np.floor(test_input*128+0.5))/128.0
test_input = test_input.reshape(1, 3, 5)
print ('Test Input shape', test_input.shape)
print('Test Input', test_input)

# Init layer kernel

k1 = np.linspace(-0.3, 0.3, num=18, dtype=np.float32)
k1 = clamp(np.floor(k1*128+0.5))/128.0

k2 = np.linspace(-0.1, 0.1, num=36, dtype=np.float32)
k2 = clamp(np.floor(k2*128+0.5))/128.0


#k5 = np.linspace(-0.5, 0.5, num=160, dtype=np.float32)
#k5 = np.random.normal(0, 0.5, size=160)
#k5 = clamp(np.floor(k5*128+0.5))/128.0

init_bias = np.array([-0.05, 0.05])
init_bias = clamp(np.floor(init_bias*128+0.5))/128.0
bias_initializer = tf.keras.initializers.constant(init_bias)

# Create functional model
input_layer = tf.keras.Input(shape=(3, 5))
reshape = tf.keras.layers.Reshape(target_shape=(3, 5, 1))(input_layer)

conv1 = ai8xTF.FusedConv2D(
    filters=2,
    kernel_size=3,
    strides=1,
    padding_size=1,
    use_bias=False,
    kernel_initializer=tf.keras.initializers.constant(k1)
    )(reshape)

conv2 = ai8xTF.FusedConv2D(
    filters=2,
    kernel_size=3,
    strides=1,
    padding_size=1,
    use_bias=False,
    kernel_initializer = tf.keras.initializers.constant(k2)
    )(conv1)
"""
conv3 = ai8xTF.FusedMaxPoolConv2(
    filters=1,
    kernel_size=3,
    strides=1,
    padding_size=1,
    pool_size=2,
    pool_strides=2,
    use_bias=False,
    kernel_initializer=tf.keras.initializers.constant(k3)
    )(conv2)

conv4 = ai8xTF.FusedAvgPoolConv2DReLU(
    filters=1,
    kernel_size=3,
    strides=1,
    padding_size=1,
    pool_size=2,
    pool_strides=2,
    use_bias=False,
    kernel_initializer=tf.keras.initializers.constant(k4)
    )(conv3)
"""
#flat = tf.keras.layers.Flatten()(conv1)
#output_layer = ai8xTF.FusedDense(5, wide=True, kernel_initializer=tf.keras.initializers.constant(k5))(flat)

model = tf.keras.Model(inputs=[input_layer], outputs=[conv1, conv2])


model.compile( optimizer = 'adam' ,
                loss = tf.keras.losses.SparseCategoricalCrossentropy ( from_logits = True ),
                metrics = ['accuracy'])

model.summary()

for layer in model.layers:
      weight = np.array((layer.get_weights()[0:1])) #weights
      # Convert to 8bit, round and clamp
      print('Weight(8-bit)=\n', clamp(np.floor(weight*128+0.5)))
      print('Weight(8-bit)shape=\n', weight.shape)
      bias = np.array((layer.get_weights()[1:2])) #bias
      # Convert to 8bit, round and clamp
      print('Bias(8-bit)=\n', clamp(np.floor(bias*128+0.5)))
      print('Bias(8-bit)shape=\n', bias.shape)
      tf.print(f"Layer: {layer.get_config ()['name']} \
                Wmin: {tf.math.reduce_min(weight)}, \
                Wmax: {tf.math.reduce_max(weight)}, \
                Bias min: {tf.math.reduce_min(bias)}, \
                Bias max: {tf.math.reduce_min(bias)}")

conv1_out, output = model.predict(test_input)

# Model output
print('Conv1 output=\n', conv1_out)
print('Conv2 output=\n', output)

# Save model
tf.saved_model.save(model,'saved_model')

# Convert to 8bit, round and clamp
saved_input = clamp(np.floor(test_input*128+0.5))
print('Input(8-bit):\n', saved_input)
print('Input(8-bit) shape', saved_input.shape)
# Save input
np.save (os.path.join(logdir, 'input_sample_1x3x5.npy'), np.array(saved_input, dtype=np.int32))

# Convert to 8bit, round and clamp
print('Conv1 output(8-bit):\n', clamp(np.floor(conv1_out*128+0.5)))
print('Conv1 output(8-bit) shape', conv1_out.shape)
print('Conv2 output(8-bit):\n', clamp(np.floor(output*128+0.5)))
print('Conv2 output(8-bit) shape', output.shape)

exit(0)