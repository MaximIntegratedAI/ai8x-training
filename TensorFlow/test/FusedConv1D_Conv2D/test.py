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
        with open(self.filename, "w+") as self.log:
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
#test_input = np.random.normal(0, 0.5, size=(7,7))
test_input = np.arange(0,63,1).reshape(9,7)/128.0

print (test_input.shape)
test_input = clamp(np.floor(test_input*128+0.5))/128.0
print (test_input.shape)
test_input = np.reshape(test_input,(1, 9, 7))
print ('Test Input shape', test_input.shape)
print('Test Input', test_input)

# Init layer kernel
k1_size = 7*6
init_kernel1 = np.linspace(-0.9, 0.9, num=k1_size, dtype=np.float32)
init_kernel1 = clamp(np.floor(init_kernel1*128+0.5))/128.0
kernel_initializer1 = tf.keras.initializers.constant(init_kernel1)

k2_size = 36*3
init_kernel2 = np.linspace(-0.7, 0.6, num=k2_size, dtype=np.float32)
init_kernel2 = clamp(np.floor(init_kernel2*128+0.5))/128.0
kernel_initializer2 = tf.keras.initializers.constant(init_kernel2)

init_bias = np.array([0.5])
bias_initializer = tf.keras.initializers.constant(init_bias)

# Create functional model
input_layer = tf.keras.Input(shape=(9,7))
#reshape1 = tf.keras.layers.Reshape(target_shape=(9,7))(input_layer)
conv1 = ai8xTF.FusedConv1D(
    filters=6,
    kernel_size=1,
    strides=1,
    padding_size=0,
    use_bias=False,
    kernel_initializer=kernel_initializer1,
    )(input_layer)
    # Conversion 1D to 2D
reshape2 = tf.keras.layers.Reshape(target_shape=(3,3,6))(conv1)#(3, 2, 6))(conv1)
conv2 = ai8xTF.FusedConv2D(
    filters=2,
    kernel_size=3,
    strides=1,
    padding_size=1,
    use_bias=False,
    kernel_initializer=kernel_initializer2,
    )(reshape2)
#flat = tf.keras.layers.Flatten()(conv1)
#model = tf.keras.Model(inputs=[input_layer], outputs=[conv1,reshape2,conv2])
model = tf.keras.Model(inputs=[input_layer], outputs=[conv2])


model.compile( optimizer = 'adam' ,
                loss = tf.keras.losses.SparseCategoricalCrossentropy ( from_logits = True ),
                metrics = ['accuracy'])

model.summary()

for layer in model.layers:
      weight = np.array((layer.get_weights()[0:1])) #weights
      # Convert to 8bit and round
      print('Weight(8-bit)=\n', clamp(np.floor(weight*128+0.5)))
      bias = (layer.get_weights()[1:2]) #bias
      print('Bias=', bias)
      tf.print(f"Layer: {layer.get_config ()['name']} \
                Wmin: {tf.math.reduce_min(weight)}, \
                Wmax: {tf.math.reduce_max(weight)}, \
                Bias min: {tf.math.reduce_min(bias)}, \
                Bias max: {tf.math.reduce_min(bias)}")


#output1,output_reshape1,output = model.predict(test_input)
output = model.predict(test_input)

# Model output
#print('Model output =', output)

# Save model
tf.saved_model.save(model,'saved_model')

saved_input = clamp(np.floor(test_input*128+0.5))
print('Input(8-bit)\n:', saved_input)
#saved_input = saved_input.swapaxes(0,2)
saved_input = saved_input.reshape(9,7)
saved_input = saved_input.swapaxes(0,1) #swap row,col
print('Saved Input(8-bit) for izer\n:', saved_input)

print(saved_input.shape)
# Save input
np.save (os.path.join(logdir, 'input_sample_7x9.npy'), np.array(saved_input, dtype=np.int32))
#print('OutputConv1(8-bit)\n:', clamp(np.floor(output1*128+0.5)))
#print(output1.shape)

#print('OutputReshape1(8-bit)\n:', clamp(np.floor(output_reshape1*128+0.5)))
#print(output_reshape1.shape)

print('Output(8-bit)\n:', clamp(np.floor(output*128+0.5)))
print(output.shape)


exit(0)
