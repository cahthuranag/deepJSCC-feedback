import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, PReLU
import tensorflow_compression as tfc

class Encoder(Layer):
    """Build encoder from specified architecture"""

    def __init__(self, conv_depth, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 256
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters,
                (9, 9),
                name="layer_0",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_0"),
            ),
            PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_1"),
            ),
            PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_2"),
            ),
            PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_3",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_3"),
            ),
            PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                conv_depth,
                (5, 5),
                name="layer_out",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


class Decoder(Layer):
    """Build decoder from specified architecture"""

    def __init__(self, n_channels, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 256
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_out",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_out", inverse=True),
            ),
            PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_0",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True),
            ),
            PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True),
            ),
            PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_2",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                n_channels,
                (9, 9),
                name="layer_3",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.sigmoid,
            ),
        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


def real_awgn(x, stddev):
    """Implements the real additive white Gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # Additive white Gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn
    return y


# Load and preprocess the dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images to match the input shape of the encoder and decoder
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Set hyperparameters
num_epochs = 10
batch_size = 32
noise_stddev = 0.1

# Create the encoder, decoder, and optimizer
encoder = Encoder(conv_depth=32)
decoder = Decoder(n_channels=1)
optimizer = Adam()

# Define the loss function and metrics
loss_fn = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanSquaredError()]

# Training loop
for epoch in range(num_epochs):
    print("Epoch:", epoch + 1)
    num_batches = x_train.shape[0] // batch_size
    epoch_loss = tf.keras.metrics.Mean()

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size

        x_batch = x_train[start_idx:end_idx]
        with tf.GradientTape() as tape:
            encoded = encoder(x_batch)
            transmitted = real_awgn(encoded, noise_stddev)
            reconstructed = decoder(transmitted)
            loss = loss_fn(x_batch, reconstructed)

        gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))

        epoch_loss(loss)

    print("Loss:", epoch_loss.result())

# Transmit a sample image and obtain the output
sample_image = x_test[0:1]
encoded_image = encoder(sample_image)
transmitted_image = real_awgn(encoded_image, noise_stddev)
reconstructed_image = decoder(transmitted_image)

print("Original Image Shape:", sample_image.shape)
print("Reconstructed Image Shape:", reconstructed_image.shape)

