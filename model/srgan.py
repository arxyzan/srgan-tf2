from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from model.common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11


class ResBlock(layers.Layer):
    def __init__(self, num_filters, momentum=0.8):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(num_filters, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization(momentum=momentum)
        self.bn2 = BatchNormalization(momentum=momentum)
        self.prelu = PReLU(shared_axes=[1, 2])
        self.conv2 = Conv2D(num_filters, kernel_size=3, padding='same')
        self.add = Add()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([inputs, x])
        return x


class Upsample(layers.Layer):
    def __init__(self, num_filters=64):
        super().__init__()
        self.conv = Conv2D(num_filters, kernel_size=3, padding='same')
        self.shuffle = Lambda(pixel_shuffle(scale=2))
        self.prelu = PReLU(shared_axes=[1, 2])

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.shuffle(x)
        x = self.prelu(x)
        return x


class DiscriminatorBlock(layers.Layer):
    def __init__(self, num_filters, strides=1, batchnorm=True, momentum=0.8):
        super().__init__()
        self.batchnorm = batchnorm
        self.conv = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')
        self.bn = BatchNormalization(momentum=momentum)
        self.lrelu = LeakyReLU(alpha=0.2)

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        if self.batchnorm:
            x = self.bn(x)
        x = self.lrelu(x)
        return x


class Discriminator(models.Model):
    def __init__(self, num_filters=64):
        super().__init__()
        self.num_filters = num_filters
        self.normalize = Lambda(normalize_m11)
        self.discriminator1 = DiscriminatorBlock(self.num_filters, strides=1, batchnorm=False)
        self.discriminator2 = DiscriminatorBlock(self.num_filters, strides=2)
        self.discriminator3 = DiscriminatorBlock(self.num_filters * 2, strides=1)
        self.discriminator4 = DiscriminatorBlock(self.num_filters * 2, strides=2)
        self.discriminator5 = DiscriminatorBlock(self.num_filters * 4, strides=1)
        self.discriminator6 = DiscriminatorBlock(self.num_filters * 4, strides=2)
        self.discriminator7 = DiscriminatorBlock(self.num_filters * 8, strides=1)
        self.discriminator8 = DiscriminatorBlock(self.num_filters * 8, strides=2)
        self.flatten = Flatten()
        self.fc1 = Dense(1024)
        self.lrelu = LeakyReLU(alpha=0.2)
        self.fc2 = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.normalize(inputs)
        x = self.discriminator1(x)
        x = self.discriminator2(x)
        x = self.discriminator3(x)
        x = self.discriminator4(x)
        x = self.discriminator5(x)
        x = self.discriminator6(x)
        x = self.discriminator7(x)
        x = self.discriminator8(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        return x


class SRGAN(models.Model):
    def __init__(self, num_filters=64, num_res_blocks=16):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.normalize = Lambda(normalize_01)
        self.conv1 = Conv2D(num_filters, kernel_size=9, padding='same')
        self.prelu = PReLU(shared_axes=[1, 2])
        # 16 Residual blocks
        self.res_block = Sequential()
        for _ in range(num_res_blocks):
            self.res_block.add(ResBlock(num_filters))
        self.res_block = ResBlock(num_filters)
        self.conv2 = Conv2D(num_filters, kernel_size=3, padding='same')
        self.bn = BatchNormalization(momentum=0.8)
        self.conv3 = Conv2D(3, kernel_size=9, padding='same', activation='tanh')
        self.add = Add()
        self.shuffle = Lambda(pixel_shuffle(scale=2))
        self.upsample1 = Upsample(num_filters=num_filters * 4)
        self.upsample2 = Upsample(num_filters=num_filters * 4)
        self.denormalize = Lambda(denormalize_m11)

    def call(self, inputs, training=None, mask=None):
        x = self.normalize(inputs)
        x = self.conv1(x)
        x = x_1 = self.prelu(x)
        x = self.res_block(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.add([x_1, x])
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.conv3(x)
        x = self.denormalize(x)

        return x


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return models.Model(vgg.input, vgg.layers[output_layer].output)


if __name__ == '__main__':
    # test network
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.random.set_seed(1234)
    m = SRGAN()
    m(tf.ones((1, 24, 24, 3)))
    print('done')