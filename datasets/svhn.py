import tensorflow as tf
import numpy as np
import utils
import scipy.io as spio

from tqdm import trange

def load_data():
    train = spio.loadmat('../datasets/svhn/train.mat')
    test = spio.loadmat('../datasets/svhn/test.mat')

    x_train = np.transpose(train['X'], [3, 0, 1, 2])
    x_test = np.transpose(test['X'], [3, 0, 1, 2])

    y_train = train['y'] - 1
    y_test = test['y'] - 1

    x_full = np.concatenate((x_train, x_test))
    y_full = np.concatenate((y_train, y_test))

    return x_full, y_full

def create_cann():
    # F-Net
    inputs = tf.keras.layers.Input((32, 32, 3))

    # encoder
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    norm1 = tf.keras.layers.BatchNormalization(fused=False)(pool1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(norm1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    norm2 = tf.keras.layers.BatchNormalization(fused=False)(pool2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(norm2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    norm3 = tf.keras.layers.BatchNormalization(fused=False)(pool3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(norm3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    norm4 = tf.keras.layers.BatchNormalization(fused=False)(pool4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    norm5 = tf.keras.layers.BatchNormalization(fused=False)(conv5)

    # flatten
    z = tf.keras.layers.Flatten()(norm5)
    z = tf.keras.layers.Dense(9, activation='linear')(z)

    # add constant
    z = tf.keras.layers.Lambda(lambda x: tf.concat([tf.fill([tf.shape(x)[0], 1], tf.cast(tf.convert_to_tensor(np.float32(1)), x.dtype)), x], axis=1))(z)

    f_model = tf.keras.models.Model(inputs, z)

    # G-Net
    g_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(9, activation='linear', input_shape=[10]),
            tf.keras.layers.Lambda(lambda x: tf.concat([tf.fill([tf.shape(x)[0], 1], tf.cast(tf.convert_to_tensor(np.float32(1)), x.dtype)), x], axis=1))
    ])

    return f_model, g_model

def train_cann(f_model, g_model, x_train, y_train, batch_size):
    variables = f_model.trainable_variables + g_model.trainable_variables
    optimizer = tf.keras.optimizers.Adam(1e-4)
    num_batches = int(np.ceil(x_train.shape[0] / batch_size))
    for epoch in range(20):
        losses = utils.Welford(1)
        prog = trange(num_batches)
        for b in prog:
            prog.set_description(f'Epoch {epoch+1}/20')
            start_idx = b * batch_size
            end_idx = min((b+1) * batch_size, x_train.shape[0])
            f_batch = x_train[start_idx:end_idx]
            g_batch = y_train[start_idx:end_idx]
            with tf.GradientTape() as tape:
                f_out = f_model(f_batch)
                g_out = g_model(g_batch)
                loss = utils.compute_loss(f_out, g_out)
                grads = tape.gradient(loss, variables)
            
            if np.isnan(loss) or loss > 1e3:
                raise ValueError('Loss has become invalid')

            optimizer.apply_gradients(zip(grads, variables))
            losses.update(loss.numpy())
            prog.set_postfix({'loss': losses.mean[0]})
