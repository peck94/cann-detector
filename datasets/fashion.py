import tensorflow as tf
import numpy as np
import utils

from tqdm import trange

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_full = np.concatenate((x_train, x_test))
    y_full = np.concatenate((y_train, y_test))

    x_full = x_full.reshape(-1, 28, 28, 1)

    return x_full, y_full

def create_cann():
    f_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=[28, 28, 1]),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(fused=False),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(9, activation='linear'),

        tf.keras.layers.Lambda(lambda x: tf.concat([tf.fill([tf.shape(x)[0], 1], tf.cast(tf.convert_to_tensor(np.float32(1)), x.dtype)), x], axis=1))
    ])

    g_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(9, activation='linear', input_shape=[10]),
        tf.keras.layers.Lambda(lambda x: tf.concat([tf.fill([tf.shape(x)[0], 1], tf.cast(tf.convert_to_tensor(np.float32(1)), x.dtype)), x], axis=1))
    ])

    return f_model, g_model

def train_cann(f_model, g_model, x_train, y_train, batch_size):
    variables = f_model.trainable_variables + g_model.trainable_variables
    optimizer = tf.keras.optimizers.Adam(1e-3)
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

            optimizer.apply_gradients(zip(grads, variables))
            losses.update(loss.numpy())
            prog.set_postfix({'loss': losses.mean[0]})
