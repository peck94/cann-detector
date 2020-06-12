import tensorflow as tf
import tempfile

from tqdm import trange

def create_model(input_shape, output_shape):
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    if input_shape[0] < 32:
        d = 32 - input_shape[0]
        input_tensor = tf.keras.layers.ZeroPadding2D(padding=d)(input_tensor)
    model = tf.keras.applications.Xception(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        pooling=None,
        classes=output_shape)
    model.compile(loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    return model

def train_baseline(model, x_train, y_train, x_test, y_test, batch_size):
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        tempfile.NamedTemporaryFile().name, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch'
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.75, patience=10, verbose=0, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=1e-5
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )

    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=batch_size)
