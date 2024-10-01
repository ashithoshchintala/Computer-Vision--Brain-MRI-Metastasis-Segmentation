import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def compile_and_train(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=[dice_coefficient])
    
    model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

    # Evaluate the model
    score = model.evaluate(X_test, y_test)
    print(f"Model Evaluation Score: {score}")
