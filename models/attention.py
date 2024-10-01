from tensorflow.keras.layers import Attention

def attention_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder block
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Add attention mechanism
    attention = Attention()([conv1, pool1])

    # Decoder block
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv1), attention], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv6)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
