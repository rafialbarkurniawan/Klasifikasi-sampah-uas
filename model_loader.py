import tensorflow as tf

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Memuat model
model = load_model('./model/efficiennetB1.h5')
