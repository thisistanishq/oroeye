import tensorflow as tf

class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale
    
    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

print("Attempting to load model with dummy CustomScaleLayer...")
try:
    model = tf.keras.models.load_model('model/best_model.h5', custom_objects={'CustomScaleLayer': CustomScaleLayer})
    print("Model loaded successfully with dummy CustomScaleLayer!")
except Exception as e:
    print(f"Error loading model: {e}")
