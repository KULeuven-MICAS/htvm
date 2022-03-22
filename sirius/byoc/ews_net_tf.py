import tensorflow as tf
import numpy as np

# use batch dimension of 3, instead of specyfing shape as (3,3,3)
# otherwise dimension inferred is (1,3,3,3)
shape = (3, 3)
batch_dim = 3

x1 = tf.random.uniform(shape)

add_input_1 = tf.keras.layers.Input(shape=shape, batch_size=batch_dim)
add_input_2 = tf.keras.layers.Input(shape=shape, batch_size=batch_dim)
added_inputs = tf.keras.layers.Add()([add_input_1, add_input_2]),
model = tf.keras.models.Model(inputs=[add_input_1, add_input_2],
                              outputs=added_inputs)

model.compile()
print(model)
model.save("tfmodel")


# Tensorflow will apply quantization according to representative dataset
# Providing dummy dataset here
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(244, 244)
        yield [data.astype(np.float32), data.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_saved_model("tfmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


