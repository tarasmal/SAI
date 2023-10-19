import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = to_categorical(y_test, 10)

loaded_model = tf.keras.models.load_model('model10')
validation = x_test[-5:]
result = loaded_model.predict(validation)
print(y_test[-5:])
print(result)