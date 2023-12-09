import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
# from flask_ngrok import run_with_ngrok
import numpy as np
from PIL import Image
import base64
import io
import json

# Load dataset UwU
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_images.shape)
print(test_images.shape)
print(train_labels)

plt.imshow(train_images[69])
plt.show()


class_names = ['T-shirt', 'Broek', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Tas', 'Ankle boot']

# Create neural network model
fashion_model = tf.keras.models.Sequential()
fashion_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
fashion_model.add(tf.keras.layers.Dense(128, activation='relu'))
fashion_model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile model
fashion_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
fashion_model.fit(train_images, train_labels, epochs=3)

# accuracy test OwO
test_loss, test_accuracy = fashion_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_accuracy)

# Save model
fashion_model.save('fashion_recognition_model.keras')

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('fashion_recognition_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['imageData']

    # Convert the image data to a NumPy array
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))
    img = img.resize((28, 28))
    img.show()
    img = img.convert('L')  # Convert to grayscale
    img.show()
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28) / 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    return json.dumps({'class_name': class_name})

if __name__ == '__main__':
    app.run()
