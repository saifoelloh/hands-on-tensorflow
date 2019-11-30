import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# # Print one of datasets
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# # Trying to remove flatten and se the result
# # ------------------------------------------
# # Flatten our data into one layer from the initial form where is (nxn)
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# # Trying to changing number of last layer
# # ------------------------------------------
# # Last layer represent number of labels
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(5, activation=tf.nn.softmax)
# ])

# # Adding new layer between first, and second layer
# # ------------------------------------------
# # Adding new layer increase your predict, but in the simple case, there's no significant value
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#     tf.keras.layers.Dense(256, activation=tf.nn.relu),
#     tf.keras.layers.Dense(5, activation=tf.nn.softmax)
# ])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(training_images, training_labels, epochs=5)
# model.evaluate(test_images, test_labels)

classification = model.predict(test_images)
print(classification[0])
print(test_labels[0])
