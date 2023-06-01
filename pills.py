# Includes
from matplotlib import widgets
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import io
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import ipywidgets as widgets
from ipywidgets import TwoByTwoLayout

# Global variables
batch_size = 32
img_height = 180
img_width = 180

# Data sets
data_dir = "Drug Vision/Data Combined"

# Dataset for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# Dataset for validation
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
class_names = train_ds.class_names

# Uncomment to check if dk what data classes got
# print(class_names)

# Just some sample pictures
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Standardise the images so won't have rubbish come out
normalization_layer = layers.Rescaling(1.0 / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
num_classes = len(class_names)

# Augmenting data
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Defining the model
model = Sequential(
    [
        data_augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.9),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, name="outputs"),
    ]
)

# Compiling the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Training the model
epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Graph it out
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()


# Prediction tool
def predict(pill_path):
    img = tf.keras.utils.load_img(pill_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This pill is most likely {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    )


t = localtime()
model.save(f"{t}.h5")
tf.keras.models.load_model(f"{t}.h5").summary()

# Actually implementing it
text = widgets.HTML(
    value="<h1>Pill Checker </h1>",
)

uploaded_file = widgets.FileUpload(accept=".jpg", multiple=False)


button = widgets.Button(
    description="Check",
    disabled=False,
    button_style="",
    tooltip="Description",
    icon="check",
)
display(button, uploaded_file)


def on_button_clicked(b):
    uploaded_filename = list(uploaded_file.value.keys())[0]
    uploaded_data = uploaded_file.value[uploaded_filename]["content"]
    with open(uploaded_filename, "wb") as f:
        f.write(uploaded_data)
    im = Image.open(f"{uploaded_filename}")
    im.show()
    predict(uploaded_f