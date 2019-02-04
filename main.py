import math
import numpy as np
import os
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, LeakyReLU, MaxPool2D, ReLU, Reshape, Softmax, UpSampling2D
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split

latent_dim = 100
models_folder = "models"

def make_generator(x_train, target):
    
    def save_generated_images(generated_images, iteration):
        images_folder = "generated_images"
        target_folder = os.path.join(images_folder, "images_" + str(target))
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        batch_size = generated_images.shape[0]
        cols = int(math.sqrt(batch_size))
        rows = math.ceil(float(batch_size) / cols)
        width = generated_images.shape[2]
        height = generated_images.shape[1]
        image_blob = np.zeros((height * rows, width * cols), dtype=generated_images.dtype)
        for index, image in enumerate(generated_images):
            i = int(index / cols)
            j = index % cols
            image_blob[ width*i:width*(i+1), height*j:height*(j+1)] = image.reshape(28, 28)
        image_blob = (image_blob*255).astype("uint8")
        Image.fromarray(image_blob).save(os.path.join(target_folder, "generated_image_"+str(iteration)+".png"))


    x_train = x_train[y_train.flatten() == target]
    img_height = x_train.shape[1]
    img_width = x_train.shape[2]

    
    generator = Sequential()
    generator.add(Dense(1024, input_dim=latent_dim))
    generator.add(BatchNormalization())
    generator.add(ReLU())
    generator.add(Dense(7*7*128))
    generator.add(BatchNormalization())
    generator.add(ReLU())
    generator.add(Reshape((7, 7, 128)))
    generator.add(UpSampling2D(2))
    generator.add(Conv2D(64, 5, padding="same"))
    generator.add(BatchNormalization())
    generator.add(ReLU())
    generator.add(UpSampling2D(2))
    generator.add(Conv2D(1, 5, padding="same"))
    generator.add(Activation("sigmoid"))


    input_shape = (img_height, img_width, 1)
    discriminator = Sequential()
    discriminator.add(Conv2D(64, 5, strides=2, padding="same", input_shape=input_shape))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Conv2D(128, 5, strides=2))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Flatten())
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.5))
    discriminator.add(Dense(1))
    discriminator.add(Activation("sigmoid"))
    discriminator.compile(Adam(lr=1e-5, beta_1=0.1), "binary_crossentropy")


    discriminator.trainable = False
    dcgan = Sequential([generator, discriminator])
    dcgan.compile(Adam(lr=2e-4, beta_1=0.5), "binary_crossentropy")


    generated_label = np.ones
    real_label = np.zeros
    start = 0
    batch_size = 100
    n_iterations = 3000
    for iteration in range(1, n_iterations+1):
        random_latent_vectors = np.random.uniform(-1, 1, (batch_size, latent_dim))
        generated_images = generator.predict(random_latent_vectors)
        stop = start + batch_size
        real_images = x_train[start : stop]
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([generated_label((batch_size, 1)), real_label((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape) # Add random noise to the labels
        d_loss = discriminator.train_on_batch(combined_images, labels)
        random_latent_vectors = np.random.uniform(-1, 1, (batch_size, latent_dim))
        misleading_targets = real_label((batch_size, 1))
        g_loss = dcgan.train_on_batch(random_latent_vectors, misleading_targets)
        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0
        if iteration % 100 == 0:
            save_generated_images(generated_images, iteration)
        print("{} iteration elapsed  d_loss: {}  g_loss: {}".format(iteration, d_loss, g_loss))
        

    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    target_models_folder = os.path.join(models_folder, "models_" + str(target))
    if not os.path.exists(target_models_folder):
        os.mkdir(target_models_folder)
    generator_path = os.path.join(target_models_folder, "generator_" + str(target) + ".h5")
    discriminator_path = os.path.join(target_models_folder, "discriminator_" + str(target) + ".h5")
    generator.save(generator_path)
    discriminator.save(discriminator_path)
    return generator


(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_width = x_train.shape[2]
img_height = x_train.shape[1]
img_channels = 1
n_classes = len(np.unique(y_train))
x_max_value = x_train.max()

x_train = x_train.reshape(-1, img_height, img_width, img_channels).astype("f") / x_max_value
x_test = x_test.reshape(-1, img_height, img_width, img_channels).astype("f") / x_max_value

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

generators = []
for target in range(n_classes):
    generator = make_generator(x_train, target)
    generators.append(generator)

def train_data_generator(batch_size):
    assert batch_size % n_classes == 0
    class_batch_size = int(batch_size / n_classes)
    features = np.empty((batch_size, img_height, img_width, img_channels), dtype="f")
    targets = np.zeros((batch_size, n_classes), dtype="i")
    while True:
        for i in range(n_classes):
            random_latent_vectors = np.random.uniform(-1, 1, (class_batch_size, latent_dim))
            predictions = generators[i].predict(random_latent_vectors)
            features[ i*class_batch_size : (i+1)*class_batch_size ] = predictions
            targets[ i*class_batch_size : (i+1)*class_batch_size, i ] = 1
        yield features, targets

input_shape = (img_height, img_width, img_channels)
filters_0 = 16
filters_1 = 32
filters_2 = 64
model = Sequential()
model.add(Conv2D(filters_0, 3, padding="same", input_shape=input_shape))
model.add(MaxPool2D(2))
model.add(Conv2D(filters_1, 3, padding="same"))
model.add(MaxPool2D(2))
model.add(Conv2D(filters_2, 3, padding="same"))
model.add(MaxPool2D(2))
model.add(GlobalAveragePooling2D())
model.add(Dense(n_classes))
model.add(Softmax())
model.compile("adam", "categorical_crossentropy", metrics=["acc"])

logs_folder = "logs"
if not os.path.exists(logs_folder):
    os.mkdir(logs_folder)

if not os.path.exists(models_folder):
    os.mkdir(models_folder)


epochs = 20
batch_size = 100
steps_per_epoch = len(x_train) // batch_size
train_generator = train_data_generator(batch_size)
mnist_model_file_path = os.path.join(models_folder, "mnist_model.h5")
model_checkpoint = ModelCheckpoint(mnist_model_file_path)
tensor_board = TensorBoard(logs_folder)
model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
 validation_data=(x_val, y_val), callbacks=[model_checkpoint, tensor_board])

loss, acc = model.evaluate(x_test ,y_test)
print("Test loss: {}".format(loss))
print("Test acc: {}".format(acc))