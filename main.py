import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout, Input
from keras.optimizers import Adam
import glob
from PIL import Image


images = []
for f in glob.iglob("./processed_celeba_small/celeba/imgs/*"):
    images.append(np.asarray(Image.open(f)))

images = np.array(images)
images = (images - 127.5) / 127.5


image_shape = (64, 64, 3)
noise_dim = 100

def build_generator():
    model = Sequential()

    model.add(Dense(256 * 8 * 8, input_dim=noise_dim))
    model.add(Reshape((8, 8, 256)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))

    noise = Input(shape=(noise_dim,))
    image = model(noise)

    return Model(noise, image)


def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=image_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)
    validity = model(image)

    return Model(image, validity)




# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator()

# Generate a random image using the generator
noise = Input(shape=(noise_dim,))
generated_image = generator(noise)

# Only train the generator in combined model
discriminator.trainable = False

# Determine the validity of the generated image
validity = discriminator(generated_image)

# Combine the generator and discriminator into one model
combined = Model(noise, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))


def save_generated_samples(epoch):
    rows, cols = 5, 5
    noise = np.random.normal(0, 1, (rows * cols, noise_dim))
    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5  # Rescale images to 0-1 range

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(generated_images[idx])
            axs[i, j].axis('off')
            idx += 1
    fig.savefig(f"generated_samples/epoch_{epoch}.png")
    plt.close()

def train_gan(epochs, batch_size, sample_interval):
    # Create labels for real and fake images
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_images = images[idx]
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        generated_images = generator.predict(noise)

        discriminator_loss_real = discriminator.train_on_batch(real_images, valid)
        discriminator_loss_fake = discriminator.train_on_batch(generated_images, fake)
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        generator_loss = combined.train_on_batch(noise, valid)

        # Print the progress
        print(
            f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}")

        # Save generated samples at the sample interval
        if epoch % sample_interval == 0:
            save_generated_samples(epoch)

epochs = 20000
batch_size = 128
sample_interval = 1000

train_gan(epochs, batch_size, sample_interval)




































