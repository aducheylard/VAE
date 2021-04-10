import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#Creamos la clase VAE, proveniente de la clase 'Model' de keras
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs): #el constructor define un encoder y decoder para usar
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        #definimos las metricas
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self): #solamente retornamos las metricas de la instancia de la clase
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        #vamos a calcular la gradiente
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z) #Definimos el reconstructor como el decoder
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss #calculamos la perdida total
        grads = tape.gradient(total_loss, self.trainable_weights)#Guardamos la gradiente en una variable
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))#Optimizamos la gradiente
        #Actualizamos el estado de las metricas
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

#Funcion para graficar el espacio latente de los digitos
def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

#funcion para graficar el cluster de los digitos
def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


latent_dim = 2 # Numero de dimensiones latentes, se supone que es promeido y desv. estandar

#Creamos el modelo que actua como Encoder
encoder_inputs = keras.Input(shape=(28, 28, 1)) # Capa de input, imagen de 28x28 pixeles de 1 solo canal
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs) # Crea la primera capa oculta de 32 nodos (este es el output para la capa siguiente), con un kernel de 3x3, con la funcion de activacion de relu. Y se la pasa la variable 'endoer_inputs'
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x) # crea la segunda capa de 64 nodos, kernel de 3x3, misma funcion de activacion relu, y se le pasa la variable 'x'
x = layers.Flatten()(x) # aplana las imagenes, y se la pasa la variable 'x'
x = layers.Dense(16, activation="relu")(x) # capa densa de 16 nodos y se le pasa 'x'

'''
PREGUNTAR AL PROFE COMO 'z_mean' Y 'z_log_var' SON DISTINTAS?
'''

z_mean = layers.Dense(latent_dim, name="z_mean")(x) # retorna el promedio, y se le pasa 'x' como input...deberia ser cercano a 0
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)


z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder") #Definimos al modelo desde donde incia hasta donde termina.
encoder.summary() #Imprime el resumen del modelo

#Creamos el modelo que actua como Decoder
latent_inputs = keras.Input(shape=(latent_dim,)) #Como terminamos el encoder con 2 dimensiones (z_mean y z_log_var), como input al decoder se le deben pasar 2 dimensiones.
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs) # Primera layer de 7x7x64, con activacion de relu y 'latent_inputs' de input
x = layers.Reshape((7, 7, 64))(x) #cambiamos la dimension a 7,7,64
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x) #capa de 64 nodos, kernel de 3x3, con activacoin de relu y se le pasa 'x'
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x) #capa de 32 nodos, kernel de 3x3, con activacoin de relu y se le pasa 'x'
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x) #capa de 1 dimension, con kernel de 3x3, con activacion de sigmoid (valores entre 0 y 1)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder") #definimos al modelo que inicia desde el 'latent_input' y finaliza en 'decoder_output'
decoder.summary() #resumen del modelo

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data() #Cargamos la data del dataset
mnist_digits = np.concatenate([x_train, x_test], axis=0) #pasamos la data de entrenamiento y testeo a una sola variable
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255 #Normalizamos la data de las imagenes a valores entre 0 y 1

vae = VAE(encoder, decoder) #instanciamos el variational encoder
vae.compile(optimizer=keras.optimizers.Adam()) #Compilamos el VAE usando el optimizador de 'Adam'
vae.fit(mnist_digits, epochs=30, batch_size=128) #Entrenamos el modelo entregandole los digitos del entrenamiento, que corra durante 30 epochs, y con un tamano de muestras de 128

plot_latent_space(vae) #muestra el espacio donde se definen los digitoos

(x_train, y_train), _ = keras.datasets.mnist.load_data() #Cargamos la data de entrenamiento y sus respectivos labels
x_train = np.expand_dims(x_train, -1).astype("float32") / 255 #normalizamos la data para tener valores entr 0 y 1

plot_label_clusters(vae, x_train, y_train)