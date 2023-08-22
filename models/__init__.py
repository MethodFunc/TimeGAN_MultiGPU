import tensorflow as tf
from layers import make_rnn


class TimeGAN:
    def __init__(self, config):
        self.config = config
        self.X = tf.keras.layers.Input(shape=(self.config.train["window_size"], self.config.seq_length),
                                       name="RealData")
        self.Z = tf.keras.layers.Input(shape=(self.config.train["window_size"], self.config.seq_length),
                                       name="RandomData")

        self.embedder = make_rnn(self.config.layers["type"], self.config.layers["n_layers"],
                                 config.layers["hidden_dim"],
                                 config.layers["hidden_dim"], "Embedder")
        self.recovery = make_rnn(config.layers["type"], config.layers["n_layers"], config.layers["hidden_dim"],
                                 config.seq_length, "Recovery")

        self.generator = make_rnn(config.layers["type"], config.layers["n_layers"], config.layers["hidden_dim"],
                                  config.layers["hidden_dim"], "Generator")
        self.discriminator = make_rnn(config.layers["type"], config.layers["n_layers"], config.layers["hidden_dim"], 1,
                                      "Discriminator")
        self.supervisor = make_rnn(config.layers["type"], config.layers["n_layers"] - 1, config.layers["hidden_dim"],
                                   config.layers["hidden_dim"], "Supervisor")

    def get_basic_model(self):
        return self.embedder, self.recovery, self.generator, self.discriminator, self.supervisor

    def create_autoencoder(self):
        H = self.embedder(self.X)
        X_tilde = self.recovery(H)

        return tf.keras.models.Model(inputs=self.X, outputs=X_tilde)

    def create_adversarial_supervised(self):
        E_hat = self.generator(self.Z)
        H_hat = self.supervisor(E_hat)
        Y_fake = self.discriminator(H_hat)

        # set adversarial supervised
        adversarial_supervised = tf.keras.models.Model(inputs=self.Z,
                                                       outputs=Y_fake,
                                                       name='AdversarialNetSupervised')

        return adversarial_supervised

    def create_adversarial_emb(self):
        E_hat = self.generator(self.Z)
        Y_fake_e = self.discriminator(E_hat)
        adversarial_emb = tf.keras.models.Model(inputs=self.Z,
                                                outputs=Y_fake_e,
                                                name='AdversarialNet')

        return adversarial_emb

    def create_discriminator_model(self):
        H = self.embedder(self.X)
        Y_real = self.discriminator(H)
        discriminator_model = tf.keras.models.Model(inputs=self.X,
                                                    outputs=Y_real,
                                                    name='DiscriminatorReal')

        return discriminator_model

    def create_synthetic_model(self):
        E_hat = self.generator(self.Z)
        H_hat = self.supervisor(E_hat)
        X_hat = self.recovery(H_hat)
        synthetic_data = tf.keras.models.Model(inputs=self.Z, outputs=X_hat, name="SyntheticData")

        return synthetic_data
