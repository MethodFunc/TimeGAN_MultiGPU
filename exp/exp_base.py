import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm


class ExpBase:
    def __init__(self, data, config):
        self.raw_data = data
        self.config = config

        self.scale = MinMaxScaler()

        self.set_optim_loss()

        self.scaled_data = self.scale.fit_transform(self.raw_data).astype(np.float32)

        self.get_n_series()

    def get_n_series(self):
        data = []
        for i in tqdm(range(len(self.scaled_data) - self.config.train["window_size"])):
            data.append(self.scaled_data[i:i + self.config.train["window_size"]])

        self.n_series = len(data)
        self.data = np.array(data)

    @staticmethod
    def real_dataset(data, config):
        dataset = tf.data.Dataset.from_tensor_slices(data) \
            .shuffle(buffer_size=config.train["window_size"]) \
            .batch(config.train["batch_size"], drop_remainder=True)

        return iter(dataset.repeat())

    @staticmethod
    def random_data(seq_len, n_seq):
        while True:
            yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))

    def randam_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.random_data,
                                                 args=(self.config.train["window_size"], self.config.seq_length,),
                                                 output_types=tf.float32) \
            .batch(self.config.train["batch_size"], drop_remainder=True) \
            .repeat()

        return iter(dataset)

    def set_optim_loss(self):
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        lr = float(self.config.train["lr"]) if isinstance(self.config.train["lr"], str) else self.config.train["lr"]
        beta1 = self.config.train["beta1"]
        beta2 = self.config.train["beta2"]

        self.autoencoder_optim = tf.keras.optimizers.Adam(learning_rate=lr,
                                                          beta_1=beta1,
                                                          beta_2=beta2)
        self.supoervisor_optim = tf.keras.optimizers.Adam(learning_rate=lr,
                                                          beta_1=beta1,
                                                          beta_2=beta2)
        self.generator_optim = tf.keras.optimizers.Adam(learning_rate=lr,
                                                        beta_1=beta1,
                                                        beta_2=beta2)
        self.discriminator_optim = tf.keras.optimizers.Adam(learning_rate=lr,
                                                            beta_1=beta1,
                                                            beta_2=beta2)
        self.embedding_optim = tf.keras.optimizers.Adam(learning_rate=lr,
                                                        beta_1=beta1,
                                                        beta_2=beta2)

    def train_autoencoder_init(self, autoencoder, embedder, recovery, batch_size, x):
        with tf.GradientTape() as tape:
            x_tilde = autoencoder(x)
            embedding_loss_t0 = self.compute_loss(self.mse, x, x_tilde, batch_size)
            e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        self.autoencoder_optim.apply_gradients(zip(gradients, var_list))

        return tf.sqrt(embedding_loss_t0)

    def train_supervisor(self, embedder, supervisor, batch_size, x):
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_hat_supervised = supervisor(h)
            g_loss_s = self.compute_loss(self.mse, h[:, 1:, :], h_hat_supervised[:, 1:, :], batch_size)

        var_list = supervisor.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        self.supoervisor_optim.apply_gradients(zip(gradients, var_list))

        return g_loss_s

    def train_embedder(self, autoencoder, embedder, supervisor, recovery, batch_size, x):
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_hat_supervised = supervisor(h)
            generator_loss_supervised = self.compute_loss(self.mse, h[:, 1:, :], h_hat_supervised[:, 1:, :], batch_size)

            x_tilde = autoencoder(x)
            embedding_loss_t0 = self.compute_loss(self.mse, x, x_tilde, batch_size)
            e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        self.embedding_optim.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)

    def get_discriminator_loss(self, discriminator_model, adversarial_supervised, adversarial_emb, batch_size, x, z):
        y_real = discriminator_model(x)
        discriminator_loss_real = self.compute_loss(self.bce, tf.ones_like(y_real), y_real, batch_size)

        y_fake = adversarial_supervised(z)
        discriminator_loss_fake = self.compute_loss(self.bce, tf.zeros_like(y_fake), y_fake, batch_size)

        y_fake_e = adversarial_emb(z)
        discriminator_loss_fake_e = self.compute_loss(self.bce, tf.zeros_like(y_fake_e), y_fake_e, batch_size)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.config.train["gamma"] * discriminator_loss_fake_e)

    def train_discriminator(self, discriminator, adversarial_supervised, adversarial_emb, batch_size, x, z):
        with tf.GradientTape() as tape:
            discriminator_loss = self.get_discriminator_loss(discriminator, adversarial_supervised, adversarial_emb,
                                                             batch_size, x, z)

        var_list = discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        self.discriminator_optim.apply_gradients(zip(gradients, var_list))
        return discriminator_loss

    # @tf.function(experimental_relax_shapes=True)
    # def get_generator_moment_loss(self, y_true, y_pred):
    #     y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
    #     y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
    #     g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
    #     g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
    #
    #     return g_loss_mean + g_loss_var

    @tf.function(experimental_relax_shapes=True)
    def get_generator_moment_loss(self, y_true, y_pred):
        y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
        g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
        g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
        total_loss = g_loss_mean + g_loss_var

        # 배치 크기를 얻고 손실을 복제합니다.
        batch_size = tf.shape(y_true)[0]
        return tf.repeat(total_loss, batch_size)

    def train_generator(self, adversarial_supervised, adversarial_emb, synthetic_model, embedder, supervisor, generator,
                        batch_size, x, z):
        with tf.GradientTape() as tape:
            y_fake = adversarial_supervised(z)
            generator_loss_unsupervised = self.compute_loss(self.bce, tf.ones_like(y_fake), y_fake, batch_size)

            y_fake_e = adversarial_emb(z)
            generator_loss_unsupervised_e = self.compute_loss(self.bce, tf.ones_like(y_fake_e), y_fake_e, batch_size)
            h = embedder(x)
            h_hat_supervised = supervisor(h)
            generator_loss_supervised = self.compute_loss(self.mse, h[:, 1:, :], h_hat_supervised[:, 1:, :], batch_size)

            x_hat = synthetic_model(z)
            generator_moment_loss = self.compute_loss(self.get_generator_moment_loss, x, x_hat, batch_size)
            # generator_moment_loss = self.get_generator_moment_loss(x, x_hat)

            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100 * tf.sqrt(generator_loss_supervised) +
                              100 * generator_moment_loss)

        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        self.generator_optim.apply_gradients(zip(gradients, var_list))

        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    @tf.function(experimental_relax_shapes=True)
    def autoencoder_step(self, strategy, autoencoder, embedder, recovery, batch_size, inps):
        per_replica_losses = strategy.run(self.train_autoencoder_init,
                                          args=(autoencoder, embedder, recovery, batch_size, inps,))

        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function(experimental_relax_shapes=True)
    def supervisor_step(self, strategy, embedder, supervisor, batch_size, inps):
        per_replica_losses = strategy.run(self.train_supervisor, args=(embedder, supervisor, batch_size, inps,))

        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function(experimental_relax_shapes=True)
    def gen_step(self, strategy, adversarial_supervised, adversarial_emb, synthetic, embedder, supervisor, generator,
                 batch_size, inps1,
                 inps2):
        per_replica_losses = strategy.run(self.train_generator, args=(
            adversarial_supervised, adversarial_emb, synthetic, embedder, supervisor, generator, batch_size, inps1,
            inps2,))

        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function(experimental_relax_shapes=True)
    def emb_step(self, strategy, autoencoder, embedder, supervisor, recovery, batch_size, inps):
        per_replica_losses = strategy.run(self.train_embedder,
                                          args=(autoencoder, embedder, supervisor, recovery, batch_size, inps,))

        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function(experimental_relax_shapes=True)
    def discrib_step(self, strategy, discriminator, adversarial_supervised, adversarial_emb, batch_size, inps1, inps2):
        per_replica_losses = strategy.run(self.train_discriminator,
                                          args=(
                                              discriminator, adversarial_supervised, adversarial_emb, batch_size, inps1,
                                              inps2,
                                          ))

        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @staticmethod
    def compute_loss(loss_fn, true, pred, global_batch_size):
        per_example_loss = loss_fn(true, pred)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
