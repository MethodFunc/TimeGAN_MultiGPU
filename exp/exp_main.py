from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from exp.exp_base import ExpBase
from models import TimeGAN

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams["figure.figsize"] = (16, 9)
plt.style.use("bmh")

K.set_floatx('float32')


class Experience(ExpBase):
    def __init__(self, data, config):
        super().__init__(data=data, config=config)
        self.global_batch_size = None
        self.strategy = None
        self.local_batch_szie = config.train["batch_size"]

        self.cuda_setting()
        self.model = TimeGAN(config)


    def cuda_setting(self):
        num_gpus = len(tf.config.list_physical_devices('GPU'))

        if num_gpus == 1:
            self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
            self.global_batch_size = self.local_batch_szie

        elif num_gpus > 1:
            if self.config.multi_gpu:
                self.strategy = tf.distribute.MirroredStrategy()
                self.global_batch_size = self.local_batch_szie * num_gpus
            else:
                gpu_num = self.config.gpu_num

                physical_devices = tf.config.experimental.list_physical_devices('GPU')
                if physical_devices:
                    try:
                        tf.config.experimental.set_visible_devices(physical_devices[gpu_num], "GPU")
                        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
                    except Exception as err:
                        print(f"passed: {err}")
                        pass

                self.strategy = tf.distribute.OneDeviceStrategy(f"GPU:0")
                self.global_batch_size = self.local_batch_szie
        else:
            self.strategy = tf.distribute.OneDeviceStrategy("CPU:0")
            self.global_batch_size = self.local_batch_szie

    def train_model(self):
        real_series = self.real_dataset(self.data, self.config)
        random_series = self.randam_dataset()

        with self.strategy.scope():
            self.embedder, self.recovery, self.generator, self.discriminator, self.supervisor = self.model.get_basic_model()

            print("#1. Autoencoder Train")
            autoencoder = self.model.create_autoencoder()
            for step in tqdm(range(self.config.train["train_step"])):
                X_ = next(real_series)
                step_e_loss_t0 = self.autoencoder_step(self.strategy, autoencoder, self.embedder, self.recovery,
                                                       self.global_batch_size, X_)

            print("#2. Supervisor Train")
            for step in tqdm(range(self.config.train["train_step"])):
                X_ = next(real_series)
                step_g_loss_s = self.supervisor_step(self.strategy, self.embedder, self.supervisor,
                                                     self.global_batch_size, X_)

            print("Create Other Model")
            adversarial_supervised = self.model.create_adversarial_supervised()
            adversarial_emb = self.model.create_adversarial_emb()
            synthetic_model = self.model.create_synthetic_model()
            discriminator_model = self.model.create_discriminator_model()

            print("#3. Train TimeGAN")
            step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
            for step in tqdm(range(self.config.train["train_step"])):
                # Train generator (twice as often as discriminator)
                for kk in range(2):
                    X_ = next(real_series)
                    Z_ = next(random_series)

                    # Train generator
                    step_g_loss_u, step_g_loss_s, step_g_loss_v = self.gen_step(self.strategy, adversarial_supervised,
                                                                                adversarial_emb,
                                                                                synthetic_model, self.embedder,
                                                                                self.supervisor, self.generator,
                                                                                self.global_batch_size, X_, Z_)
                    # Train embedder
                    step_e_loss_t0 = self.emb_step(self.strategy, autoencoder, self.embedder, self.supervisor,
                                                   self.recovery, self.global_batch_size, X_)

                X_ = next(real_series)
                Z_ = next(random_series)
                step_d_loss = self.get_discriminator_loss(discriminator_model, adversarial_supervised,
                                                          adversarial_emb, self.global_batch_size, X_, Z_)

                if step_d_loss > 0.15:
                    step_d_loss = self.discrib_step(self.strategy, discriminator_model, adversarial_supervised,
                                                    adversarial_emb, self.global_batch_size, X_, Z_)

                if step % 1000 == 0:
                    print(f' {step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                          f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')

        self.synthetic_model = synthetic_model

    def save_model(self, model_nane, path="./model"):
        if not hasattr(self, "synthetic_model"):
            raise "Should be train_model execute first."

        if not Path(path).exists():
            Path(path).mkdir(parents=True)
        print("Save Synthetic model")
        self.synthetic_model.save(f"{path}/{model_nane}")
        print(f"Save Done : {model_nane}")

    def test_model(self):
        random_series = self.randam_dataset()
        if not hasattr(self, "synthetic_model"):
            raise "Should be train_model execute first."

        print("Generate data")
        generated_data = []
        for i in tqdm(range(int(self.n_series / self.config.train["batch_size"]))):
            Z_ = next(random_series)
            d = self.synthetic_model(Z_)
            generated_data.append(d)

        generated_data = np.array(np.vstack(generated_data))

        generated_data = (self.scale.inverse_transform(np.array(generated_data)
                                                       .reshape(-1, self.config.seq_length))
                          .reshape(-1, self.config.train["window_size"], self.config.seq_length))

        print("create comparison 1. plot")
        self.comparison_plots(generated_data)

        t_data = (self.scale.inverse_transform(self.data.reshape(-1, self.config.seq_length))).reshape(self.data.shape)

        print("create comparison 2. plot")
        self.comparison_plot2(t_data, generated_data)

        try:
            print("create base windspeed plot")
            self.base_ws_plot(t_data, generated_data)
        except:
            pass

        print("verification. PCA & tsne plot")
        self.verification(generated_data)

    def comparison_plots(self, generated_data):
        fig, axes = plt.subplots(nrows=3, ncols=3)
        axes = axes.flatten()

        synthetic = generated_data[np.random.randint(self.n_series // 2)]

        idx = np.random.randint(len(self.data) - self.config.train["window_size"])
        real = self.raw_data.iloc[idx: idx + self.config.train["window_size"]]

        for j, ticker in enumerate(self.raw_data.columns):
            (pd.DataFrame({'Real': real.iloc[:, j].values,
                           'Synthetic': synthetic[:, j]})

             .plot(ax=axes[j],
                   title=ticker,
                   secondary_y='Synthetic', style=['-', '--'],
                   lw=1))
        sns.despine()
        fig.tight_layout()
        plt.show()

    def comparison_plot2(self, t_data, generated_data):
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(32, 18))
        axes = ax.flatten()

        for i in range(len(self.raw_data.columns)):
            axes[i].plot(t_data[::self.config.train["window_size"], :, i].ravel()[-1000:], label="True")
            axes[i].plot(generated_data[::self.config.train["window_size"], :, i].ravel()[-1000:], label="GAN")
            axes[i].set_title(self.raw_data.columns[i])
            axes[i].legend()

        sns.despine()
        fig.tight_layout()
        plt.show()

    def base_ws_plot(self, t_data, generated_data):
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(32, 18))
        axes = ax.flatten()

        for i in range(1, len(self.raw_data.columns)):
            axes[i - 1].scatter(t_data[::self.config.train["window_size"], :, 0].ravel(),
                                t_data[::self.config.train["window_size"], :, i].ravel(),
                                label="True")
            axes[i - 1].scatter(generated_data[::self.config.train["window_size"], :, 0].ravel(),
                                generated_data[::self.config.train["window_size"], :, i].ravel(), label="TimeGAN")
            axes[i - 1].set_title(self.raw_data.columns[i])
            axes[i - 1].legend()

        sns.despine()
        fig.tight_layout()
        plt.show()

    def verification(self, generated_data):
        sample_size = 250
        data2 = np.array(self.data[:generated_data.shape[0]])
        idx = np.random.permutation(len(data2))[:sample_size]

        # Data preprocessing
        real_sample = np.asarray(data2)[idx]
        synthetic_sample = np.asarray(generated_data)[idx]

        synthetic_sample = self.scale.transform(synthetic_sample.reshape(-1, len(self.raw_data.columns)))

        real_sample_2d = real_sample.reshape(-1, self.config.train["window_size"])
        synthetic_sample_2d = synthetic_sample.reshape(-1, self.config.train["window_size"])

        pca = PCA(n_components=2)
        pca.fit(real_sample_2d)
        pca_real = (pd.DataFrame(pca.transform(real_sample_2d))
                    .assign(Data='Real'))
        pca_synthetic = (pd.DataFrame(pca.transform(synthetic_sample_2d))
                         .assign(Data='Synthetic'))
        pca_result = pca_real.append(pca_synthetic).rename(
            columns={0: '1st Component', 1: '2nd Component'})

        tsne_data = np.concatenate((real_sample_2d,
                                    synthetic_sample_2d), axis=0)

        tsne = TSNE(n_components=2,
                    verbose=1,
                    perplexity=40)
        tsne_result = tsne.fit_transform(tsne_data)

        tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
        tsne_result.loc[sample_size * 6:, 'Data'] = 'Synthetic'

        fig, axes = plt.subplots(ncols=2, figsize=(16, 9))

        sns.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                        hue='Data', style='Data', ax=axes[0])
        sns.despine()
        axes[0].set_title('PCA Result')

        sns.scatterplot(x='X', y='Y',
                        data=tsne_result,
                        hue='Data',
                        style='Data',
                        ax=axes[1])
        sns.despine()

        axes[1].set_title('t-SNE Result')
        fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions',
                     fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=.88)
        plt.show()
