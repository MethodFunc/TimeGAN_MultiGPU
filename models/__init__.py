import tensorflow as tf
from layers import make_rnn




def create_layers(config):
    embedder = make_rnn(config.layers["type"], config.layers["n_layers"], config.layers["hidden_dim"],
                             config.layers["hidden_dim"], "Embedder")
    recovery = make_rnn(config.layers["type"], config.layers["n_layers"], config.layers["hidden_dim"],
                             config.seq_length, "Recovery")

    generator = make_rnn(config.layers["type"], config.layers["n_layers"], config.layers["hidden_dim"],
                              config.layers["hidden_dim"], "Generator")
    discriminator = make_rnn(config.layers["type"], config.layers["n_layers"], config.layers["hidden_dim"], 1,
                                  "Discriminator")
    supervisor = make_rnn(config.layers["type"], config.layers["n_layers"] - 1, config.layers["hidden_dim"],
                               config.layers["hidden_dim"], "Supervisor")


    return embedder, recovery, generator, discriminator, supervisor