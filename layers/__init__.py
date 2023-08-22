import tensorflow as tf

def make_rnn(rnn_type, n_layers, hidden_dim, output_dim, name):
    model = tf.keras.models.Sequential(name=name)

    for i in range(n_layers):
        if rnn_type == "simple":
            model.add(tf.keras.layers.SimpleRNN(units=hidden_dim, dropout=0.15,
                                                return_sequences=True,
                                                name=f"SimpleRNN_{i + 1}"))
        if rnn_type == "gru":
            model.add(tf.keras.layers.GRU(units=hidden_dim, dropout=0.15,
                                          return_sequences=True,
                                          name=f"GRU{i + 1}"))

        if rnn_type == "lstm":
            model.add(tf.keras.layers.LSTM(units=hidden_dim, dropout=0.15,
                                           return_sequences=True,
                                           name=f"LSTM{i + 1}"))

    model.add(tf.keras.layers.Dense(output_dim, activation="sigmoid", name="OUT"))

    return model
