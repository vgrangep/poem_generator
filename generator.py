#!/usr/bin/env python3

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
import pandas as pd


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


def chunks(arr, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(arr), n):
        if (len(arr[i:i + n]) >= n):
            yield arr[i:i + n]



def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello





def main():
    path_to_file = "./poem_dataset.csv"
    print('load dataset')
    raw_data = pd.read_csv(path_to_file, encoding="utf-8")
    data = raw_data.copy()

    # parse content
    print('Parse content')
    text = ''
    for i, row in data.iterrows():
        text += row['Content']

    vocab = sorted(set(text))

    corpus = []
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    def int_to_text(ints):
        try:
            ints = ints.numpy()
        except:
            pass
        return ''.join(idx2char[ints])

    def text_to_int(text):
        return np.array([char2idx[c] for c in text])

    for _, row in data.iterrows():
        corpus.append(text_to_int(row['Content']))

    seq_length = 100

    text_as_int = []
    for row in corpus:
        chks = list((chunks(row, seq_length + 1)))
        text_as_int.append(chks)

    flatten_text_as_int_n1 = np.array(text_as_int).flatten()
    flatten_text_as_int_n2 = [np.array(flattened).flatten() for flattened in flatten_text_as_int_n1]
    flatten_text_as_int = np.array(flatten_text_as_int_n2).flatten().flatten()
    flat_list = [item for sublist in flatten_text_as_int_n2 for item in sublist]

    char_dataset = tf.data.Dataset.from_tensor_slices(flat_list)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)  # we use map to apply the  function to every entry

    # Time to model
    BATCH_SIZE = 64
    VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss)
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(data, epochs=1, callbacks=[checkpoint_callback])

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    def generate_text(model, start_string):
        # Evaluation step (generating text using the learned model)

        # Number of characters to generate
        num_generate = 800

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.05

        # Here batch size == 1
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension

            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return start_string + ''.join(text_generated)

    inp = input("Type a starting string: ")
    print(generate_text(model, inp))

if __name__ == "__main__":
    main()