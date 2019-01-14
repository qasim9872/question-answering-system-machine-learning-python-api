# Training Encoder-Decoder model to represent word embeddings and finally
# save the trained model as 'model.h5'

import argparse
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from file_utils import load_pkl_data

# load a clean dataset


def load_dataset(directory, lang):
    dev = load_pkl_data(directory + "dev-" + lang + ".pkl")
    test = load_pkl_data(directory + "test-" + lang + ".pkl")
    train = load_pkl_data(directory + "train-" + lang + ".pkl")
    raw = load_pkl_data(directory + lang + ".pkl")
    return dev, test, train, raw

# fit a tokenizer


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units,
                        input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument(
        '--dataset', dest='dataset', metavar='datasetFile', help='dataset', required=True)
    requiredNamed.add_argument(
        '--output', dest='output', metavar='outputDirectory', help='dataset directory', required=True)
    args = parser.parse_args()

    datasetDir = args.dataset
    outputDir = args.output

    # Load english data
    eng_dev, eng_test, eng_train, eng_raw = load_dataset(datasetDir, "english")

    # prepare english tokenizer
    eng_tokenizer = create_tokenizer(eng_raw)
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(eng_raw)
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % (eng_length))

    # load sparql data
    sparql_dev, sparql_test, sparql_train, sparql_raw = load_dataset(
        datasetDir, "sparql")

    # prepare sparql tokenizer
    sparql_tokenizer = create_tokenizer(sparql_raw)
    sparql_vocab_size = len(sparql_tokenizer.word_index) + 1
    sparql_length = max_length(sparql_raw)
    print('Sparql Vocabulary Size: %d' % sparql_vocab_size)
    print('Sparql Max Length: %d' % (sparql_length))

    # prepare training data
    trainX = encode_sequences(eng_tokenizer, eng_length, eng_train)
    trainY = encode_sequences(sparql_tokenizer, sparql_length, sparql_train)
    trainY = encode_output(trainY, sparql_vocab_size)

    # prepare validation data
    testX = encode_sequences(eng_tokenizer, eng_length, eng_test)
    testY = encode_sequences(sparql_tokenizer, sparql_length, sparql_test)
    testY = encode_output(testY, sparql_vocab_size)

    # define model
    model = define_model(eng_vocab_size, sparql_vocab_size,
                         eng_length, sparql_length, 256)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # summarize defined model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    # fit model
    filename = 'model.h5'
    checkpoint = ModelCheckpoint(
        filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(
        testX, testY), callbacks=[checkpoint], verbose=2)
