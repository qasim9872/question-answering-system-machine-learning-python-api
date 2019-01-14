import argparse

from pickle import load
from numpy import array, argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

from file_utils import load_pkl_data

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

# Map an integer to a word


def map_int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Predict the target sequence


def predict_sequence(model, tokenizer, source):
    pred = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in pred]
    target = list()
    for i in integers:
        word = map_int_to_word(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

# Evaluate the model


def evaluate_model(model, tokenizer, source, raw_eng_dataset, raw_sparql_dataset):
    predicted, actual = list(), list()
    for i, source in enumerate(source):
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_source = raw_sparql_dataset[i], raw_eng_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' %
                  (raw_source, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())

    # Bleu Scores
    print('Bleu-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('Bleu-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('Bleu-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('Bleu-4: %f' %
          corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


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
    
    # Prepare data
    trainX = encode_sequences(eng_tokenizer, eng_length, eng_train)
    testX = encode_sequences(eng_tokenizer, eng_length, eng_test)

    model = load_model('model.h5')

    print('Testing on trained examples')
    evaluate_model(model, eng_tokenizer, trainX, eng_train, sparql_train)

    print('Testing on test examples')
    evaluate_model(model, eng_tokenizer, testX, eng_test, sparql_test)
