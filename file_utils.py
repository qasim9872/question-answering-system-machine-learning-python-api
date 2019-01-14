import pickle as pkl

# Load the file to preprocess
def load_file(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

def load_pkl_data(filename):
    file = open(filename, 'rb')
    return pkl.load(file)

# Save the cleaned data to the given filename
def save_pkl_data(sentences, filename):
    pkl.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)