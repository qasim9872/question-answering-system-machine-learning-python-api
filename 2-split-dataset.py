import argparse
import random
from file_utils import load_pkl_data, save_pkl_data

TRAINING_PERCENTAGE = 80
TEST_PERCENTAGE = 20
DEV_PERCENTAGE = 0

def get_length(data):
    return len(data)

def resolve_percentage(length, percentage):
    return int(length * (percentage / 100))

def save_dev_test_train(lang, outputDir, dev, test, train):
    print("Saving data to: " + outputDir)
    save_pkl_data(dev, outputDir + "dev-" + lang + ".pkl")
    save_pkl_data(test, outputDir + "test-" + lang + ".pkl")
    save_pkl_data(train, outputDir + "train-" + lang + ".pkl")

def split_data(english, sparql, length, outputDir):
    
    # Calculate percentages to split in
    lines_to_train = resolve_percentage(length, TRAINING_PERCENTAGE)
    lines_to_test = resolve_percentage(length, TEST_PERCENTAGE)
    lines_to_dev = resolve_percentage(length, DEV_PERCENTAGE)

    # randomize the dataset
    combined = list(zip(english, sparql))
    random.shuffle(combined)
    english, sparql = zip(*combined)

    # extract the data in the right proportion
    eng_train = english[:lines_to_train]
    eng_test = english[lines_to_train: lines_to_train+lines_to_test]
    eng_dev = english[lines_to_train + lines_to_test: lines_to_train + lines_to_test + lines_to_dev]
    save_dev_test_train("english", outputDir, eng_dev, eng_test, eng_train)

    sparql_train = sparql[:lines_to_train]
    sparql_test = sparql[lines_to_train: lines_to_train+lines_to_test]
    sparql_dev = sparql[lines_to_train + lines_to_test: lines_to_train + lines_to_test + lines_to_dev]
    save_dev_test_train("sparql", outputDir, sparql_dev, sparql_test, sparql_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--dataset', dest='dataset', metavar='datasetFile', help='dataset', required=True)
    requiredNamed.add_argument('--output', dest='output', metavar='outputDirectory', help='dataset directory', required=True)
    args = parser.parse_args()

    datasetDir = args.dataset
    outputDir = args.output

    # get path to files
    pathToEnglishPklFile = datasetDir + "original_english.pkl"
    pathToSparqlPklFile = datasetDir + "original_sparql.pkl"

    # load files from path
    original_english = load_pkl_data(pathToEnglishPklFile)
    original_sparql = load_pkl_data(pathToSparqlPklFile)

    eng_length = get_length(original_english)
    sparql_length = get_length(original_sparql)

    if (eng_length == sparql_length):
        # split dataset
        split_data(original_english, original_sparql, eng_length, outputDir)
    else:
        raise Exception("The length of lines in original_english.pkl does not match original_sparql.pkl")





