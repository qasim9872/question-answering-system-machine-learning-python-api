# This file converts the format of the file to pkl which can be used by keras
import argparse
import numpy as np
import pickle as pkl
from file_utils import load_file, save_pkl_data

def load_parse_file(filename):
    text = load_file(filename)
    lines = text.split('\n')
    return np.array(lines)

# The input data is in a file with extension .en
# The output data is in a file with extension .sparql

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--dataset', dest='dataset', metavar='datasetFile', help='dataset', required=True)
    requiredNamed.add_argument('--output', dest='output', metavar='outputDirectory', help='dataset directory', required=True)
    args = parser.parse_args()

    datasetDir = args.dataset
    outputDir = args.output

    dataFilePrefix = "data_300"
    englishDataFileName = dataFilePrefix + ".en"
    sparqlDataFileName = dataFilePrefix + ".sparql"

    pathToEnglishDataFile = datasetDir + englishDataFileName
    pathToSparqlDataFile = datasetDir + sparqlDataFileName

    print("path to english file: " + pathToEnglishDataFile)
    print("path to sparql file: " + pathToSparqlDataFile)

    sparql_lines = load_parse_file(pathToSparqlDataFile)
    save_pkl_data(sparql_lines, "original_sparql.pkl")

    english_lines = load_parse_file(pathToEnglishDataFile)
    save_pkl_data(english_lines, "original_english.pkl")








