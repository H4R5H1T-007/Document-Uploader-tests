import sys
sys.path.append('../')
from nltk.metrics.segmentation import pk, windowdiff
from text_segmentation_model import Splitter
import pandas as pd
import json

with open("text_segmentation_test.json") as f:
    test_config = json.load(f)

def is_boundary_sentence(text):
    text = Splitter.text_to_sentences(text)
    return ''.join(['0'*(len(text) - 1), '1'])

def calculate_boundary_values(text_list):
    return ''.join([is_boundary_sentence(text) for text in text_list])

number_of_samples = test_config['number_of_samples']
dataset_location = test_config['dataset_location']
window_width = test_config['window_width']

test_dataset = pd.read_csv(dataset_location, index_col=0)
test_doc = "".join([x[0].replace('\n','') for x in test_dataset.values.tolist()][:number_of_samples])
model = Splitter(test_doc)
predicted_value = model.predict()
true_value = [x[0] for x in test_dataset.values.tolist()][:number_of_samples]
predicted = calculate_boundary_values(predicted_value)
actual = calculate_boundary_values(true_value)

print(f"The Pk value for choi dataset by using {str(model)} method is {pk(actual, predicted)}.")
# print(windowdiff(actual, predicted, window_width))