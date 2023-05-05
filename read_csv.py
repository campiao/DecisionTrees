import numpy as np
import pandas as pd
from learn_dt import learn_decision_tree


def read_csv():
    csv_name = input("Nome do csv: ")
    print()
    dataframe = pd.read_csv(csv_name)
    input_data = dataframe.to_numpy()
    input_data = input_data[:, 1:]
    attributes = dataframe.columns.to_numpy()
    attributes = attributes[1:-1:]
    positives = 0
    negatives = 0
    for classification in input_data[:, -1]:
        if classification == 'Yes':
            positives += 1
        else:
            negatives += 1
    unique_values = {}
    values_of_attr = {}
    for a in attributes:
        index = np.where(attributes == a)
        unique_values[a] = len(np.unique(input_data[:, index]))
        values_of_attr[a] = np.unique(input_data[:, index])
    tree = learn_decision_tree(input_data, attributes, [], positives, negatives, unique_values,
                        values_of_attr)
    print(tree)
