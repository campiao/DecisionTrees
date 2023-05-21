from argparse import ArgumentParser
from sys import argv, exit
import numpy as np
import pandas as pd

from decisiontree import DecisionTree


def main():
    parser = ArgumentParser("Decision Tree Generator using ID3 Algorithm")
    parser.add_argument('-e', '--examples', help='CSV file name to train the learning tree')
    parser.add_argument('-t', '--tests', help='CSV file name to test the learning tree obtained')
    parser.add_argument('-p', '--print', action='store_true', help='print the decision tree')

    args = parser.parse_args()

    if len(argv) == 1:
        parser.print_help()
        exit(0)

    with open(args.examples, 'rt') as db:
        training_data = pd.read_csv(db)
        label = training_data.columns[-1]
        tree = ID3(training_data, label)
        print(tree)


def total_entropy(examples, label, possible_lables):
    number_rows = examples.shape[0]
    entropy = 0

    for label_value in possible_lables:
        number_label_cases = examples[examples[label] == label_value].shape[0]
        label_entropy = -(number_label_cases / number_rows) * np.log2(number_label_cases / number_rows)
        entropy += label_entropy

    return entropy


def entropy(examples, label, possible_labels):
    number_rows = examples.shape[0]
    entropy = 0

    for label_value in possible_labels:
        number_label_cases = examples[examples[label] == label_value].shape[0]
        label_entropy = 0
        if number_label_cases != 0:
            label_prob = number_label_cases / number_rows
            label_entropy = -label_prob * np.log2(label_prob)
        entropy += label_entropy
    return entropy


def info_gain(attribute, examples, label, possible_labels):
    attr_possible_values = examples[attribute].unique()
    number_rows = examples.shape[0]
    attr_info_gain = 0.0
    print(number_rows)

    for attr_value in attr_possible_values:
        attr_value_examples = examples[examples[attribute] == attr_value]
        attr_value_number_rows = attr_value_examples.shape[0]
        attr_value_entropy = entropy(attr_value_examples, label, possible_labels)
        attr_value_prob = attr_value_number_rows / number_rows
        attr_info_gain += attr_value_prob * attr_value_entropy

    return total_entropy(examples, label, possible_labels) - attr_info_gain


def most_info_gain(examples, label, possible_labels):
    possible_attributes = examples.columns.drop([label, 'ID'])
    print(possible_attributes)

    max_info_gain = -1
    max_info_attribute = None

    for attr in possible_attributes:
        attr_info_gain = info_gain(attr, examples, label, possible_labels)
        if attr_info_gain > max_info_gain:
            max_info_gain = attr_info_gain
            max_info_attribute = attr

    return max_info_attribute


def generate_branch(attribute, examples, label, possible_labels):
    attr_values_dict = examples[attribute].value_counts(sort=False)
    branch = {}

    for attr_value, positives in attr_values_dict.iteritems():
        attr_value_examples = examples[examples[attribute] == attr_value]
        isPure = False

        for label_value in possible_labels:
            label_positives = examples[examples[label] == label_value].shape[0]

            if label_positives == positives:
                branch[attr_value] = label_value
                examples = examples[examples[attribute] != attr_value]
                isPure = True

        if not isPure:
            branch[attr_value] = '?'

    return branch, examples


def build_tree(root, previous_attr_value, examples, label, possible_labels):
    if examples.shape[0] != 0:
        max_info_attr = most_info_gain(examples, label, possible_labels)
        tree, examples = generate_branch(max_info_attr, examples, label, possible_labels)
        next_node = None

        if previous_attr_value is not None:
            root[previous_attr_value] = {}
            root[previous_attr_value][max_info_attr] = tree
            next_node = root[previous_attr_value][max_info_attr]
        else:
            root[max_info_attr] = tree
            next_node = root[max_info_attr]

        for node, branch in list(next_node.items()):
            if branch == '?':
                attr_value_examples = examples[examples[max_info_attr] == node]
                build_tree(next_node, node, attr_value_examples, label, possible_labels)


def ID3(data, label):
    training_data = data.copy()
    tree = {}
    possible_labels = training_data[label].unique()
    build_tree(tree, None, training_data, label, possible_labels)
    return tree


if __name__ == '__main__':
    main()
