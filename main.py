from argparse import ArgumentParser
from sys import argv, exit
import numpy as np
import pandas as pd

global attribute_possible_values


def main():
    parser = ArgumentParser("Decision Tree Generator using ID3 Algorithm")
    parser.add_argument('-e', '--examples', help='CSV file name to train the learning tree')
    parser.add_argument('-t', '--tests', help='CSV file name to test the learning tree obtained')

    args = parser.parse_args()

    if len(argv) == 1:
        parser.print_help()
        exit(0)

    with open(args.examples, 'rt') as db:
        training_data = pd.read_csv(db)
        global attribute_possible_values
        attribute_possible_values = {}
        for collumn in training_data.columns:
            attribute_possible_values[collumn] = training_data[collumn].unique()
        label = training_data.columns[-1]
        tree = ID3(training_data, label)
        print('Decision Tree:')
        print_tree(tree)

    if args.tests is not None:
        '''Leitura do csv dos testes'''
        with open(args.tests, 'rt') as td:
            test_data = pd.read_csv(td)

            predictions = []
            for _, row in test_data.iterrows():
                prediction = transverse_tree(tree, row)
                predictions.append(prediction)
            correct = training_data[label].tolist()
            accuracy = 0
            for i in range(len(predictions)):
                if predictions[i] == correct[i]:
                    accuracy += 1
            accuracy /= len(training_data)
            print(f"Predictions: {predictions}\nAccuracy: {accuracy}")


def transverse_tree(tree, row):
    for attr, subtree in tree.items():
        value = row[attr]
        if isinstance(subtree, dict):
            if isinstance(value, str) and value not in subtree:
                return None  # Value not present in the tree, return None

            if isinstance(value, str):
                if value in subtree:
                    subtree = subtree[value]
                else:
                    return None  # Value not present in the tree, return None
            else:
                split_operator, split_value = list(subtree.keys())[0].split()
                if split_operator == '<=':
                    if value <= float(split_value):
                        subtree = subtree['<= ' + split_value]
                    else:
                        subtree = subtree['> ' + split_value]
                else:
                    if value > float(split_value):
                        subtree = subtree['> ' + split_value]
                    else:
                        subtree = subtree['<= ' + split_value]

            if isinstance(subtree, dict):
                return transverse_tree(subtree, row)
            else:
                return subtree[0]
        else:
            return subtree[0]


def total_entropy(examples, label, possible_lables):
    number_rows = examples.shape[0]
    entropy_value = 0

    for label_value in possible_lables:
        number_label_cases = examples[examples[label] == label_value].shape[0]
        label_entropy = 0
        if number_label_cases > 0:
            label_entropy = -(number_label_cases / number_rows) * np.log2(number_label_cases / number_rows)
        entropy_value += label_entropy

    return entropy_value


def entropy(examples, label, possible_labels):
    number_rows = examples.shape[0]
    entropy_value = 0

    for label_value in possible_labels:
        number_label_cases = examples[examples[label] == label_value].shape[0]
        label_entropy = 0
        if number_label_cases > 0:
            label_prob = number_label_cases / number_rows
            label_entropy = -(label_prob * np.log2(label_prob))
        entropy_value += label_entropy
    return entropy_value


def info_gain(attribute, examples, label, possible_labels):
    attr_possible_values = examples[attribute].unique()
    number_rows = examples.shape[0]
    attr_info_gain = 0.0

    for attr_value in attr_possible_values:
        attr_value_examples = examples[examples[attribute] == attr_value]
        attr_value_number_rows = attr_value_examples.shape[0]
        attr_value_entropy = entropy(attr_value_examples, label, possible_labels)
        attr_value_prob = attr_value_number_rows / number_rows
        attr_info_gain += attr_value_prob * attr_value_entropy

    return total_entropy(examples, label, possible_labels) - attr_info_gain


def most_info_gain(examples, label, possible_labels, possible_attributes):
    max_info_gain = -1
    max_info_attribute = None

    for attr in possible_attributes:
        attr_info_gain = info_gain(attr, examples, label, possible_labels)
        if attr_info_gain > max_info_gain:
            max_info_gain = attr_info_gain
            max_info_attribute = attr
    return max_info_attribute


def most_common_label(parent_examples):
    labels = parent_examples.iloc[:, -1]
    return labels.value_counts().idxmax()


def calculate_best_split_value(examples, attribute, label, possible_labels):
    attribute_values = sorted(examples[attribute].unique().tolist())
    best_split_value = None
    best_information_gain = float('-inf')

    middle_values = []
    for i in range(len(attribute_values) - 1):
        middle_values.append((attribute_values[i] + attribute_values[i + 1]) / 2)

    if len(attribute_values) == 1:
        # All instances have the same value for the attribute
        return attribute_values[0]

    for value in middle_values:
        less_equal = examples[examples[attribute] <= value]
        bigger = examples[examples[attribute] > value]

        q1 = len(less_equal) / len(examples)
        q2 = len(bigger) / len(examples)

        entropy1 = entropy(less_equal, label, possible_labels)
        entropy2 = entropy(bigger, label, possible_labels)

        information_gain = total_entropy(examples, label, possible_labels) - (q1 * entropy1) - (q2 * entropy2)

        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_split_value = value

    return best_split_value


def generate_branch(attribute, examples, label, possible_labels, parent_examples):
    attr_values_dict = examples[attribute].value_counts(sort=False)
    global attribute_possible_values
    possible_val = attribute_possible_values[attribute]
    for value in possible_val:
        if value not in attr_values_dict.keys():
            attr_values_dict[value] = 0
    branch = {}
    next_examples = examples.copy()  # Cria uma cópia dos exemplos

    for attr_value, positives in attr_values_dict.items():
        attr_value_examples = examples[examples[attribute] == attr_value]
        isPure = False

        for label_value in possible_labels:
            label_positives = attr_value_examples[attr_value_examples[label] == label_value].shape[0]

            if label_positives == positives:
                if label_positives == 0 and positives == 0:
                    label_value = most_common_label(parent_examples)
                branch[attr_value] = (label_value, label_positives)
                next_examples = next_examples[next_examples[attribute] != attr_value]
                isPure = True

        if not isPure:
            branch[attr_value] = ('?', -1)

    if branch:
        return branch, next_examples
    else:
        return None, None


def generate_branch_cont(attribute, examples, label, possible_labels, parent_examples):
    best_value_split = calculate_best_split_value(examples, attribute, label, possible_labels)
    less_equal = examples[examples[attribute] <= best_value_split]
    bigger = examples[examples[attribute] > best_value_split]

    next_examples = examples.copy()
    branch = {}

    isPure = False
    for label_value in possible_labels:
        label_positives = less_equal[less_equal[label] == label_value].shape[0]
        count = less_equal.shape[0]

        if label_positives == count:
            if label_positives == 0 and count == 0:
                label_value = most_common_label(parent_examples)
            branch[f"<= {best_value_split}"] = (label_value, label_positives)
            next_examples = next_examples[next_examples[attribute] > best_value_split]
            isPure = True

    if not isPure:
        branch[f"<= {best_value_split}"] = ('?', -1)

    isPure = False
    for label_value in possible_labels:
        label_positives = bigger[bigger[label] == label_value].shape[0]
        count = bigger.shape[0]

        if label_positives == count:
            if label_positives == 0 and count == 0:
                label_value = most_common_label(parent_examples)
            branch[f"> {best_value_split}"] = (label_value, label_positives)
            next_examples = next_examples[next_examples[attribute] <= best_value_split]
            isPure = True

    if not isPure:
        branch[f"> {best_value_split}"] = ('?', -1)

    if branch:
        return branch, next_examples, best_value_split
    else:
        return None, None, None


def build_tree(root, previous_attr_value, examples, label, possible_labels, parent_examples, possible_attributes):
    if examples.shape[0] != 0:
        if not possible_attributes.any():
            label_value = most_common_label(parent_examples)
            root[previous_attr_value] = (label_value, examples.shape[0])
            return
        max_info_attr = most_info_gain(examples, label, possible_labels, possible_attributes)
        remaining_attr = possible_attributes.drop(max_info_attr)
        split_value = None
        if examples[max_info_attr].dtype == 'object':
            tree, next_examples = generate_branch(max_info_attr, examples, label, possible_labels, parent_examples)
            flag = False
        else:
            tree, next_examples, split_value = generate_branch_cont(max_info_attr, examples, label, possible_labels,
                                                                    parent_examples)
            flag = True

        if previous_attr_value is not None:
            root[previous_attr_value] = {}
            root[previous_attr_value][max_info_attr] = tree
            next_node = root[previous_attr_value][max_info_attr]
        else:
            root[max_info_attr] = tree
            next_node = root[max_info_attr]

        for node, branch in list(next_node.items()):
            if branch[0] == '?':
                if flag:
                    if '<=' in node:
                        attr_value_examples = next_examples[next_examples[max_info_attr] <= split_value]
                    else:
                        attr_value_examples = next_examples[next_examples[max_info_attr] > split_value]
                else:
                    attr_value_examples = next_examples[next_examples[max_info_attr] == node]
                build_tree(next_node, node, attr_value_examples, label, possible_labels, examples, remaining_attr)


def ID3(data, label):
    training_data = data.copy()
    tree = {}
    possible_labels = training_data[label].unique()
    id_name = training_data.columns[0]
    possible_attributes = training_data.columns.drop([label, id_name])
    build_tree(tree, None, training_data, label, possible_labels, training_data, possible_attributes)
    return tree


def print_tree(tree, indent=''):
    if isinstance(tree, dict):
        for key, value in tree.items():
            if isinstance(value, dict):
                print(f'{indent}{key}:')
                print_tree(value, indent + '  ')
            else:
                print(f'{indent}{key}: {value[0]}  ({value[1]})')


if __name__ == '__main__':
    main()
