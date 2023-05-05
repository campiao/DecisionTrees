import math
import random

from read_csv import np


def learn_decision_tree(examples, attributes, parent_examples, pos, neg, unique_v,
                        values_of_attr):
    if examples is None:
        return plurality_value(parent_examples)
    elif all_examples_same_class(examples):
        return examples[0, :][:, 1]
    elif not np.any(attributes):
        return plurality_value(examples)
    else:
        max_attr = [2, ""]
        for a in attributes:
            gain = importance(a, examples, pos, neg, unique_v)
            if gain < max_attr[0]:
                max_attr[0] = gain
                max_attr[1] = a
        A = max_attr[1]
        tree = [(A, None)]
        for value in values_of_attr[A]:
            exa = None
            for exe in examples:
                index = np.where(attributes == A)
                if exe[index] == value:
                    if exa is None:
                        exa = np.array(exe)
                        continue
                    exa = np.vstack((exa, exe))
            new = attributes[attributes != A]
            subtree = learn_decision_tree(exa, new, examples, pos, neg, unique_v,
                                          values_of_attr)
            tree.append((f"{A}=={value}", subtree))
        return tree


def importance(attribute, examples, positives, negatives, unique_values):
    entropy = 0
    d = unique_values[attribute]
    pk = nk = 0
    for exa in examples:
        exa = exa[-1]
        if exa == 'Yes':
            pk += 1
        else:
            nk += 1
    for i in range(1, d + 1):
        entropy += (pk + nk) / (positives + negatives) * boolean_random_variable_entropy(pk / float(pk + nk))

    return boolean_random_variable_entropy(positives / (positives + negatives)) + entropy


def boolean_random_variable_entropy(prob):
    return (prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))


def plurality_value(examples):
    pos = 0
    neg = 0
    res = ['Yes', 'No']
    for output in examples:
        output = output[-1]
        if output == 'Yes':
            pos += 1
        else:
            neg += 1
    valor_max = max(pos, neg)
    if pos == neg:
        return random.choice(res)
    if pos == valor_max:
        return 'Yes'
    return 'No'


def all_examples_same_class(examples):
    classification = None
    for exa in examples:
        exa = exa[-1]
        if classification is None:
            classification = exa
            continue
        if exa != classification:
            return False
    return True
