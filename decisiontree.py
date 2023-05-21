import math
from copy import deepcopy

import numpy as np


class DecisionTree:

    def __init__(self, examples, attributes, label):
        self.label = label
        self.transformations = {}
        self.possibleLabelValues = np.unique(examples[:,-1])
        self.examples = examples
        self.root = None

    def ID3(self,examples,attributes,chosen_attr):
        if not np.any(examples):
            return plurality_value()


