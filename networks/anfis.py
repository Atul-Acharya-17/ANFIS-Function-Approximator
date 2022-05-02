from typing import List
import torch
from networks.fuzzification import FuzzificationLayer
from networks.input import InputLayer
from networks.rule import RuleLayer
from networks.normalization import NormalizationLayer
from networks.defuzzification import DefuzzificationLayer
from networks.aggregation import AggregationLayer


class ANFIS(torch.nn.Module):

    def __init__(self, input_features=2, num_membership_functions=2, membership_funcs_params=[]):
        super(ANFIS, self).__init__()

        self.input_features = input_features
        self.num_membership_functions = num_membership_functions

        self.input_layer = InputLayer()

        self.fuzzification_layers = torch.nn.ModuleList()

        for i in range(input_features):
            fuzzification_layer = torch.nn.ModuleList()
            for j in range(num_membership_functions):
                parameter = membership_funcs_params[i][j]
                a, b, c = parameter
                layer = FuzzificationLayer(a, b, c)
                fuzzification_layer.append(layer)

            self.fuzzification_layers.append(fuzzification_layer)

        self.rule_layer = RuleLayer()

        self.normalization_layer = NormalizationLayer()

        self.defuzzification_layers = torch.nn.ModuleList()
        for _ in range(input_features*num_membership_functions):
            defuzzification_layer = DefuzzificationLayer(input_features)
            self.defuzzification_layers.append(defuzzification_layer)

        self.aggregation_layer = AggregationLayer()

    def forward(self, x:List):

        # Generate Inputs
        inputs = []
        for i in range(len(x)):
            inputs.append(self.input_layer(x[i]))

        rule_outputs = []
        # Fuzzify Inputs and pass to Rule Layer
        for i in range(self.input_features):
            for m in range(self.num_membership_functions):
                for j in range(i+1, self.input_features):
                    for n in range(self.num_membership_functions):
                        layer_1 = self.fuzzification_layers[i][m]
                        layer_2 = self.fuzzification_layers[j][n]
                        output = self.rule_layer([layer_1(inputs[i]), layer_2(inputs[j])])
                        rule_outputs.append(output)

        # Normalize outputs of rule layers
        normalization_output = self.normalization_layer(rule_outputs).permute(1, 0)

        # Defuzzify
        defuzzification_outputs = []
        for idx, layer in enumerate(self.defuzzification_layers):
            output = layer(inputs, torch.unsqueeze(normalization_output[idx], 1))
            defuzzification_outputs.append(output)

        # Aggregation
        final_output = self.aggregation_layer(defuzzification_outputs)

        return final_output



