import sys
from math import sqrt
from torch import no_grad, zeros, empty, where, topk, long
from torch.nn.init import calculate_gain

class GnT(object):
    """
    Generate-and-Test algorithm for feed forward neural networks, based on maturity-threshold based replacement
    """
    def __init__(self, net, hidden_activation, opt,
                decay_rate=0.99, replacement_rate=1e-4, maturity_threshold=100, 
                util_type='adaptable_contribution'):
        
        super(GnT, self).__init__()

        self.net = net.layers
        self.num_hidden_layers = len(self.net)-1
        self.dev = next(self.net.parameters()).device
        self.opt = opt
        self.opt_type = 'sgd'

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.util = [zeros(self.net[i].fc.out_features, device=self.dev) for i in range(self.num_hidden_layers)]
        self.bias_corrected_util = [zeros(self.net[i].fc.out_features, device=self.dev) for i in range(self.num_hidden_layers)]
        self.ages = [zeros(self.net[i].fc.out_features, device=self.dev) for i in range(self.num_hidden_layers)]
        self.mean_feature_act = [zeros(self.net[i].fc.out_features, device=self.dev) for i in range(self.num_hidden_layers)]
        
        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]

        self.bounds = self.compute_bounds(hidden_activation=hidden_activation)

    def compute_bounds(self, hidden_activation):
        bounds = [calculate_gain(nonlinearity=hidden_activation) * sqrt(3 / self.net[i].fc.in_features) for i in range(self.num_hidden_layers)]
        bounds.append(1 * sqrt(3 / self.net[self.num_hidden_layers].fc.in_features))
        return bounds

    def update_utility(self, layer_idx=0, features=None, next_features=None):
        with no_grad():
            self.util[layer_idx] *= self.decay_rate
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            current_layer = self.net[layer_idx].fc
            next_layer = self.net[layer_idx + 1].fc

            output_wight_mag = next_layer.weight.data.abs().mean(dim=0)

            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_feature_act[layer_idx] +=  (1 - self.decay_rate) * features.mean(dim=0)
            input_wight_mag = current_layer.weight.data.abs().mean(dim=1)
 
            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

            if self.util_type == 'weight':
                new_util = output_wight_mag
            elif self.util_type == 'contribution':
                new_util = output_wight_mag * features.abs().mean(dim=0)
            elif self.util_type == 'adaptable_contribution':
                new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0) / input_wight_mag
            else:
                new_util = 0

            self.util[layer_idx] += (1 - self.decay_rate) * new_util

            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace = [empty(0, dtype=long).to(self.dev) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace
        
        for i in range(self.num_hidden_layers):
            self.ages[i] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_idx=i, features=features[i])
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
            self.accumulated_num_features_to_replace[i] += num_new_features_to_replace

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
            self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace

            if num_new_features_to_replace == 0: continue

            """
            Find features to replace in the current layer
            """
            new_features_to_replace = topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                                 num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            """
            Initialize utility for new features
            """
            self.util[i][new_features_to_replace] = 0
            self.mean_feature_act[i][new_features_to_replace] = 0.

            num_features_to_replace[i] = num_new_features_to_replace
            features_to_replace[i] = new_features_to_replace

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer = self.net[i].fc
                next_layer = self.net[i+1].fc
                
                current_layer.weight.data[features_to_replace[i], :] *= 0.0
                current_layer.weight.data[features_to_replace[i], :] += \
                    empty(num_features_to_replace[i], current_layer.in_features).uniform_(-self.bounds[i], self.bounds[i]).to(self.dev)

                current_layer.bias.data[features_to_replace[i]] *= 0
                """
                # Update bias to correct for the removed features and set the outgoing weights and ages to zero
                """
                next_layer.bias.data += (next_layer.weight.data[:, features_to_replace[i]] * \
                                                self.mean_feature_act[i][features_to_replace[i]] / \
                                                (1 - self.decay_rate ** self.ages[i][features_to_replace[i]])).sum(dim=1)
                next_layer.weight.data[:, features_to_replace[i]] = 0
                self.ages[i][features_to_replace[i]] = 0


    def gen_and_test(self, features):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace, num_features_to_replace)
