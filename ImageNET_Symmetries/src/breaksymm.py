from torch import where, rand, topk, long, empty, zeros, no_grad, tensor, unique, cat, randperm
from torch.nn import Conv2d, Linear, Softmax
from torch.nn.init import calculate_gain

from utils import get_layer_bound

class BreakSymmetry(object):
    """
    Generate-and-Test algorithm for ConvNets, maturity threshold based tester, accumulates probability of replacement.
    """

    def __init__(self, net, hidden_activation, opt, num_last_filter_outputs=4,
                replacement_rate=1e-4, init='kaiming', maturity_threshold=100):

        super(BreakSymmetry, self).__init__()

        # Model properties
        self.net = net
        self.num_hidden_layers = int(len(self.net.layers)/2)

        self.dev = next(self.net.layers.parameters()).device
        self.opt = opt
        self.opt_type = 'sgd'

        # Algorithm HyperParameters
        self.replacement_rate = replacement_rate
        self.num_last_filter_outputs = num_last_filter_outputs
        self.maturity_threshold = maturity_threshold

        # Initialization results/arrays = 32,64,128,128,128
        self.ages = []
        self.num_new_features_to_replace = []

        for i in range(self.num_hidden_layers):
            layer = self.net.layers[i * 2]
            dim = layer.out_channels if isinstance(layer, Conv2d) else layer.out_features 
            self.ages.append(zeros(dim, device=self.dev))
            self.num_new_features_to_replace.append(self.replacement_rate * dim)

        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        gain = calculate_gain(nonlinearity=hidden_activation)    
        bounds = [get_layer_bound(layer=self.net.layers[i * 2], init=init, gain=gain) for i in range(self.num_hidden_layers)]
        bounds.append(get_layer_bound(layer=self.net.layers[-1], init=init, gain=1))
        return bounds

    def test_features(self):
        """
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """

        # colors = 32,64,512,128,128

        features_to_replace_input_indices = [None for _ in range(self.num_hidden_layers)]
        features_to_replace_output_indices = [None for _ in range(self.num_hidden_layers)]
        base_nodes_to_modify_indices = [None for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]

        for i in range(self.num_hidden_layers):
            self.ages[i] += 1

            # Find the no. of features to replace
            eligible_feature_indices = where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0: continue
            self.accumulated_num_features_to_replace[i] += self.num_new_features_to_replace[i]

            # Case when the number of features to be replaced is between 0 and 1.
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
            self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace
            if num_new_features_to_replace == 0: continue

            # Find features to replace in the current layer
            if self.net.covering_colors == None: 
                self.net.covering_coloring()

            colors = self.net.covering_colors
            colors_layer = colors[i].to(self.dev)
            idx_colors = unique(colors_layer,dim=0)

            nontrivial_fibers = {}

            for cc in idx_colors:
                mask = (colors_layer == cc)
                mask = mask.all(dim=1) 
                fiber = where(mask)[0]

                if len(fiber) > 1:
                    nontrivial_fibers[fiber[0]] = fiber[1:]

            if len(nontrivial_fibers) == 0: continue

            lift_features = cat(list(nontrivial_fibers.values()), dim=0)
            intersection = lift_features[where(lift_features.unsqueeze(1) == eligible_feature_indices)[0]]
            new_features_to_replace = intersection[randperm(len(intersection))[:num_new_features_to_replace]]

            inverse_dict = {}

            for base_nodes, lift_nodes in nontrivial_fibers.items():
                for nn in lift_nodes:
                    inverse_dict[nn.item()] = base_nodes

            base_nodes_to_modify = tensor([inverse_dict[nn.item()] for nn in new_features_to_replace])

            # Initialize utility for new features
            num_features_to_replace[i] = num_new_features_to_replace
            features_to_replace_input_indices[i] = new_features_to_replace
            features_to_replace_output_indices[i] = new_features_to_replace
            base_nodes_to_modify_indices[i] = base_nodes_to_modify

            # Case of Pooling
            if i==2:
                features_to_replace_output_indices[i] = \
                    (new_features_to_replace*self.num_last_filter_outputs).repeat_interleave(self.num_last_filter_outputs) + \
                    tensor([i for i in range(self.num_last_filter_outputs)], device=self.dev).repeat(new_features_to_replace.size()[0])

                base_nodes_to_modify_indices[i] = \
                    new_features_to_replace.repeat_interleave(self.num_last_filter_outputs)

        return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace, base_nodes_to_modify_indices

    def gen_new_features(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace, base_nodes_to_modify_indices):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        
        with no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer = self.net.layers[i * 2]
                next_layer = self.net.layers[i * 2 + 2]

                current_layer.bias.data[features_to_replace_input_indices[i]] *= 0.0
                current_layer.weight.data[features_to_replace_input_indices[i], :] *= 0.0

                if isinstance(current_layer, Linear):
                    current_layer.weight.data[features_to_replace_input_indices[i], :] -= - \
                        empty(num_features_to_replace[i], current_layer.in_features, device =self.dev). \
                            uniform_(-self.bounds[i], self.bounds[i])
                elif isinstance(current_layer, Conv2d):
                    current_layer.weight.data[features_to_replace_input_indices[i], :] -= - \
                        empty([num_features_to_replace[i]] + list(current_layer.weight.shape[1:]), device=self.dev). \
                            uniform_(-self.bounds[i], self.bounds[i])


                # Set the outgoing weights of base nodes.
                if len(base_nodes_to_modify_indices[i]) != len(features_to_replace_output_indices[i]):
                    print('aqui')
                    exit()

                for id_base, id_lif in zip(base_nodes_to_modify_indices[i], features_to_replace_output_indices[i]):
                    next_layer.weight.data[:, id_base] += next_layer.weight.data[:, id_lif]
                
                # Set the outgoing weights and ages to zero
                next_layer.weight.data[:, features_to_replace_output_indices[i]] = 0
                self.ages[i][features_to_replace_input_indices[i]] = 0

    def gen_and_test(self):
        features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace, base_nodes_to_modify_indices = self.test_features()
        self.gen_new_features(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace, base_nodes_to_modify_indices)