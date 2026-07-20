from torch import stack, unique, arange, randperm, norm, where
from torch.nn import Module, Linear, ReLU, ModuleList
from symmetries.coloring import fibration_linear, opfibration_linear, covering
from symmetries.collapse import collapse_linear
from numpy import cumsum
from compression_methods import ablation_linear
import copy
import torch.nn.utils.prune as prune

class MLP(Module):
  def __init__(self, input_size, hidden_sizes=[500,500,500], num_classes=10):
    super(MLP,self).__init__()

    self.dims = [input_size] + hidden_sizes + [num_classes]
    self.num_layers = len(hidden_sizes)
    self.activation = ReLU()
    self.layers = ModuleList([Linear(self.dims[i],self.dims[i+1]) for i in range(self.num_layers + 1)])

    self.symmetries = {'fibration': None,
                    'opfibration':None,
                    'covering': None}

  def forward(self,x):
    activations = []

    for i in range(self.num_layers):
      x = self.layers[i](x)
      activations.append(x.detach())
      x = self.activation(x)

    out = self.layers[self.num_layers](x)

    return activations, out

  def fibration_coloring(self, clustering_method, distance_thrs):

      assert len(distance_thrs) == self.num_layers, f"distance_thr has {len(distance_thrs)} entries, expected {self.num_layers}"

      colors = []
      current_colors = arange(self.dims[0])

      for i in range(self.num_layers):
        current_colors = fibration_linear(weights=self.layers[i].weight.data, 
                                        bias = self.layers[i].bias.data,
                                        in_colors=current_colors, 
                                        clustering_method=clustering_method,
                                        distance_thr=distance_thrs[i].item())

        colors.append(current_colors)

      self.symmetries['fibration'] = colors

  def opfibration_coloring(self, clustering_method, distance_thrs):

      assert len(distance_thrs) == self.num_layers, f"distance_thr has {len(distance_thrs)} entries, expected {self.num_layers}"

      colors = [None for i in range(self.num_layers)]
      current_colors = arange(self.dims[-1])

      for i in range(self.num_layers, 0, -1):
        current_colors = opfibration_linear(weights=self.layers[i].weight.data,
                                      bias=self.layers[i].bias.data,
                                      out_colors=current_colors, 
                                      clustering_method=clustering_method,
                                      distance_thr=distance_thrs[i-1].item())

        colors[i-1] = current_colors

      self.symmetries['opfibration'] = colors

  def covering_coloring(self, clustering_method, fib_thrs, op_thrs):
      self.fibration_coloring(clustering_method, fib_thrs)
      self.opfibration_coloring(clustering_method, op_thrs)

      colors = []

      for idx_l in range(self.num_layers):
        covers = covering(self.symmetries['fibration'][idx_l],self.symmetries['opfibration'][idx_l])
        colors.append(covers) 

      self.symmetries['covering'] = colors


  def num_colors(self, symmetry='covering'):
      return [unique(colors_layer).shape[0] for colors_layer in self.symmetries[symmetry]]

  def compute_dWs_and_params(self, colors_cov):
      dWs         = {}
      W_colls     = {}
      n_in        = self.dims[0]
      total_params = 0

      self.layers[0].in_colors = arange(self.dims[0])
      for i in range(self.num_layers):
          self.layers[i+1].in_colors = colors_cov[i]
      for i in range(self.num_layers):
          self.layers[i].out_colors = colors_cov[i]
      self.layers[self.num_layers].out_colors = arange(self.dims[-1])

      for i, layer in enumerate(self.layers):
          coll_layer, dW, db = collapse_linear(layer)
          dWs[f'layers.{i}.weight']     = dW
          dWs[f'layers.{i}.bias']       = db
          W_colls[f'layers.{i}.weight'] = coll_layer.weight.data
          W_colls[f'layers.{i}.bias']   = coll_layer.bias.data

          n_out         = layer.out_colors.max().item() + 1
          total_params += n_out * (n_in + 1)
          n_in          = n_out

      return dWs, W_colls, total_params

  def collapse_version(self, symmetry='covering'):
      colors_cov = self.symmetries[symmetry]
      dWs, W_colls, _ = self.compute_dWs_and_params(colors_cov)

      collapsed_hidden_sizes = [self.layers[i].out_colors.unique().shape[0] for i in range(self.num_layers)]
      mlp_coll = MLP(input_size=self.dims[0], hidden_sizes=collapsed_hidden_sizes, num_classes=self.dims[-1])

      for i, coll_layer in enumerate(mlp_coll.layers):
          coll_layer.weight.data = W_colls[f'layers.{i}.weight']
          coll_layer.bias.data   = W_colls[f'layers.{i}.bias']

      return mlp_coll, dWs

  def ablation_version(self, num_nodes_total_ablation):

      list_nodes = randperm(sum(self.dims[1:-1]))[:num_nodes_total_ablation]
      accumulative_nodes = cumsum([0] + self.dims[1:-1]).tolist()

      L_abl = [[x.item()-accumulative_nodes[i] for x in list_nodes if accumulative_nodes[i] <= x < accumulative_nodes[i+1]] for i in range(self.num_layers)]

      ablation_hidden_sizes = [len(layer) for layer in L_abl]
      mlp_abl = MLP(input_size=self.dims[0], hidden_sizes=ablation_hidden_sizes, num_classes=self.dims[-1])

      auxs = [None]+ L_abl + [None]

      for i, layer in enumerate(self.layers):
        abl_layer = ablation_linear(layer, 
                                    nodes_ablation_in=auxs[i], 
                                    nodes_ablation_out=auxs[i+1])

        mlp_abl.layers[i].weight.data = abl_layer.weight.data
        mlp_abl.layers[i].bias.data = abl_layer.bias.data

      return mlp_abl

  def pruning_version(self, amount):
      net_pruned = copy.deepcopy(self)
      layer_pruned = [None for layer_idx in range(self.num_layers)]
      num_nodes_pruned = [None for layer_idx in range(self.num_layers)]

      for layer_idx in range(self.num_layers):
        prune.ln_structured(net_pruned.layers[layer_idx], name='weight', amount=amount, n=1, dim=0)
        prune.remove(net_pruned.layers[layer_idx], 'weight')

      for layer_idx in range(self.num_layers):
        fc_norms = norm(net_pruned.layers[layer_idx].weight, p=1, dim=1)
        layer_pruned[layer_idx] = where(fc_norms > 1e-7)[0]
        num_nodes_pruned[layer_idx] = len(layer_pruned[layer_idx])

      mlp_pruned = MLP(784, num_nodes_pruned, 10)

      auxs = [None]+ layer_pruned + [None]

      for i, layer in enumerate(self.layers):
        abl_layer = ablation_linear(layer, 
                                    nodes_ablation_in=auxs[i], 
                                    nodes_ablation_out=auxs[i+1])

        mlp_pruned.layers[i].weight.data = abl_layer.weight.data
        mlp_pruned.layers[i].bias.data = abl_layer.bias.data

      return mlp_pruned