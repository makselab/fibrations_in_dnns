from torch import stack, unique
from torch.nn import Module, Linear, ReLU
from coloring import fibration_linear, opfibration_linear

class MLP(Module):
  def __init__(self, input_size, hidden_sizes, num_classes):
    super(MLP,self).__init__()
    self.fc1 = Linear(input_size, hidden_sizes[0])
    self.fc2 = Linear(hidden_sizes[0], hidden_sizes[1])
    self.fc3 = Linear(hidden_sizes[1], hidden_sizes[2])
    self.fc4 = Linear(hidden_sizes[2], num_classes)
    self.relu = ReLU()
    
    self.fibration_colors = None
    self.opfibration_colors = None
    self.covering_colors = None

  def forward(self,x):
    activations = []
    out = self.fc1(x)
    activations.append(out.detach())
    out = self.relu(out)
    out = self.fc2(out)
    activations.append(out.detach())
    out = self.relu(out)
    out = self.fc3(out)
    activations.append(out.detach())
    out = self.relu(out)
    out = self.fc4(out)

    return activations, out

  def fibration_coloring(self, threshold=0.8, bias=False):

      colors = []

      term_bias = self.fc1.bias.data if bias else None
      
      clusters = fibration_linear(weights=self.fc1.weight.data, 
                                  in_clusters=None, 
                                  threshold=threshold, 
                                  first_layer = True,
                                  bias = term_bias)

      colors.append(clusters)

      term_bias = self.fc2.bias.data if bias else None

      clusters = fibration_linear(weights=self.fc2.weight.data, 
                                  in_clusters=clusters, 
                                  threshold=threshold, 
                                  first_layer = False,
                                  bias = term_bias)

      colors.append(clusters)

      term_bias = self.fc3.bias.data if bias else None

      clusters = fibration_linear(weights=self.fc3.weight.data, 
                                  in_clusters=clusters, 
                                  threshold=threshold, 
                                  first_layer = False,
                                  bias = term_bias)

      colors.append(clusters)

      self.fibration_colors = colors

  def opfibration_coloring(self, threshold=0.5):
      colors = [None, None, None]

      clusters = opfibration_linear(weights=self.fc4.weight.data, 
                          out_clusters=None, 
                          threshold=threshold, 
                          last_layer = True)

      colors[2] = clusters

      clusters = opfibration_linear(weights=self.fc3.weight.data, 
                          out_clusters=clusters, 
                          threshold=threshold, 
                          last_layer = False)

      colors[1] = clusters

      clusters = opfibration_linear(weights=self.fc2.weight.data, 
                          out_clusters=clusters, 
                          threshold=threshold, 
                          last_layer = False)

      colors[0] = clusters

      self.opfibration_colors = colors

  def covering_coloring(self, fib_thr=0.8, op_thr = 0.5, bias=False):
      self.fibration_coloring(fib_thr, bias)
      self.opfibration_coloring(op_thr)

      colors = [None, None, None]

      for idx_l in range(3):
        matrix_colors = stack((self.fibration_colors[idx_l], self.opfibration_colors[idx_l]), dim=0)
        _, colors[idx_l] = unique(matrix_colors, dim=1, return_inverse=True)

      self.covering_colors = colors