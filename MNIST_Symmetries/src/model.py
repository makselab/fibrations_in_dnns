import torch
import numpy as np
from torch import unique, arange, randperm, norm, where, multinomial, no_grad, cat, quantile
from torch.nn import Module, Linear, ReLU, ModuleList
from symmetries.coloring import fibration_linear, opfibration_linear, covering
from symmetries.collapse import collapse_linear
from numpy import cumsum
from compression_methods import ablation_linear, KFEBottleneck
from symmetries.loss_coloring import loss_coloring_linear
from symmetries.loss_collapse import collapse_linear as loss_collapse_linear
import copy
import torch.nn.utils.prune as prune
from torch.linalg import eigh


class MLP(Module):
  def __init__(self, input_size, hidden_sizes=[500,500,500], num_classes=10):
    super(MLP,self).__init__()

    self.dims = [input_size] + hidden_sizes + [num_classes]
    self.num_layers = len(hidden_sizes)
    self.activation = ReLU()
    self.layers = ModuleList([Linear(self.dims[i],self.dims[i+1]) for i in range(self.num_layers + 1)])

    self.symmetries  = {'fibration': None,
                     'opfibration':None,
                     'covering': None,
                     'loss': None}
    self._loss_S     = None

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


  def loss_coloring(self, dataloader, criterion, distance_threshold,
                    clustering_method={'name': 'agg_clustering', 'cfg': {'linkage': 'average'}}):

      device  = next(self.parameters()).device
      A_sum   = [None] * self.num_layers
      S_sum   = [None] * self.num_layers
      N_total = 0

      for x, y in dataloader:
          x = x.view(x.shape[0], -1).to(device)
          y = y.to(device)
          N = x.shape[0]
          N_total += N

          self.zero_grad()
          activations, preacts = [], []
          inp = x

          for layer_idx in range(self.num_layers):
              activations.append(inp)
              z = self.layers[layer_idx](inp)
              z.retain_grad()
              preacts.append(z)
              inp = self.activation(z)

          out  = self.layers[self.num_layers](inp)
          loss = criterion(out, y)
          loss.backward()

          for layer_idx in range(self.num_layers):
              a  = activations[layer_idx].detach()
              gs = preacts[layer_idx].grad.detach()

              ones  = torch.ones(N, 1, device=device)
              a_aug = torch.cat([a, ones], dim=1)

              A_batch = a_aug.t() @ a_aug   # sum over batch
              S_batch = gs.t()   @ gs

              if A_sum[layer_idx] is None:
                  A_sum[layer_idx] = A_batch
                  S_sum[layer_idx] = S_batch
              else:
                  A_sum[layer_idx] += A_batch
                  S_sum[layer_idx] += S_batch

      colors     = []
      S_matrices = []
      for layer_idx in range(self.num_layers):
          A = A_sum[layer_idx] / N_total
          S = S_sum[layer_idx] / N_total

          W = self.layers[layer_idx].weight.data
          b = self.layers[layer_idx].bias.data

          labels = loss_coloring_linear(W, b, S, A, clustering_method, distance_threshold)
          colors.append(labels)
          S_matrices.append(S.cpu().numpy())

      self.symmetries['loss'] = colors
      self._loss_S            = S_matrices

  def collapse_loss_version(self):
      colors          = self.symmetries['loss']
      collapsed_sizes = [unique(c).shape[0] for c in colors]
      mlp = MLP(input_size=self.dims[0], hidden_sizes=collapsed_sizes, num_classes=self.dims[-1])

      # Assign in_colors / out_colors to each layer
      # Layer 0: input has no colors (identity), output colored by colors[0]
      # Layer l: input colored by colors[l-1], output colored by colors[l]
      # Output layer (num_layers): input colored by colors[-1], output has no colors
      self.layers[0].in_colors  = arange(self.dims[0])
      self.layers[0].out_colors = colors[0]
      for l in range(1, self.num_layers):
          self.layers[l].in_colors  = colors[l - 1]
          self.layers[l].out_colors = colors[l]
      self.layers[self.num_layers].in_colors  = colors[-1]
      self.layers[self.num_layers].out_colors = arange(self.dims[-1])

      # Collapse hidden layers
      for l in range(self.num_layers):
          coll_layer, _, _ = loss_collapse_linear(self.layers[l], self._loss_S[l])
          mlp.layers[l].weight.data = coll_layer.weight.data
          mlp.layers[l].bias.data   = coll_layer.bias.data

      # Collapse output layer: only input (no output collapse)
      coll_out, _, _ = loss_collapse_linear(self.layers[self.num_layers], self._loss_S[-1], collapse_in=True, collapse_out=False)
      mlp.layers[self.num_layers].weight.data = coll_out.weight.data
      mlp.layers[self.num_layers].bias.data   = coll_out.bias.data

      return mlp

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

      mlp_pruned = MLP(input_size=self.dims[0], hidden_sizes=num_nodes_pruned, num_classes=self.dims[-1])

      auxs = [None]+ layer_pruned + [None]

      for i, layer in enumerate(self.layers):
        abl_layer = ablation_linear(layer, 
                                    nodes_ablation_in=auxs[i], 
                                    nodes_ablation_out=auxs[i+1])

        mlp_pruned.layers[i].weight.data = abl_layer.weight.data
        mlp_pruned.layers[i].bias.data = abl_layer.bias.data

      return mlp_pruned

  def pfp_version(self, x, amount):
      # Provable Filter Pruning (Liebenwein et al., ICLR 2020)
      # X: batch of input points used to estimate sensitivities (the paper's S)
      # amount: fraction of filters to prune per layer (same convention as
      #         pruning_version's `amount`)
 
      with no_grad():
        activations, _ = self.forward(x)
 
      layer_pfp = [None for layer_idx in range(self.num_layers)]
      reweight_factors = [None for layer_idx in range(self.num_layers)]
 
      for layer_idx in range(self.num_layers):
        a      = self.activation(activations[layer_idx])          # a^l(x), (N, eta_l)
        W_next = self.layers[layer_idx + 1].weight.data            # (eta_{l+1}, eta_l)
 
        numerator   = W_next.unsqueeze(0) * a.unsqueeze(1)         # w_ij a_j(x), (N, eta_{l+1}, eta_l)
        denominator = a @ W_next.t()                               # sum_k w_ik a_k(x), (N, eta_{l+1})
        sensitivities_per_x = numerator / denominator.unsqueeze(-1).clamp_min(1e-12)
 
        s = sensitivities_per_x.abs().amax(dim=(0, 1))              # s_j^l, (eta_l,)
        p = s / s.sum()
 
        num_nodes = a.shape[1]
        m = max(1, round((1 - amount) * num_nodes))
 
        samples = multinomial(p, m, replacement=True)
        kept_nodes, counts = unique(samples, return_counts=True)
 
        layer_pfp[layer_idx]        = kept_nodes
        reweight_factors[layer_idx] = counts.float() / (m * p[kept_nodes])
 
      num_nodes_pfp = [len(nodes) for nodes in layer_pfp]
      mlp_pfp = MLP(input_size=self.dims[0], hidden_sizes=num_nodes_pfp, num_classes=self.dims[-1])
 
      auxs = [None] + layer_pfp + [None]
 
      for i, layer in enumerate(self.layers):
        working_layer = copy.deepcopy(layer)
 
        if i > 0:
          # reweight incoming columns for the filters sampled in layer i-1
          working_layer.weight.data[:, layer_pfp[i - 1]] *= reweight_factors[i - 1]
 
        abl_layer = ablation_linear(working_layer,
                                    nodes_ablation_in=auxs[i],
                                    nodes_ablation_out=auxs[i + 1])
 
        mlp_pfp.layers[i].weight.data = abl_layer.weight.data
        mlp_pfp.layers[i].bias.data = abl_layer.bias.data
 
      return mlp_pfp

  def eigendamage_version(self, x, y, amount, criterion):
      # EigenDamage (Wang et al., ICML 2019).
 
      net_ed = copy.deepcopy(self)

      # ---------------------------------------------------------------
 
      # manual forward pass (self.forward detaches activations, which would
      # block the backward pass needed to get grad_s below)
      self.zero_grad()
      activations = []
      preacts     = []
      N = x.shape[0]

      for layer_idx in range(self.num_layers):
        activations.append(x)
        z = self.layers[layer_idx](x)
        z.retain_grad()
        preacts.append(z)
        x = self.activation(z)
 
      out  = self.layers[self.num_layers](x)
      loss = criterion(out, y)
      loss.backward()

      # --------------------------------------------------------------- 
 
      for layer_idx in range(self.num_layers):
        a  = activations[layer_idx].detach()          # (N, n_in)
        gs = preacts[layer_idx].grad.detach()          # (N, n_out)
 
        A = (a.t() @ a) / N                            # eq. (2): E[a a^T]
        S = (gs.t() @ gs) / N                          # eq. (2): E[grad_s grad_s^T]
 
        Lambda_A, Q_A = eigh(A)
        Lambda_S, Q_S = eigh(S)
 
        W  = self.layers[layer_idx].weight.data.t()    # (n_in, n_out), paper convention
        Wp = Q_A.t() @ W @ Q_S                          # rotated weight, eq. (15)
 
        Theta = (Wp ** 2) * Lambda_A.unsqueeze(1) * Lambda_S.unsqueeze(0)  # Alg. 2, line 4
 
        row_importance = Theta.sum(dim=1)              # eigen-channels of Q_A (input side)
        col_importance = Theta.sum(dim=0)              # eigen-channels of Q_S (output side)
        tau = quantile(cat([row_importance, col_importance]), amount)
 
        kept_in  = where(row_importance > tau)[0]
        kept_out = where(col_importance > tau)[0]
 
        if len(kept_in) == 0:
          kept_in = row_importance.argsort(descending=True)[:1]
        if len(kept_out) == 0:
          kept_out = col_importance.argsort(descending=True)[:1]
 
        Q_in    = Q_A[:, kept_in].contiguous()          # (n_in, r_in)
        Q_out   = Q_S[:, kept_out].contiguous()         # (n_out, r_out)
        Wp_kept = Wp[kept_in][:, kept_out].contiguous() # (r_in, r_out)
        bias    = self.layers[layer_idx].bias.data.clone()
 
        net_ed.layers[layer_idx] = KFEBottleneck(Q_in, Wp_kept, Q_out, bias)
 
      return net_ed


