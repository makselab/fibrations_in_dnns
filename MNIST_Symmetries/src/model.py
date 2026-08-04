import torch
import numpy as np
from torch import unique, arange, randperm, norm, where, multinomial, no_grad, cat, quantile
from torch.nn import Module, Linear, ReLU, ModuleList
from symmetries.coloring import fibration_linear, opfibration_linear, covering
from symmetries.collapse import collapse_linear
from numpy import cumsum
from compression_methods import ablation_linear, KFEBottleneck
from symmetries.loss_coloring import cluster_multilayer_shared_budget
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


  def loss_coloring(self, x, y, criterion, F_max):

      # Manual forward pass to retain gradients on pre-activations
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

      layers_data = []
      S_matrices  = []
      for layer_idx in range(self.num_layers):
          a  = activations[layer_idx].detach()                   # (N, n_in)
          gs = preacts[layer_idx].grad.detach()                  # (N, n_out)

          ones  = torch.ones(N, 1, device=a.device)
          a_aug = torch.cat([a, ones], dim=1)                    # (N, n_in+1)
          A = ((a_aug.t() @ a_aug) / N).cpu().numpy()           # (n_in+1, n_in+1)
          S = ((gs.t()  @ gs)  / N).cpu().numpy()               # (n_out, n_out)

          W = self.layers[layer_idx].weight.data.cpu().numpy()   # (n_out, n_in)
          b = self.layers[layer_idx].bias.data.cpu().numpy()[:, None]
          w = np.hstack([W, b])                                  # (n_out, n_in+1)

          layers_data.append({"w": w, "S": S, "A": A})
          S_matrices.append(S)

      results, _ = cluster_multilayer_shared_budget(layers_data, F_max, verbose=False)

      colors = []
      for layer_idx, (clusters, _) in enumerate(results):
          n_out = self.dims[layer_idx + 1]
          color_tensor = torch.zeros(n_out, dtype=torch.long)
          for color_idx, cluster_set in enumerate(clusters):
              for node_idx in cluster_set:
                  color_tensor[node_idx] = color_idx
          colors.append(color_tensor)

      self.symmetries['loss'] = colors
      self._loss_S            = S_matrices

  def collapse_loss_version(self):
      collapsed_sizes = [unique(c).shape[0] for c in self.symmetries['loss']]
      mlp = MLP(input_size=self.dims[0], hidden_sizes=collapsed_sizes, num_classes=self.dims[-1])

      def row_weights(S, c_idx):
          """r_j = sum_{l in c} S[j,l] for each j in c. Fallback: uniform if all zeros."""
          r = S[np.ix_(c_idx, c_idx)].sum(axis=1)
          M = r.sum()
          if M < 1e-15:
              r = np.ones(len(c_idx))
              M = float(len(c_idx))
          return r, M

      # Hidden layers
      for l in range(self.num_layers):
          S_l          = self._loss_S[l]
          W_l          = self.layers[l].weight.data.cpu().numpy()   # (n_out, n_in)
          b_l          = self.layers[l].bias.data.cpu().numpy()     # (n_out,)
          curr_colors  = self.symmetries['loss'][l].numpy()
          prev_colors  = self.symmetries['loss'][l - 1].numpy() if l > 0 else None

          K_l   = collapsed_sizes[l]
          K_prev = mlp.dims[l]
          W_new = np.zeros((K_l, K_prev))
          b_new = np.zeros(K_l)

          for c in range(K_l):
              c_idx    = np.where(curr_colors == c)[0]
              r, M_c   = row_weights(S_l, c_idx)

              # Weighted average of weight rows in original input space
              w_avg = (r @ W_l[c_idx, :]) / M_c   # (dims[l],)
              b_new[c] = (r @ b_l[c_idx]) / M_c

              if l == 0:
                  W_new[c] = w_avg
              else:
                  # Project w_avg (dims[l]) → K_{l-1}: sum components per previous cluster
                  for i, val in enumerate(w_avg):
                      W_new[c, prev_colors[i]] += val

          mlp.layers[l].weight.data = torch.tensor(W_new, dtype=torch.float32)
          mlp.layers[l].bias.data   = torch.tensor(b_new, dtype=torch.float32)

      # Output layer: collapse columns with the same weighted average
      S_last      = self._loss_S[-1]
      W_out       = self.layers[self.num_layers].weight.data.cpu().numpy()  # (dims[-1], n_last)
      last_colors = self.symmetries['loss'][-1].numpy()
      K_last      = collapsed_sizes[-1]
      W_out_new   = np.zeros((self.dims[-1], K_last))

      for c in range(K_last):
          c_idx  = np.where(last_colors == c)[0]
          r, M_c = row_weights(S_last, c_idx)
          W_out_new[:, c] = (W_out[:, c_idx] @ r) / M_c

      mlp.layers[self.num_layers].weight.data = torch.tensor(W_out_new, dtype=torch.float32)
      mlp.layers[self.num_layers].bias.data   = self.layers[self.num_layers].bias.data.cpu().clone()

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


