import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch.distributions import Normal, kl_divergence
import math

class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float = 0.1):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_var = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_var = nn.Parameter(torch.randn(out_features) * 0.1 - 5)
        
        # Prior distributions
        self.weight_prior = Normal(0, prior_std)
        self.bias_prior = Normal(0, prior_std)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            # Sample weights and biases
            weight_std = torch.exp(0.5 * self.weight_log_var)
            bias_std = torch.exp(0.5 * self.bias_log_var)
            
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean values
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        # KL divergence between posterior and prior
        weight_posterior = Normal(self.weight_mu, torch.exp(0.5 * self.weight_log_var))
        bias_posterior = Normal(self.bias_mu, torch.exp(0.5 * self.bias_log_var))
        
        weight_kl = kl_divergence(weight_posterior, self.weight_prior).sum()
        bias_kl = kl_divergence(bias_posterior, self.bias_prior).sum()
        
        return weight_kl + bias_kl

class BayesianGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, prior_std: float = 0.1):
        super(BayesianGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prior_std = prior_std
        
        # GCN layer with Bayesian weights
        self.gcn = GCNConv(in_channels, out_channels)
        
        # Replace linear transformation with Bayesian version
        self.bayesian_lin = BayesianLinear(in_channels, out_channels, prior_std)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor = None, sample: bool = True) -> torch.Tensor:
        # Apply Bayesian linear transformation
        x = self.bayesian_lin(x, sample=sample)
        
        # Apply graph convolution (message passing)
        x = self.gcn.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        return self.bayesian_lin.kl_divergence()

class BayesianGATConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 dropout: float = 0.0, prior_std: float = 0.1):
        super(BayesianGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.prior_std = prior_std
        
        # Bayesian linear transformations for queries, keys, values
        self.lin_q = BayesianLinear(in_channels, out_channels * heads, prior_std)
        self.lin_k = BayesianLinear(in_channels, out_channels * heads, prior_std)
        self.lin_v = BayesianLinear(in_channels, out_channels * heads, prior_std)
        
        # Attention mechanism
        self.att = nn.Parameter(torch.randn(1, heads, 2 * out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None, sample: bool = True) -> torch.Tensor:
        H, C = self.heads, self.out_channels
        
        # Linear transformations
        q = self.lin_q(x, sample=sample).view(-1, H, C)
        k = self.lin_k(x, sample=sample).view(-1, H, C)  
        v = self.lin_v(x, sample=sample).view(-1, H, C)
        
        # Compute attention
        row, col = edge_index
        alpha = (torch.cat([q[row], k[col]], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Softmax normalization
        alpha = torch.softmax(alpha, dim=1)
        
        # Apply attention to values
        out = v[col] * alpha.unsqueeze(-1)
        
        # Aggregate messages
        out = torch.zeros(x.size(0), H, C, device=x.device, dtype=x.dtype)
        out = out.index_add_(0, row, out)
        
        return out.view(-1, H * C)
    
    def kl_divergence(self) -> torch.Tensor:
        return (self.lin_q.kl_divergence() + 
                self.lin_k.kl_divergence() + 
                self.lin_v.kl_divergence())

class VariationalDropout(nn.Module):
    def __init__(self, dropout_rate: float = 0.1):
        super(VariationalDropout, self).__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return x
            
        # Variational dropout with learned parameters
        log_alpha = torch.log(torch.tensor(self.dropout_rate / (1 - self.dropout_rate)))
        
        if self.training:
            # Sample from log-normal distribution
            eps = torch.randn_like(x)
            concrete = torch.sigmoid((torch.log(eps + 1e-8) - torch.log(1 - eps + 1e-8) + log_alpha) / 0.1)
            return x * concrete
        else:
            return x * (1 - self.dropout_rate)

class UncertaintyQuantification(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(UncertaintyQuantification, self).__init__()
        self.mean_head = BayesianLinear(input_dim, output_dim)
        self.var_head = BayesianLinear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple:
        mean = self.mean_head(x, sample=sample)
        log_var = self.var_head(x, sample=sample)
        var = F.softplus(log_var) + 1e-6  # Ensure positive variance
        
        return mean, var
    
    def kl_divergence(self) -> torch.Tensor:
        return self.mean_head.kl_divergence() + self.var_head.kl_divergence()
    
    def sample_prediction(self, x: torch.Tensor, n_samples: int = 100) -> tuple:
        means, vars = [], []
        
        for _ in range(n_samples):
            mean, var = self.forward(x, sample=True)
            means.append(mean)
            vars.append(var)
            
        means = torch.stack(means, dim=0)
        vars = torch.stack(vars, dim=0)
        
        # Compute predictive mean and uncertainty
        pred_mean = means.mean(dim=0)
        # Total uncertainty = epistemic + aleatoric
        epistemic_uncertainty = means.var(dim=0)
        aleatoric_uncertainty = vars.mean(dim=0)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return pred_mean, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty