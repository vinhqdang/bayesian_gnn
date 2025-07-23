import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Optional, List, Tuple
import numpy as np

from .bayesian_layers import (
    BayesianLinear, BayesianGCNConv, BayesianGATConv, 
    VariationalDropout, UncertaintyQuantification
)

class BayesianGNN(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 gnn_type: str = 'GCN',
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 prior_std: float = 0.1,
                 use_uncertainty: bool = True):
        super(BayesianGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.gnn_type = gnn_type
        self.n_heads = n_heads
        self.dropout = dropout
        self.prior_std = prior_std
        self.use_uncertainty = use_uncertainty
        
        # Input projection
        self.input_proj = BayesianLinear(input_dim, hidden_dims[0], prior_std)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if gnn_type == 'GCN':
                layer = BayesianGCNConv(hidden_dims[i], hidden_dims[i+1], prior_std)
            elif gnn_type == 'GAT':
                layer = BayesianGATConv(hidden_dims[i], hidden_dims[i+1] // n_heads, 
                                     heads=n_heads, dropout=dropout, prior_std=prior_std)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            self.gnn_layers.append(layer)
        
        # Variational dropout
        self.var_dropout = VariationalDropout(dropout)
        
        # Output layer
        if use_uncertainty:
            self.output_layer = UncertaintyQuantification(hidden_dims[-1], output_dim)
        else:
            self.output_layer = BayesianLinear(hidden_dims[-1], output_dim, prior_std)
        
        # Temporal attention for time series
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1], 
            num_heads=4, 
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None, batch: torch.Tensor = None,
                sample: bool = True) -> torch.Tensor:
        
        # Input projection
        x = self.input_proj(x, sample=sample)
        x = F.relu(x)
        x = self.var_dropout(x, training=self.training)
        
        # GNN layers
        for i, layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'GAT':
                x = layer(x, edge_index, edge_attr, sample=sample)
            else:
                x = layer(x, edge_index, edge_attr, sample=sample)
            
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = self.var_dropout(x, training=self.training)
        
        # Global pooling (if batch is provided)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        if self.use_uncertainty:
            return self.output_layer(x, sample=sample)
        else:
            return self.output_layer(x, sample=sample)
    
    def kl_divergence(self) -> torch.Tensor:
        kl_div = self.input_proj.kl_divergence()
        
        for layer in self.gnn_layers:
            kl_div += layer.kl_divergence()
        
        if hasattr(self.output_layer, 'kl_divergence'):
            kl_div += self.output_layer.kl_divergence()
            
        return kl_div
    
    def predict_with_uncertainty(self, x: torch.Tensor, edge_index: torch.Tensor,
                               edge_attr: torch.Tensor = None, batch: torch.Tensor = None,
                               n_samples: int = 100) -> dict:
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                if self.use_uncertainty:
                    mean, var = self.forward(x, edge_index, edge_attr, batch, sample=True)
                    # Sample from predictive distribution
                    pred = torch.normal(mean, torch.sqrt(var))
                    predictions.append(pred.cpu().numpy())
                else:
                    pred = self.forward(x, edge_index, edge_attr, batch, sample=True)
                    predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Compute statistics
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        pred_lower = np.percentile(predictions, 2.5, axis=0)
        pred_upper = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean': pred_mean,
            'std': pred_std,
            'lower_ci': pred_lower,
            'upper_ci': pred_upper,
            'samples': predictions
        }

class TemporalBayesianGNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 sequence_length: int = 30,
                 gnn_type: str = 'GCN',
                 rnn_type: str = 'LSTM',
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 prior_std: float = 0.1):
        super(TemporalBayesianGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.rnn_type = rnn_type
        
        # Bayesian GNN for each time step
        self.bgnn = BayesianGNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],  # Output to RNN
            gnn_type=gnn_type,
            n_heads=n_heads,
            dropout=dropout,
            prior_std=prior_std,
            use_uncertainty=False
        )
        
        # Temporal modeling
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=hidden_dims[-1],
                hidden_size=hidden_dims[-1],
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=hidden_dims[-1],
                hidden_size=hidden_dims[-1],
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
        
        # Final prediction layer with uncertainty
        self.final_layer = UncertaintyQuantification(hidden_dims[-1], output_dim)
        
    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None, sample: bool = True) -> tuple:
        
        batch_size, seq_len, n_nodes, n_features = x_seq.shape
        
        # Process each time step through GNN
        gnn_outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t].view(-1, n_features)  # (batch_size * n_nodes, n_features)
            
            # Create batch index for pooling
            batch_idx = torch.arange(batch_size, device=x_t.device).repeat_interleave(n_nodes)
            
            # Forward through GNN
            gnn_out = self.bgnn(x_t, edge_index, edge_attr, batch_idx, sample=sample)
            gnn_outputs.append(gnn_out)
        
        # Stack temporal outputs
        gnn_sequence = torch.stack(gnn_outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
        
        # Process through RNN
        rnn_out, _ = self.rnn(gnn_sequence)
        
        # Use last time step for prediction
        final_hidden = rnn_out[:, -1, :]
        
        # Final prediction with uncertainty
        return self.final_layer(final_hidden, sample=sample)
    
    def kl_divergence(self) -> torch.Tensor:
        return self.bgnn.kl_divergence() + self.final_layer.kl_divergence()

class EnsembleBayesianGNN(nn.Module):
    def __init__(self, n_models: int = 5, **model_kwargs):
        super(EnsembleBayesianGNN, self).__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList([
            BayesianGNN(**model_kwargs) for _ in range(n_models)
        ])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None, batch: torch.Tensor = None,
                sample: bool = True) -> torch.Tensor:
        
        outputs = []
        for model in self.models:
            if model.use_uncertainty:
                mean, var = model(x, edge_index, edge_attr, batch, sample=sample)
                outputs.append((mean, var))
            else:
                pred = model(x, edge_index, edge_attr, batch, sample=sample)
                outputs.append(pred)
        
        if self.models[0].use_uncertainty:
            means = torch.stack([out[0] for out in outputs], dim=0)
            vars = torch.stack([out[1] for out in outputs], dim=0)
            
            # Ensemble mean and variance
            ensemble_mean = means.mean(dim=0)
            ensemble_var = (vars + means**2).mean(dim=0) - ensemble_mean**2
            
            return ensemble_mean, ensemble_var
        else:
            predictions = torch.stack(outputs, dim=0)
            ensemble_mean = predictions.mean(dim=0)
            return ensemble_mean
    
    def kl_divergence(self) -> torch.Tensor:
        return sum(model.kl_divergence() for model in self.models) / self.n_models