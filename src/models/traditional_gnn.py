import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool
from typing import List, Tuple, Optional
import numpy as np


class TraditionalGNN(nn.Module):
    """Traditional (non-Bayesian) GNN for comparison with Bayesian variants"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 gnn_type: str = 'GCN',
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super(TraditionalGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.gnn_type = gnn_type
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if gnn_type == 'GCN':
                layer = GCNConv(hidden_dims[i], hidden_dims[i+1])
            elif gnn_type == 'GAT':
                layer = GATConv(hidden_dims[i], hidden_dims[i+1] // n_heads, 
                               heads=n_heads, dropout=dropout)
            elif gnn_type == 'GraphSAGE':
                layer = GraphSAGE(hidden_dims[i], hidden_dims[i+1])
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            self.gnn_layers.append(layer)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Temporal attention for time series
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1], 
            num_heads=4, 
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # GNN layers
        for i, layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'GAT':
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index, edge_attr)
            
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # Global pooling (if batch is provided)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        return self.output_layer(x)
    
    def predict_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor,
                           edge_attr: torch.Tensor = None, batch: torch.Tensor = None,
                           n_samples: int = 100) -> dict:
        """Monte Carlo Dropout for uncertainty estimation"""
        self.train()  # Keep dropout active
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, edge_index, edge_attr, batch)
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


class TemporalTraditionalGNN(nn.Module):
    """Traditional GNN with temporal modeling using LSTM/GRU"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 sequence_length: int = 30,
                 gnn_type: str = 'GCN',
                 rnn_type: str = 'LSTM',
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super(TemporalTraditionalGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.rnn_type = rnn_type
        
        # Traditional GNN for each time step
        self.gnn = TraditionalGNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],  # Output to RNN
            gnn_type=gnn_type,
            n_heads=n_heads,
            dropout=dropout
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
        
        # Final prediction layer
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        
        batch_size, seq_len, n_nodes, n_features = x_seq.shape
        
        # Process each time step through GNN
        gnn_outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t].view(-1, n_features)  # (batch_size * n_nodes, n_features)
            
            # Create batch index for pooling
            batch_idx = torch.arange(batch_size, device=x_t.device).repeat_interleave(n_nodes)
            
            # Forward through GNN
            gnn_out = self.gnn(x_t, edge_index, edge_attr, batch_idx)
            gnn_outputs.append(gnn_out)
        
        # Stack temporal outputs
        gnn_sequence = torch.stack(gnn_outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
        
        # Process through RNN
        rnn_out, _ = self.rnn(gnn_sequence)
        
        # Use last time step for prediction
        final_hidden = rnn_out[:, -1, :]
        
        # Final prediction
        return self.final_layer(final_hidden)


class EnsembleTraditionalGNN(nn.Module):
    """Ensemble of traditional GNNs for improved performance"""
    
    def __init__(self, n_models: int = 5, **model_kwargs):
        super(EnsembleTraditionalGNN, self).__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList([
            TraditionalGNN(**model_kwargs) for _ in range(n_models)
        ])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        
        outputs = []
        for model in self.models:
            pred = model(x, edge_index, edge_attr, batch)
            outputs.append(pred)
        
        # Ensemble prediction (average)
        predictions = torch.stack(outputs, dim=0)
        ensemble_mean = predictions.mean(dim=0)
        
        return ensemble_mean
    
    def predict_with_ensemble_uncertainty(self, x: torch.Tensor, edge_index: torch.Tensor,
                                        edge_attr: torch.Tensor = None, 
                                        batch: torch.Tensor = None) -> dict:
        """Predict with ensemble-based uncertainty estimation"""
        self.eval()
        
        with torch.no_grad():
            outputs = []
            for model in self.models:
                pred = model(x, edge_index, edge_attr, batch)
                outputs.append(pred.cpu().numpy())
        
        predictions = np.array(outputs)
        
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


class GraphTransformer(nn.Module):
    """Graph Transformer architecture for comparison"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super(GraphTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Position encoding for nodes
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.size(0)
        if seq_len <= self.pos_encoding.size(0):
            x = x + self.pos_encoding[:seq_len]
        
        # Reshape for transformer (add batch dimension if needed)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Create attention mask based on graph structure
        # For simplicity, we'll use full attention (no masking)
        # In practice, you'd create masks based on edge_index
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global pooling if batch is provided
        if batch is not None:
            # Group by batch and take mean
            x = global_mean_pool(x.squeeze(0), batch)
        else:
            # Take mean across all nodes
            x = x.mean(dim=1)
        
        # Output layer
        return self.output_layer(x)


class HybridGNN(nn.Module):
    """Hybrid model combining different GNN architectures"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 dropout: float = 0.1):
        super(HybridGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Different GNN branches
        self.gcn_branch = TraditionalGNN(
            input_dim, hidden_dims, hidden_dims[-1], 'GCN', dropout=dropout
        )
        self.gat_branch = TraditionalGNN(
            input_dim, hidden_dims, hidden_dims[-1], 'GAT', dropout=dropout
        )
        
        # Fusion layer
        self.fusion_layer = nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        
        # Forward through different branches
        gcn_out = self.gcn_branch(x, edge_index, edge_attr, batch)
        gat_out = self.gat_branch(x, edge_index, edge_attr, batch)
        
        # Concatenate and fuse
        combined = torch.cat([gcn_out, gat_out], dim=-1)
        fused = self.fusion_layer(combined)
        fused = F.relu(fused)
        
        # Final prediction
        return self.output_layer(fused)