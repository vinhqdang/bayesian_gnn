import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class StockGraphBuilder:
    def __init__(self, correlation_threshold: float = 0.3, top_k_edges: int = None):
        self.correlation_threshold = correlation_threshold
        self.top_k_edges = top_k_edges
        
    def calculate_correlation_matrix(self, features: np.ndarray, method: str = 'pearson') -> np.ndarray:
        n_stocks = features.shape[0]
        correlation_matrix = np.zeros((n_stocks, n_stocks))
        
        for i in range(n_stocks):
            for j in range(n_stocks):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    if method == 'pearson':
                        # Calculate correlation on returns
                        returns_i = features[i][:, 0]  # Returns column
                        returns_j = features[j][:, 0]
                        
                        # Remove NaN values
                        valid_indices = ~(np.isnan(returns_i) | np.isnan(returns_j))
                        if np.sum(valid_indices) > 10:  # Minimum data points
                            corr, _ = pearsonr(returns_i[valid_indices], returns_j[valid_indices])
                            correlation_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0
                        else:
                            correlation_matrix[i, j] = 0
                    
                    elif method == 'cosine':
                        # Calculate cosine similarity on feature vectors
                        feat_i = features[i].mean(axis=0).reshape(1, -1)
                        feat_j = features[j].mean(axis=0).reshape(1, -1)
                        similarity = cosine_similarity(feat_i, feat_j)[0, 0]
                        correlation_matrix[i, j] = abs(similarity)
        
        return correlation_matrix
    
    def build_adjacency_matrix(self, correlation_matrix: np.ndarray) -> np.ndarray:
        n_stocks = correlation_matrix.shape[0]
        adjacency_matrix = np.zeros((n_stocks, n_stocks))
        
        if self.top_k_edges is not None:
            # Keep top-k strongest correlations for each node
            for i in range(n_stocks):
                # Get indices of top-k correlations (excluding self-connection)
                correlations = correlation_matrix[i].copy()
                correlations[i] = 0  # Remove self-connection
                top_k_indices = np.argsort(correlations)[-self.top_k_edges:]
                
                for j in top_k_indices:
                    if correlations[j] > 0:  # Only positive correlations
                        adjacency_matrix[i, j] = correlations[j]
                        adjacency_matrix[j, i] = correlations[j]  # Make symmetric
        else:
            # Use correlation threshold
            adjacency_matrix = (correlation_matrix > self.correlation_threshold).astype(float)
            adjacency_matrix = adjacency_matrix * correlation_matrix  # Weight by correlation
            
        # Remove self-loops
        np.fill_diagonal(adjacency_matrix, 0)
        
        return adjacency_matrix
    
    def create_edge_index_and_weights(self, adjacency_matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert adjacency matrix to edge list format
        edge_indices = np.where(adjacency_matrix > 0)
        edge_index = torch.tensor(np.array([edge_indices[0], edge_indices[1]]), dtype=torch.long)
        edge_weights = torch.tensor(adjacency_matrix[edge_indices], dtype=torch.float32)
        
        return edge_index, edge_weights
    
    def build_graph(self, features: np.ndarray, symbols: List[str], 
                   node_features: Optional[np.ndarray] = None) -> Data:
        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(features)
        
        # Build adjacency matrix
        adjacency_matrix = self.build_adjacency_matrix(correlation_matrix)
        
        # Create edge index and weights
        edge_index, edge_attr = self.create_edge_index_and_weights(adjacency_matrix)
        
        # Prepare node features
        if node_features is None:
            # Use aggregated statistical features as node features
            node_features = self.create_node_features(features)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(symbols)
        )
        
        # Add metadata
        graph_data.symbols = symbols
        graph_data.correlation_matrix = correlation_matrix
        graph_data.adjacency_matrix = adjacency_matrix
        
        return graph_data
    
    def create_node_features(self, features: np.ndarray) -> np.ndarray:
        n_stocks, n_timesteps, n_features = features.shape
        node_features = []
        
        for i in range(n_stocks):
            stock_features = features[i]
            
            # Statistical aggregations
            feature_stats = []
            for j in range(n_features):
                feat_col = stock_features[:, j]
                valid_feat = feat_col[~np.isnan(feat_col)]
                
                if len(valid_feat) > 0:
                    feature_stats.extend([
                        np.mean(valid_feat),
                        np.std(valid_feat),
                        np.min(valid_feat),
                        np.max(valid_feat),
                        np.median(valid_feat)
                    ])
                else:
                    feature_stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            node_features.append(feature_stats)
        
        return np.array(node_features)
    
    def add_temporal_edges(self, graph_data: Data, lag: int = 1) -> Data:
        # Add temporal connections between the same stock at different time steps
        # This creates a dynamic graph structure
        n_nodes = graph_data.num_nodes
        
        # Create temporal edges (stock_i at time t -> stock_i at time t+1)
        temporal_edge_index = []
        for i in range(n_nodes):
            for t in range(lag):
                temporal_edge_index.append([i, i])  # Self-temporal connection
        
        temporal_edge_index = torch.tensor(temporal_edge_index, dtype=torch.long).t()
        temporal_edge_attr = torch.ones(temporal_edge_index.shape[1], dtype=torch.float32)
        
        # Combine with existing edges
        combined_edge_index = torch.cat([graph_data.edge_index, temporal_edge_index], dim=1)
        combined_edge_attr = torch.cat([graph_data.edge_attr, temporal_edge_attr])
        
        graph_data.edge_index = combined_edge_index
        graph_data.edge_attr = combined_edge_attr
        
        return graph_data
    
    def visualize_graph(self, graph_data: Data, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        
        # Create NetworkX graph for visualization
        G = nx.Graph()
        
        # Add nodes
        for i, symbol in enumerate(graph_data.symbols):
            G.add_node(i, label=symbol)
        
        # Add edges
        edge_index = graph_data.edge_index.numpy()
        edge_attr = graph_data.edge_attr.numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            weight = edge_attr[i]
            G.add_edge(src, dst, weight=weight)
        
        # Draw graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights], 
                              alpha=0.6, edge_color='gray')
        
        # Draw labels
        labels = {i: symbol for i, symbol in enumerate(graph_data.symbols)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Stock Correlation Graph")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return G