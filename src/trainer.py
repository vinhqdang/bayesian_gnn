import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from models.bayesian_gnn import BayesianGNN, TemporalBayesianGNN
from data.data_loader import StockDataLoader, MarketDataPreprocessor
from data.graph_builder import StockGraphBuilder

class BayesianGNNTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def elbo_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                  kl_weight: float = 1e-4) -> torch.Tensor:
        """Evidence Lower Bound (ELBO) loss for Bayesian models"""
        
        # Likelihood term (reconstruction loss)
        if self.model.use_uncertainty:
            mean, var = predictions
            # Negative log likelihood for Gaussian distribution
            likelihood_loss = 0.5 * torch.log(2 * np.pi * var) + 0.5 * (targets - mean)**2 / var
            likelihood_loss = likelihood_loss.mean()
        else:
            likelihood_loss = nn.MSELoss()(predictions, targets)
        
        # KL divergence term (regularization)
        kl_div = self.model.kl_divergence()
        
        # ELBO = -log p(y|x) + β * KL(q(θ)||p(θ))
        elbo = likelihood_loss + kl_weight * kl_div
        
        return elbo, likelihood_loss, kl_div
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   kl_weight: float = 1e-4) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        total_likelihood = 0
        total_kl = 0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            x, edge_index, edge_attr, y = [b.to(self.device) for b in batch]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(x, edge_index, edge_attr, sample=True)
            
            # Calculate loss
            loss, likelihood_loss, kl_div = self.elbo_loss(predictions, y, kl_weight)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_likelihood += likelihood_loss.item()
            total_kl += kl_div.item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'likelihood': total_likelihood / n_batches,
            'kl_div': total_kl / n_batches
        }
    
    def validate_epoch(self, val_loader: DataLoader, kl_weight: float = 1e-4) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0
        total_likelihood = 0
        total_kl = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, edge_index, edge_attr, y = [b.to(self.device) for b in batch]
                
                # Forward pass (multiple samples for better uncertainty estimation)
                predictions = self.model(x, edge_index, edge_attr, sample=True)
                
                # Calculate loss
                loss, likelihood_loss, kl_div = self.elbo_loss(predictions, y, kl_weight)
                
                total_loss += loss.item()
                total_likelihood += likelihood_loss.item()
                total_kl += kl_div.item()
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'likelihood': total_likelihood / n_batches,
            'kl_div': total_kl / n_batches
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              n_epochs: int = 100, lr: float = 1e-3, kl_weight: float = 1e-4,
              patience: int = 10, save_path: str = None) -> Dict[str, List[float]]:
        """Full training loop"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader, optimizer, kl_weight)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, kl_weight)
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train Likelihood: {train_metrics['likelihood']:.4f}, KL Div: {train_metrics['kl_div']:.4f}")
            
            # Save training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': best_val_loss,
                        'training_history': self.training_history
                    }, save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.training_history
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['train_loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def create_data_loaders(features: np.ndarray, targets: np.ndarray, 
                       edge_index: torch.Tensor, edge_attr: torch.Tensor,
                       batch_size: int = 32, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders"""
    
    # Convert to tensors
    x_tensor = torch.FloatTensor(features)
    y_tensor = torch.FloatTensor(targets)
    
    # Split data
    n_samples = len(features)
    n_train = int(n_samples * train_split)
    
    # Create datasets
    train_dataset = TensorDataset(
        x_tensor[:n_train], 
        edge_index.repeat(n_train, 1, 1),
        edge_attr.repeat(n_train, 1),
        y_tensor[:n_train]
    )
    
    val_dataset = TensorDataset(
        x_tensor[n_train:],
        edge_index.repeat(n_samples - n_train, 1, 1),
        edge_attr.repeat(n_samples - n_train, 1),
        y_tensor[n_train:]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def prepare_training_data(symbols: List[str], start_date: str, end_date: str) -> Dict:
    """Prepare data for training"""
    
    print("Loading stock data...")
    data_loader = StockDataLoader(symbols, start_date, end_date)
    data_loader.fetch_data()
    
    print("Preprocessing data...")
    preprocessor = MarketDataPreprocessor()
    processed_data = preprocessor.preprocess_data(data_loader)
    
    print("Building graph...")
    graph_builder = StockGraphBuilder(correlation_threshold=0.3, top_k_edges=5)
    graph_data = graph_builder.build_graph(
        processed_data['features'], 
        processed_data['symbols']
    )
    
    return {
        'features': processed_data['features'],
        'X_train': processed_data['X_train'],
        'X_test': processed_data['X_test'],
        'y_train': processed_data['y_train'],
        'y_test': processed_data['y_test'],
        'symbols': processed_data['symbols'],
        'graph_data': graph_data
    }

if __name__ == "__main__":
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing training data...")
    data = prepare_training_data(SYMBOLS, START_DATE, END_DATE)
    
    # Model configuration
    model_config = {
        'input_dim': data['X_train'].shape[-1],
        'hidden_dims': [64, 32],
        'output_dim': 1,
        'gnn_type': 'GCN',
        'dropout': 0.1,
        'prior_std': 0.1,
        'use_uncertainty': True
    }
    
    # Create model
    model = BayesianGNN(**model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data['X_train'].reshape(-1, data['X_train'].shape[-1]),
        data['y_train'],
        data['graph_data'].edge_index,
        data['graph_data'].edge_attr,
        batch_size=32
    )
    
    # Create trainer
    trainer = BayesianGNNTrainer(model, device)
    
    # Train model
    print("Starting training...")
    training_history = trainer.train(
        train_loader, val_loader,
        n_epochs=100,
        lr=1e-3,
        kl_weight=1e-4,
        save_path='best_model.pth'
    )
    
    # Plot results
    trainer.plot_training_history()
    
    print("Training completed!")