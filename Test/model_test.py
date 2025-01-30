import sys
import os
sys.path.append('../')

from models import DuaLGR, EnDecoder
from utils import load_data, normalize_weight
from settings import get_settings
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Generate synthetic test data
num_nodes = 100  # Number of nodes
feat_dim = 64    # Feature dimension
hidden_dim = 32  # Hidden dimension
latent_dim = 16  # Latent dimension
class_num = 3    # Number of classes
graph_num = 2    # Number of views

# Generate synthetic features
shared_feature = torch.randn(num_nodes, feat_dim)
shared_feature_label = torch.randn(num_nodes, feat_dim)

# Generate synthetic labels
labels = torch.randint(0, class_num, (num_nodes,))

# Generate synthetic adjacency matrices
adjs_labels = []
for _ in range(graph_num):
    # Create a random sparse adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes)
    # Add random edges (20% density)
    edge_prob = torch.rand(num_nodes, num_nodes)
    adj[edge_prob < 0.2] = 1
    # Make it symmetric
    adj = torch.maximum(adj, adj.t())
    adjs_labels.append(adj)

# Initialize weights and pseudo labels
best_a = [1e-12 for i in range(graph_num)]
weights = normalize_weight(best_a)

# Keep pseudo_label on CPU since it needs to be converted to numpy
pseudo_label = torch.zeros(labels.size(0), dtype=torch.long)

# Move tensors to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shared_feature = shared_feature.to(device)
shared_feature_label = shared_feature_label.to(device)
labels = labels.to(device)
adjs_labels = [adj.to(device) for adj in adjs_labels]

# Initialize encoder
endecoder = EnDecoder(
    feat_dim,
    hidden_dim,
    class_num
).to(device)

# Initialize model
dualrg = DuaLGR(
    feat_dim=feat_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    endecoder=endecoder,
    class_num=class_num,
    num_view=graph_num
).to(device)

# Print model configuration
print("Model Configuration:")
print(f"Feature Dimension: {feat_dim}")
print(f"Hidden Dimension: {hidden_dim}")
print(f"Latent Dimension: {latent_dim}")
print(f"Number of Classes: {class_num}")
print(f"Number of Views: {graph_num}")
print(f"Number of Nodes: {num_nodes}")

# Run model and collect outputs
with torch.no_grad():
    # Keep pseudo_label on CPU for numpy conversion
    a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = dualrg(
        shared_feature, 
        adjs_labels, 
        weights, 
        pseudo_label,  # This stays on CPU
        alpha=0.5
    )

print("\nModel Output Shapes:")
print(f"a_pred shape: {a_pred.shape}")
print(f"x_pred shape: {x_pred.shape}")
print(f"z_all length: {len(z_all)}")
print(f"q_all length: {len(q_all)}")
print(f"a_pred_x shape: {a_pred_x.shape}")
print(f"x_pred_x shape: {x_pred_x.shape}")

