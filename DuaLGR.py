import argparse
import os.path

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from utils import load_data, normalize_weight, cal_homo_ratio, load_knowledge_graph, create_subgraphs
from models import EnDecoder, DuaLGR, GNN
from evaluation import eva
from settings import get_settings
import matplotlib.pyplot as plt
from visulization import plot_loss, plot_tsne
import pandas as pd

torch.autograd.set_detect_anomaly(True)  # Temporarily enable for debugging

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='knowledge_graph', help='datasets: acm, dblp, texas, chameleon, acm00, acm01, acm02, acm03, acm04, acm05, knowledge_graph')
parser.add_argument('--train', type=bool, default=True, help='training mode')
parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number')
parser.add_argument('--use_cuda', type=bool, default=True, help='use CUDA if available')
parser.add_argument('--subgraph_size', type=int, default=1000, help='number of nodes per subgraph')
args = parser.parse_args()

dataset = args.dataset
train = args.train
cuda_device = args.cuda_device
use_cuda = args.use_cuda
subgraph_size = args.subgraph_size

settings = get_settings(dataset)

path = settings.path
weight_soft = settings.weight_soft
alpha = settings.alpha
quantize = settings.quantize
varepsilon = settings.varepsilon
endecoder_hidden_dim = settings.endecoder_hidden_dim
hidden_dim = settings.hidden_dim
latent_dim = settings.latent_dim
pretrain = settings.pretrain
epoch = settings.epoch
patience = settings.patience
endecoder_lr = settings.endecoder_lr
endecoder_weight_decay = settings.endecoder_weight_decay
lr = settings.lr

weight_decay = settings.weight_decay
update_interval = settings.update_interval
random_seed = settings.random_seed
torch.manual_seed(random_seed)

if dataset == 'knowledge_graph':
    labels, adjs_labels, shared_feature, shared_feature_label, graph_num = load_knowledge_graph(path)
else:
    labels, adjs_labels, shared_feature, shared_feature_label, graph_num = load_data(dataset, path)

for v in range(graph_num):
    r = cal_homo_ratio(adjs_labels[v].cpu().numpy(), labels.cpu().numpy(), self_loop=True)
    print(r)
print('nodes: {}'.format(shared_feature_label.shape[0]))
print('features: {}'.format(shared_feature_label.shape[1]))
print('class: {}'.format(labels.max() + 1))

feat_dim = shared_feature.shape[1]
class_num = int(labels.max() + 1)
y = labels.cpu().numpy()

endecoder = EnDecoder(feat_dim, endecoder_hidden_dim, class_num)
model = DuaLGR(feat_dim, hidden_dim, latent_dim, endecoder, class_num=class_num, num_view=graph_num)
print(model)
if use_cuda:
    torch.cuda.set_device(cuda_device)
    torch.cuda.manual_seed(random_seed)
    endecoder = endecoder.cuda()
    model = model.cuda()
    # Process adjacency matrices in batches
    batch_size = subgraph_size  # Use subgraph_size as batch size
    processed_adjs = []
    for adj_labels in adjs_labels:
        if adj_labels.shape[0] > batch_size:
            # Process large adjacency matrices in chunks
            chunks = []
            for i in range(0, adj_labels.shape[0], batch_size):
                end_idx = min(i + batch_size, adj_labels.shape[0])
                chunk = adj_labels[i:end_idx, i:end_idx].cuda()
                chunks.append(chunk)
            processed_adjs.append(chunks)
        else:
            processed_adjs.append(adj_labels.cuda())
    adjs_labels = processed_adjs
    shared_feature = shared_feature.cuda()
    shared_feature_label = shared_feature_label.cuda()
    
    # Free up some memory
    torch.cuda.empty_cache()
device = shared_feature.device

if train:
    # Clear GPU cache before training
    torch.cuda.empty_cache()

    # =============================================== pretrain endecoder ============================
    print('shared_feature_label for clustering...')
    kmeans = KMeans(n_clusters=class_num, n_init=5)
    y_pred = kmeans.fit_predict(shared_feature_label.data.cpu().numpy())
    eva(y, y_pred, 'Kz')
    print()

    optimizer_endecoder = Adam(endecoder.parameters(), lr=endecoder_lr, weight_decay=endecoder_weight_decay)

    for epoch_num in range(pretrain):
        endecoder.train()
        loss_re = 0.
        loss_a = 0.

        # Process in batches
        for i in range(0, shared_feature.shape[0], subgraph_size):
            end_idx = min(i + subgraph_size, shared_feature.shape[0])
            batch_feature = shared_feature[i:end_idx]
            batch_feature_label = shared_feature_label[i:end_idx]
            
            a_pred, x_pred, z_norm = endecoder(batch_feature)
            
            # Compute loss for adjacency matrix prediction
            for v in range(graph_num):
                if isinstance(adjs_labels[v], list):  # If adjacency matrix was chunked
                    chunk_idx = i // subgraph_size
                    if chunk_idx < len(adjs_labels[v]):
                        batch_adj = adjs_labels[v][chunk_idx]
                        loss_a += F.binary_cross_entropy(a_pred, batch_adj)
                else:  # If adjacency matrix was small enough to fit in GPU
                    batch_adj = adjs_labels[v][i:end_idx, i:end_idx]
                    loss_a += F.binary_cross_entropy(a_pred, batch_adj)
            
            loss_re += F.binary_cross_entropy(x_pred, batch_feature_label)

            # Free up memory
            del a_pred, x_pred
            torch.cuda.empty_cache()

        loss = loss_re + loss_a
        optimizer_endecoder.zero_grad()
        loss.backward()
        optimizer_endecoder.step()
        print('epoch: {}, loss:{}, loss_re:{}, loss_a: {}'.format(epoch_num, loss, loss_re, loss_a))

        if epoch_num == pretrain - 1:
            print('Pretrain complete...')
            # Process final embeddings in batches
            all_z_norm = []
            with torch.no_grad():
                for i in range(0, shared_feature.shape[0], subgraph_size):
                    end_idx = min(i + subgraph_size, shared_feature.shape[0])
                    batch_feature = shared_feature[i:end_idx]
                    _, _, batch_z_norm = endecoder(batch_feature)
                    all_z_norm.append(batch_z_norm.cpu())
            
            z_norm = torch.cat(all_z_norm, dim=0)
            kmeans = KMeans(n_clusters=class_num, n_init=5)
            y_pred = kmeans.fit_predict(z_norm.numpy())
            eva(y, y_pred, 'Kz')
            break


    # =========================================Train=============================================================
    print('Begain trains...')
    param_all = []
    for v in range(graph_num+1):
        param_all.append({'params': model.cluster_layer[v]})
    param_all.append({'params': model.gnn.parameters()})
    optimizer_model = Adam(param_all, lr=lr, weight_decay=weight_decay)

    best_a = [1e-12 for i in range(graph_num)]
    weights = normalize_weight(best_a)

    # Initialize clustering
    with torch.no_grad():
        model.eval()
        pseudo_label = y_pred
        all_z_all = []
        # Process in batches
        for i in range(0, shared_feature.shape[0], subgraph_size):
            end_idx = min(i + subgraph_size, shared_feature.shape[0])
            batch_feature = shared_feature[i:end_idx]
            batch_adjs = []
            for adj in adjs_labels:
                if isinstance(adj, list):
                    chunk_idx = i // subgraph_size
                    if chunk_idx < len(adj):
                        batch_adjs.append(adj[chunk_idx])
                    else:
                        batch_adjs.append(adj[-1])
                else:
                    batch_adjs.append(adj[i:end_idx, i:end_idx])
            
            batch_pseudo_label = pseudo_label[i:end_idx]  # Get batch pseudo labels
            a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(batch_feature, batch_adjs, weights, batch_pseudo_label, alpha, quantize=quantize, varepsilon=varepsilon)
            all_z_all.append(z_all[-1].cpu())

        # Combine results from all batches
        z_combined = torch.cat(all_z_all, dim=0)
        kmeans = KMeans(n_clusters=class_num, n_init=5)
        y_pred = kmeans.fit_predict(z_combined.numpy())
        for v in range(graph_num+1):
            model.cluster_layer[v].data = torch.tensor(kmeans.cluster_centers_).to(device)
        pseudo_label = y_pred

    bad_count = 0
    best_acc = 1e-12
    best_nmi = 1e-12
    best_ari = 1e-12
    best_f1 = 1e-12
    best_epoch = 0

    nmi_list = []
    acc_list = []
    loss_list = []

    for epoch_num in range(epoch):
        model.train()
        loss_re = 0.
        loss_kl = 0.
        loss_re_a = 0.
        loss_re_ax = 0.
        loss_re_x = 0.

        # Process in batches
        for i in range(0, shared_feature.shape[0], subgraph_size):
            end_idx = min(i + subgraph_size, shared_feature.shape[0])
            batch_feature = shared_feature[i:end_idx]
            batch_adjs = []
            for adj in adjs_labels:
                if isinstance(adj, list):
                    chunk_idx = i // subgraph_size
                    if chunk_idx < len(adj):
                        batch_adjs.append(adj[chunk_idx])
                    else:
                        batch_adjs.append(adj[-1])
                else:
                    batch_adjs.append(adj[i:end_idx, i:end_idx])
            
            # Get the correct subset of pseudo labels for this batch
            batch_pseudo_label = pseudo_label[i:end_idx]
            
            optimizer_model.zero_grad()
            
            try:
                # Forward pass with batch_pseudo_label instead of full pseudo_label
                a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(
                    batch_feature, batch_adjs, weights, batch_pseudo_label, 
                    alpha, quantize=quantize, varepsilon=varepsilon
                )
                
                # KMeans clustering
                kmeans = KMeans(n_clusters=class_num, n_init=5)
                y_prim = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
                batch_pseudo_label = y_prim
                
                # Update weights
                for v in range(graph_num):
                    y_pred = kmeans.fit_predict(z_all[v].detach().cpu().numpy())
                    a = eva(y_prim, y_pred, visible=False, metrics='nmi')
                    best_a[v] = a
                
                weights = normalize_weight(best_a, p=weight_soft)
                
                # Calculate KL divergence loss
                p = model.target_distribution(q_all[-1].detach())  # Detach target distribution
                kl_loss = torch.zeros(1, device=device)
                for v in range(graph_num):
                    kl_loss += F.kl_div(q_all[v].log(), p, reduction='batchmean')
                kl_loss += F.kl_div(q_all[-1].log(), p, reduction='batchmean')
                
                # Calculate reconstruction losses
                a_pred = torch.clamp(a_pred, 0, 1)
                x_pred = torch.clamp(x_pred, 0, 1)
                batch_feature = batch_feature.clamp(0, 1)
                
                recon_loss_a = torch.zeros(1, device=device)
                for v in range(graph_num):
                    batch_adj = batch_adjs[v].float().clamp(0, 1)
                    recon_loss_a += F.binary_cross_entropy(a_pred, batch_adj)
                
                recon_loss_x = F.binary_cross_entropy(x_pred, batch_feature)
                
                # Combine losses
                total_loss = recon_loss_a + recon_loss_x + kl_loss
                
                # Backward pass and optimization
                total_loss.backward()
                optimizer_model.step()
                
                # Update metrics
                loss_re_a += recon_loss_a.item()
                loss_re_x += recon_loss_x.item()
                loss_kl += kl_loss.item()
                loss_re = loss_re_a + loss_re_x
                
                # Update global pseudo_label
                pseudo_label[i:end_idx] = y_prim
                
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print("CUDA out of memory. Reducing batch size and retrying...")
                    torch.cuda.empty_cache()
                    new_end_idx = i + (end_idx - i) // 2
                    continue
                else:
                    raise e

            print('epoch: {}, loss: {:.4f}, loss_re: {:.4f}, loss_kl: {:.4f}, loss_re_a: {:.4f}, loss_re_x: {:.4f}, badcount: {}'.format(
                epoch_num, total_loss.item(), loss_re, loss_kl, loss_re_a, loss_re_x, bad_count))
        if epoch_num % update_interval == 0:
            model.eval()
            with torch.no_grad():
                # Create subgraphs first
                subgraphs = create_subgraphs(shared_feature, adjs_labels, subgraph_size)
                
                # Initialize metrics for all subgraphs
                all_predictions = []
                
                for subgraph in subgraphs:
                    sub_shared_feature, sub_adjs_labels = subgraph
                    a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(
                        sub_shared_feature, 
                        sub_adjs_labels, 
                        weights, 
                        pseudo_label, 
                        alpha, 
                        quantize=quantize, 
                        varepsilon=varepsilon
                    )
                    
                    kmeans = KMeans(n_clusters=class_num, n_init=5)
                    y_eval = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
                    all_predictions.extend(y_eval)
                
                # Convert predictions to numpy array
                all_predictions = np.array(all_predictions)
                
                # Evaluate on the complete predictions
                nmi, acc, ari, f1 = eva(y, all_predictions, str(epoch_num) + 'Kz')
                nmi_list.append(nmi)
                acc_list.append(acc)

        if acc > best_acc:
            if os.path.exists('./pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc)):
                os.remove('./pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc))
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = epoch_num
            bad_count = 0
            torch.save({'state_dict': model.state_dict(),
                        'state_dict_endecoder': endecoder.state_dict(),
                        'weights': weights,
                        'pseudo_label': pseudo_label},
                       './pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc))
            print('best acc:{}, best nmi:{}, best ari:{}, best f1:{}, bestepoch:{}'.format(
                                         best_acc, best_nmi, best_ari, best_f1, best_epoch))
        else:
            bad_count += 1

        if bad_count >= patience:
            print('complete training, best acc:{}, best nmi:{}, best ari:{}, best f1:{}, bestepoch:{}'.format(
                best_acc, best_nmi, best_ari, best_f1, best_epoch))
            print()
            break

if not train:
    model_name = settings.model_name
else:
    model_name = 'dualgr_{}_acc{:.4f}'.format(dataset, best_acc)

best_model = torch.load('./pkl/'+model_name+'.pkl', map_location=shared_feature.device)
state_dic = best_model['state_dict']
state_dic_encoder = best_model['state_dict_endecoder']
weights = best_model['weights']
pseudo_label = best_model['pseudo_label']

endecoder.load_state_dict(state_dic_encoder)
model.load_state_dict(state_dic)

model.eval()
with torch.no_grad():
    model.endecoder = endecoder
    all_predictions = []
    
    for i in range(0, shared_feature.shape[0], subgraph_size):
        end_idx = min(i + subgraph_size, shared_feature.shape[0])
        batch_feature = shared_feature[i:end_idx]
        batch_adjs = []
        for adj in adjs_labels:
            if isinstance(adj, list):
                chunk_idx = i // subgraph_size
                if chunk_idx < len(adj):
                    batch_adjs.append(adj[chunk_idx])
                else:
                    batch_adjs.append(adj[-1])
            else:
                batch_adjs.append(adj[i:end_idx, i:end_idx])
        
        # Get the correct subset of pseudo labels
        batch_pseudo_label = pseudo_label[i:end_idx]
        
        a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(
            batch_feature, batch_adjs, weights, batch_pseudo_label, 
            alpha, quantize=quantize, varepsilon=varepsilon
        )
        
        kmeans = KMeans(n_clusters=class_num, n_init=5)
        y_eval = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
        all_predictions.extend(y_eval)
    
    # Evaluate using all predictions
    all_predictions = np.array(all_predictions)
    nmi, acc, ari, f1 = eva(y, all_predictions, 'Final Kz')

    # Save clustering results to CSV
    clustering_results = pd.DataFrame({'Node': range(len(y_eval)), 'Cluster': y_eval})
    clustering_results.to_csv('clustering_results.csv', index=False)

print('Test complete...')
