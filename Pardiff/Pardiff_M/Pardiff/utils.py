import os
import re
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial
from torch_geometric.utils import to_networkx, to_dense_adj, to_dense_batch
from torch_geometric.data import Data
from pardiff.parallel.transform import ToParallelBlocks


def extract_dense_graph(batch, one_hot=True):
    """Convert batched PyG graph to dense format, remove self-loops, apply mask."""
    batch_idx = getattr(batch, 'batch', None)
    x, mask = to_dense_batch(batch.x, batch_idx)
    adj = to_dense_adj(batch.edge_index, batch_idx, batch.edge_attr, max_num_nodes=x.size(1))

    eye_mask = torch.eye(adj.size(1), dtype=torch.bool, device=adj.device)

    if one_hot:
        adj[:, :, :, 0] = 1 - adj.sum(dim=-1)
        adj[~(mask.unsqueeze(1) * mask.unsqueeze(2))] = 0
        adj[:, eye_mask] = 0
    else:
        adj[:, eye_mask] = 0
        adj[~(mask.unsqueeze(1) * mask.unsqueeze(2))] = -1

    return x, adj, mask


def visualize_graphs(graph_list, node_size=20, edge_width=0.2, layout='spring', node_feat='x', edge_feat='edge_attr', block_attr=None):
    """Draws a list of NetworkX graphs in a grid layout."""
    num_graphs = len(graph_list)
    rows = int(np.sqrt(num_graphs))
    cols = int(np.ceil(num_graphs / rows))
    plt.figure(figsize=(rows * 4, cols * 3))

    for idx, graph in enumerate(graph_list):
        pos = nx.spring_layout(graph) if layout == 'spring' else nx.spectral_layout(graph)
        plt.subplot(rows, cols, idx + 1)

        node_colors = [v for _, v in graph.nodes(data=node_feat)] if node_feat else 'blue'
        edge_colors = [v for _, _, v in graph.edges(data=edge_feat)] if edge_feat else 'black'
        size = node_size

        if block_attr:
            block_vals = np.array([v for _, v in graph.nodes(data=block_attr)])
            size *= (1 + 5 * block_vals)

        nx.draw(
            graph, pos=pos,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=size,
            linewidths=edge_width,
            cmap='coolwarm'
        )
    return plt


def draw_pyg_graphs(data_objects, **kwargs):
    """Convert PyG data to NetworkX graphs and visualize."""
    graphs = [
        to_networkx(
            data,
            to_undirected='lower',
            node_attrs=([kwargs['node_attr']] if kwargs.get('node_attr') else []) +
                        ([kwargs['block']] if kwargs.get('block') else []),
            edge_attrs=[kwargs['edge_attr']] if kwargs.get('edge_attr') else []
        )
        for data in data_objects
    ]
    return visualize_graphs(graphs, **kwargs)


def convert_batch_to_pyg(nodes, edges, block_ids=None):
    """Reconstruct PyG Data objects from one-hot batch tensors."""
    node_mask = nodes.sum(dim=-1) == 1
    batch_ids, node_ids = node_mask.nonzero(as_tuple=True)
    node_feats = nodes[batch_ids, node_ids]

    edge_mask = (edges.sum(dim=-1) == 1) & (edges[..., 0] == 0)
    e_batch, src_idx, tgt_idx = edge_mask.nonzero(as_tuple=True)
    edge_feats = edges[e_batch, src_idx, tgt_idx]

    if block_ids is not None:
        valid_blocks = block_ids >= 0
        blk_batch, blk_idx = valid_blocks.nonzero(as_tuple=True)
        blk_vals = block_ids[blk_batch, blk_idx]

    data_list = []
    for i in range(nodes.size(0)):
        x_i = node_feats[batch_ids == i].argmax(dim=-1)
        edges_i = torch.stack([src_idx[e_batch == i], tgt_idx[e_batch == i]])
        edge_attrs_i = edge_feats[e_batch == i].argmax(dim=-1)
        blk_i = blk_vals[blk_batch == i] if block_ids is not None else None

        data = Data(x=x_i, edge_index=edges_i, edge_attr=edge_attrs_i, block_id=blk_i)
        data_list.append(data)

    return data_list


def validate_block_assignment(generated_nodes, generated_edges, predicted_blocks, max_hops=3):
    """Checks if generated block assignments match block decomposition logic."""
    samples = convert_batch_to_pyg(generated_nodes, generated_edges, predicted_blocks)
    transform = partial(ToParallelBlocks(max_hops=max_hops), only_return_blockid=True)
    processed = [transform(sample) for sample in samples]
    match = [(d.block_id == d.node_block_id).all().int().item() for d in processed]
    return match


def convert_to_discrete_tuples(nodes, edges):
    """Extract integer node labels and adjacency matrices for metric evaluation."""
    mask = nodes.sum(dim=-1) == 1
    lengths = mask.sum(dim=-1)
    return [
        (nodes[i, :lengths[i]].argmax(dim=-1), edges[i, :lengths[i], :lengths[i]].argmax(dim=-1))
        for i in range(len(lengths))
    ]


def get_best_or_latest_ckpt(path, return_latest=False):
    """Find the checkpoint with the lowest validation loss or most recent epoch."""
    best_loss = float('inf')
    best_epoch = 0
    best_file = None
    latest_file = None

    regex = re.compile(r'epoch=(\d+)-val_loss=([\d.]+)\.ckpt')
    for f in os.listdir(path):
        match = regex.search(f)
        if not match:
            continue
        epoch, val_loss = int(match.group(1)), float(match.group(2))
        if epoch > best_epoch:
            best_epoch = epoch
            latest_file = f
        if val_loss < best_loss:
            best_loss = val_loss
            best_file = f

    chosen = latest_file if return_latest else best_file
    return os.path.join(path, chosen) if chosen else "No matching checkpoint found."
