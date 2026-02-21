"""Defines all graph embedding models"""
from functools import reduce
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from common import utils
from common import feature_preprocess
from common.attribute_encoder import AttributeEncoder, EdgeAttributeEncoder

# GNN -> concat -> MLP graph classification baseline
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred#.argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)

# Order embedder model -- contains a graph embedding model `emb_model`
class OrderEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e

    def criterion(self, pred, intersect_embs, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0); 
        for negative examples, the e term is trained to be at least greater than self.margin.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=utils.get_device()), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0,
            device=utils.get_device()), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss

class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        if len(feature_preprocess.FEATURE_AUGMENT) > 0:
            self.feat_preprocess = feature_preprocess.Preprocess(input_dim)
            input_dim = self.feat_preprocess.dim_out
        else:
            self.feat_preprocess = None

        # --- Heterogeneous attribute support ---
        # Node type embedding (categorical)
        node_type_vocab = getattr(args, 'node_type_vocab', 0)
        node_type_dim = getattr(args, 'node_type_dim', 0)
        self.has_node_type = node_type_vocab > 0 and node_type_dim > 0
        if self.has_node_type:
            self.node_type_emb = nn.Embedding(node_type_vocab, node_type_dim)
            nn.init.normal_(self.node_type_emb.weight, 0, 0.01)
        else:
            self.node_type_emb = None
        self.node_type_dim = node_type_dim if self.has_node_type else 0

        # General attribute encoder (optional, for custom attributes)
        attr_config = getattr(args, 'attribute_config', None)
        attr_dim = getattr(args, 'attr_dim', 0)
        if attr_config and attr_dim > 0:
            self.attribute_encoder = AttributeEncoder(attr_dim, attr_config)
            self.attr_dim = attr_dim if self.attribute_encoder.has_attributes else 0
        else:
            self.attribute_encoder = None
            self.attr_dim = 0

        # Total semantic dimension
        self.semantic_dim = self.node_type_dim + self.attr_dim

        # Learnable weight for semantic features (init small to preserve
        # pretrained structural geometry)
        if self.semantic_dim > 0:
            self.alpha = nn.Parameter(torch.tensor(0.1))
            input_dim = input_dim + self.semantic_dim

        # Edge type support (R-GCN style)
        edge_type_vocab = getattr(args, 'edge_type_vocab', 0)
        edge_type_dim = getattr(args, 'edge_type_dim', 0)
        self.edge_attr_encoder = EdgeAttributeEncoder(edge_type_vocab,
                                                       edge_type_dim)
        self.edge_type_dim = edge_type_dim if self.edge_attr_encoder.active else 0
        # --- End heterogeneous attribute support ---

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, 3*hidden_dim if
            args.conv_type == "PNA" else hidden_dim))

        conv_model = self.build_conv_model(args.conv_type, 1)
        if args.conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()

        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                self.n_layers))

        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "PNA":
                self.convs_sum.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(3*hidden_input_dim, hidden_dim))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)
        if args.conv_type == "PNA":
            post_input_dim *= 3
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim))
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = args.skip
        self.conv_type = args.conv_type

    def build_conv_model(self, model_type, n_inner_layers):
        etd = self.edge_type_dim
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            return lambda i, h: GINConv(nn.Sequential(
                nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)
                ), edge_type_dim=etd)
        elif model_type == "SAGE":
            return lambda i, h: SAGEConv(i, h, edge_type_dim=etd)
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, n_inner_layers)
        elif model_type == "PNA":
            return lambda i, h: SAGEConv(i, h, edge_type_dim=etd)
        else:
            print("unrecognized model type")

    def _conv_with_edge_type(self, conv, x, edge_index, edge_type_emb=None):
        """Call a conv layer, passing edge_type_emb only if supported."""
        if edge_type_emb is not None and hasattr(conv, 'edge_transform'):
            return conv(x, edge_index, edge_type_emb=edge_type_emb)
        return conv(x, edge_index)

    def forward(self, data):
        if self.feat_preprocess is not None:
            if not hasattr(data, "preprocessed"):
                data = self.feat_preprocess(data)
                data.preprocessed = True
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        # --- Append semantic embeddings (node types + custom attributes) ---
        # NOTE: We use 'node_type_idx' / 'edge_type_idx' because DeepSnap
        # reserves the names 'node_type' and 'edge_type' for its own
        # heterogeneous graph handling and silently drops them.
        if self.semantic_dim > 0:
            semantic_parts = []
            if self.has_node_type:
                node_type = getattr(data, 'node_type_idx', None)
                if node_type is not None:
                    semantic_parts.append(
                        self.node_type_emb(node_type.long()))
                else:
                    semantic_parts.append(
                        torch.zeros(x.size(0), self.node_type_dim,
                                    device=x.device))
            if self.attribute_encoder is not None and self.attr_dim > 0:
                attr_dict = getattr(data, 'node_attr_dict', None)
                attr_emb = self.attribute_encoder(attr_dict)
                if attr_emb is not None:
                    semantic_parts.append(attr_emb)
                else:
                    semantic_parts.append(
                        torch.zeros(x.size(0), self.attr_dim,
                                    device=x.device))
            if semantic_parts:
                semantic = torch.cat(semantic_parts, dim=-1)
                x = torch.cat([x, self.alpha * semantic], dim=-1)

        # --- Compute edge type embeddings (R-GCN) ---
        edge_type = getattr(data, 'edge_type_idx', None)
        if edge_type is not None and self.edge_attr_encoder.active:
            edge_type_emb = self.edge_attr_encoder(edge_type.long())
        else:
            edge_type_emb = None

        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs_sum) if self.conv_type=="PNA" else
            len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                    :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                if self.conv_type == "PNA":
                    x = torch.cat((
                        self._conv_with_edge_type(
                            self.convs_sum[i], curr_emb, edge_index,
                            edge_type_emb),
                        self._conv_with_edge_type(
                            self.convs_mean[i], curr_emb, edge_index,
                            edge_type_emb),
                        self._conv_with_edge_type(
                            self.convs_max[i], curr_emb, edge_index,
                            edge_type_emb)), dim=-1)
                else:
                    x = self._conv_with_edge_type(
                        self.convs[i], curr_emb, edge_index, edge_type_emb)
            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((
                        self._conv_with_edge_type(
                            self.convs_sum[i], emb, edge_index,
                            edge_type_emb),
                        self._conv_with_edge_type(
                            self.convs_mean[i], emb, edge_index,
                            edge_type_emb),
                        self._conv_with_edge_type(
                            self.convs_max[i], emb, edge_index,
                            edge_type_emb)), dim=-1)
                else:
                    x = self._conv_with_edge_type(
                        self.convs[i], emb, edge_index, edge_type_emb)
            else:
                x = self._conv_with_edge_type(
                    self.convs[i], x, edge_index, edge_type_emb)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add",
                 edge_type_dim=0):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
            out_channels)

        # R-GCN style: relation-specific message gating
        if edge_type_dim > 0:
            self.edge_transform = nn.Linear(edge_type_dim, out_channels)
            nn.init.normal_(self.edge_transform.weight, 0, 0.01)
            nn.init.zeros_(self.edge_transform.bias)
        else:
            self.edge_transform = None

    def forward(self, x, edge_index, edge_weight=None, edge_type_emb=None,
                size=None, res_n_id=None):
        # Remove self-loops; also filter edge_type_emb if present
        if edge_type_emb is not None:
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            edge_type_emb = edge_type_emb[mask]
        else:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight,
                              edge_type_emb=edge_type_emb,
                              res_n_id=res_n_id)

    def message(self, x_j, edge_weight, edge_type_emb=None):
        msg = self.lin(x_j)
        if edge_type_emb is not None and self.edge_transform is not None:
            # R-GCN: relation-specific gating on messages
            gate = torch.sigmoid(self.edge_transform(edge_type_emb))
            msg = msg * gate
        return msg

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)
        aggr_out = self.lin_update(aggr_out)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# pytorch geom GINConv + weighted edges
class GINConv(pyg_nn.MessagePassing):
    def __init__(self, nn_module, eps=0, train_eps=False, edge_type_dim=0,
                 **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn_module
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

        # R-GCN style: relation-specific message gating
        if edge_type_dim > 0:
            # Infer output dim from the nn sequential
            out_dim = None
            for m in reversed(list(self.nn.modules())):
                if isinstance(m, nn.Linear):
                    out_dim = m.out_features
                    break
            if out_dim is not None:
                self.edge_transform = nn.Linear(edge_type_dim, out_dim)
                nn.init.normal_(self.edge_transform.weight, 0, 0.01)
                nn.init.zeros_(self.edge_transform.bias)
            else:
                self.edge_transform = None
        else:
            self.edge_transform = None

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None, edge_type_emb=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Remove self-loops; also filter edge attributes if present
        if edge_type_emb is not None:
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            if edge_weight is not None:
                edge_weight = edge_weight[mask]
            edge_type_emb = edge_type_emb[mask]
        else:
            edge_index, edge_weight = pyg_utils.remove_self_loops(
                edge_index, edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(
            edge_index, x=x, edge_weight=edge_weight,
            edge_type_emb=edge_type_emb))
        return out

    def message(self, x_j, edge_weight, edge_type_emb=None):
        msg = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        if edge_type_emb is not None and self.edge_transform is not None:
            gate = torch.sigmoid(self.edge_transform(edge_type_emb))
            msg = msg * gate
        return msg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

