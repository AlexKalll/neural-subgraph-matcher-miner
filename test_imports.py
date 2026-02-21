"""Quick import test for the attribute-aware heterogeneous modifications."""
import sys
print("Python:", sys.version)

# Test 1: AttributeEncoder module
from common.attribute_encoder import AttributeEncoder, EdgeAttributeEncoder
print("[OK] attribute_encoder imports")

# Test 2: Models module (SkipLastGNN, SAGEConv, GINConv, OrderEmbedder)
from common.models import OrderEmbedder, SkipLastGNN, SAGEConv, GINConv
print("[OK] models imports")

# Test 3: Instantiate with default args (backward compat - no attributes)
import argparse
from subgraph_matching.config import parse_encoder
from common import utils

parser = argparse.ArgumentParser()
utils.parse_optimizer(parser)
parse_encoder(parser)
args = parser.parse_args([])

import torch
model = OrderEmbedder(1, args.hidden_dim, args)
print("[OK] OrderEmbedder(1, 64) created (topology-only, backward compat)")
print(f"     emb_model.semantic_dim = {model.emb_model.semantic_dim}")
print(f"     emb_model.edge_type_dim = {model.emb_model.edge_type_dim}")
assert model.emb_model.semantic_dim == 0, "semantic_dim should be 0 with default args"
assert model.emb_model.edge_type_dim == 0, "edge_type_dim should be 0 with default args"

# Test 4: Instantiate WITH node/edge types
args.node_type_vocab = 10
args.node_type_dim = 16
args.edge_type_vocab = 5
args.edge_type_dim = 8
model_het = OrderEmbedder(1, args.hidden_dim, args)
print(f"[OK] OrderEmbedder with heterogeneous support created")
print(f"     emb_model.semantic_dim = {model_het.emb_model.semantic_dim}")
print(f"     emb_model.edge_type_dim = {model_het.emb_model.edge_type_dim}")
print(f"     emb_model.alpha = {model_het.emb_model.alpha.item():.2f}")
assert model_het.emb_model.semantic_dim == 16
assert model_het.emb_model.edge_type_dim == 8

# Test 5: Forward pass without attributes (backward compat)
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
import networkx as nx

g1 = nx.path_graph(5)
for v in g1.nodes:
    g1.nodes[v]['node_feature'] = torch.ones(1)
g2 = nx.path_graph(3)
for v in g2.nodes:
    g2.nodes[v]['node_feature'] = torch.ones(1)
batch = Batch.from_data_list([DSGraph(g1), DSGraph(g2)])
batch = batch.to(utils.get_device())

with torch.no_grad():
    # topology-only model
    emb_topo = model.emb_model(batch)
    print(f"[OK] Forward pass (topology-only): shape={emb_topo.shape}")

# Test 6: Forward pass with attributes
# Create graphs with node_type_idx attribute
# NOTE: DeepSnap reserves the names 'node_type' and 'edge_type',
# so we use 'node_type_idx' and 'edge_type_idx' instead.
g1 = nx.path_graph(5)
for v in g1.nodes:
    g1.nodes[v]['node_feature'] = torch.ones(1)
    g1.nodes[v]['node_type_idx'] = torch.tensor(v % 3)  # types: 0,1,2
g2 = nx.path_graph(3)
for v in g2.nodes:
    g2.nodes[v]['node_feature'] = torch.ones(1)
    g2.nodes[v]['node_type_idx'] = torch.tensor(v % 2)  # types: 0,1

ds1 = DSGraph(g1)
ds2 = DSGraph(g2)
batch_het = Batch.from_data_list([ds1, ds2])
batch_het = batch_het.to(utils.get_device())

with torch.no_grad():
    model_het.to(utils.get_device())
    emb_het = model_het.emb_model(batch_het)
    print(f"[OK] Forward pass (with node types): shape={emb_het.shape}")

# Test 7: Different node types produce different embeddings
# Two graphs with SAME topology but DIFFERENT node types
g_a = nx.path_graph(4)
g_b = nx.path_graph(4)
for v in g_a.nodes:
    g_a.nodes[v]['node_feature'] = torch.ones(1)
    g_a.nodes[v]['node_type_idx'] = torch.tensor(0)  # all type 0
for v in g_b.nodes:
    g_b.nodes[v]['node_feature'] = torch.ones(1)
    g_b.nodes[v]['node_type_idx'] = torch.tensor(1)  # all type 1

batch_a = Batch.from_data_list([DSGraph(g_a)]).to(utils.get_device())
batch_b = Batch.from_data_list([DSGraph(g_b)]).to(utils.get_device())

with torch.no_grad():
    emb_a = model_het.emb_model(batch_a)
    emb_b = model_het.emb_model(batch_b)
    diff = torch.norm(emb_a - emb_b).item()
    print(f"[OK] Same topology, different types -> embedding diff = {diff:.6f}")
    assert diff > 1e-6, "Embeddings should differ for different node types!"

# Test 8: Order embedding loss is unchanged
labels = torch.tensor([1, 0]).to(utils.get_device())
emb_as = torch.randn(2, 64).to(utils.get_device())
emb_bs = torch.randn(2, 64).to(utils.get_device())
pred = (emb_as, emb_bs)
loss = model_het.criterion(pred, None, labels)
print(f"[OK] Order embedding loss computed: {loss.item():.4f}")

print("\n=== ALL TESTS PASSED ===")
