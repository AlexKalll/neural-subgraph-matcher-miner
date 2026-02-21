"""
Attribute Encoder Module for Heterogeneous Graph Support.

Provides modular encoding of arbitrary node/edge attributes:
  - Categorical attributes  -> nn.Embedding
  - Numerical attributes    -> Linear projection
  - Text attributes         -> Simple token embedding + mean pooling

All encodings are concatenated and projected to a fixed output dimension.
When no attributes are present, returns a zero vector (preserving
backward compatibility with topology-only synthetic data).

Embeddings are initialized with small variance N(0, 0.01) to avoid
disrupting pretrained structural geometry.

IMPORTANT - DeepSnap naming convention:
  DeepSnap reserves the attribute names 'node_type' and 'edge_type'
  for its own heterogeneous graph handling and silently drops them.
  We therefore use:
    - 'node_type_idx'  for node type indices
    - 'edge_type_idx'  for edge type indices
  These are properly propagated through DeepSnap Batch objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalEncoder(nn.Module):
    """Encodes a single categorical attribute via nn.Embedding."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Small-variance init to preserve pretrained geometry
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """
        Args:
            x: LongTensor of shape (N,) with category indices.
        Returns:
            Tensor of shape (N, embed_dim).
        """
        return self.embedding(x)


class NumericalEncoder(nn.Module):
    """Encodes numerical attributes via a small MLP projection."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        # Small-variance init
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape (N, input_dim).
        Returns:
            Tensor of shape (N, output_dim).
        """
        return self.mlp(x)


class TextEncoder(nn.Module):
    """
    Lightweight text attribute encoder.

    Uses a simple token-level embedding table + mean pooling.
    No heavy external dependencies (no transformers).
    """

    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pad_idx = pad_idx
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, token_ids):
        """
        Args:
            token_ids: LongTensor of shape (N, max_seq_len) with token indices.
        Returns:
            Tensor of shape (N, embed_dim) via mean pooling over non-pad tokens.
        """
        mask = (token_ids != self.pad_idx).unsqueeze(-1).float()  # (N, L, 1)
        emb = self.embedding(token_ids)  # (N, L, D)
        # Mean pool over non-pad positions
        lengths = mask.sum(dim=1).clamp(min=1.0)  # (N, 1)
        pooled = (emb * mask).sum(dim=1) / lengths  # (N, D)
        return pooled


class AttributeEncoder(nn.Module):
    """
    General-purpose attribute encoder that supports arbitrary combinations
    of categorical, numerical, and text attributes.

    Configuration is passed as a dict:
        attribute_config = {
            "categorical": {
                "node_type": {"vocab_size": 10, "embed_dim": 16},
                "color":     {"vocab_size": 5,  "embed_dim": 8},
            },
            "numerical": {
                "weight": {"input_dim": 1, "embed_dim": 8},
                "coords": {"input_dim": 3, "embed_dim": 16},
            },
            "text": {
                "label": {"vocab_size": 1000, "embed_dim": 16, "pad_idx": 0},
            },
        }

    If no config is provided (or empty), the encoder returns zero vectors.

    Args:
        output_dim: Fixed output dimension for the semantic embedding.
        attribute_config: Dict describing attributes to encode.
    """

    def __init__(self, output_dim, attribute_config=None):
        super().__init__()
        self.output_dim = output_dim
        self.attribute_config = attribute_config or {}

        self.cat_encoders = nn.ModuleDict()
        self.num_encoders = nn.ModuleDict()
        self.text_encoders = nn.ModuleDict()

        total_inner_dim = 0

        # Build categorical encoders
        for name, cfg in self.attribute_config.get("categorical", {}).items():
            enc = CategoricalEncoder(cfg["vocab_size"], cfg["embed_dim"])
            self.cat_encoders[name] = enc
            total_inner_dim += cfg["embed_dim"]

        # Build numerical encoders
        for name, cfg in self.attribute_config.get("numerical", {}).items():
            enc = NumericalEncoder(cfg["input_dim"], cfg["embed_dim"])
            self.num_encoders[name] = enc
            total_inner_dim += cfg["embed_dim"]

        # Build text encoders
        for name, cfg in self.attribute_config.get("text", {}).items():
            enc = TextEncoder(cfg["vocab_size"], cfg["embed_dim"],
                              pad_idx=cfg.get("pad_idx", 0))
            self.text_encoders[name] = enc
            total_inner_dim += cfg["embed_dim"]

        self._has_attributes = total_inner_dim > 0

        if self._has_attributes:
            self.projection = nn.Linear(total_inner_dim, output_dim)
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.projection.bias)
        else:
            self.projection = None

    @property
    def has_attributes(self):
        return self._has_attributes

    def forward(self, attr_dict):
        """
        Encode a dictionary of attribute tensors.

        Args:
            attr_dict: Dict mapping attribute names to tensors.
                       Missing attributes are silently skipped.
                       If empty or None, returns zero vector.

        Returns:
            Tensor of shape (N, output_dim).
        """
        if not self._has_attributes or attr_dict is None:
            # Determine N from any available tensor, or return empty
            return None

        parts = []
        n_nodes = None

        # Categorical
        for name, enc in self.cat_encoders.items():
            if name in attr_dict:
                val = attr_dict[name]
                parts.append(enc(val))
                if n_nodes is None:
                    n_nodes = val.size(0)

        # Numerical
        for name, enc in self.num_encoders.items():
            if name in attr_dict:
                val = attr_dict[name]
                if val.dim() == 1:
                    val = val.unsqueeze(-1)
                parts.append(enc(val))
                if n_nodes is None:
                    n_nodes = val.size(0)

        # Text
        for name, enc in self.text_encoders.items():
            if name in attr_dict:
                val = attr_dict[name]
                parts.append(enc(val))
                if n_nodes is None:
                    n_nodes = val.size(0)

        if not parts:
            return None

        concat = torch.cat(parts, dim=-1)
        return self.projection(concat)


class EdgeAttributeEncoder(nn.Module):
    """
    Encodes edge type as an embedding for R-GCN style message passing.

    If edge_type_vocab == 0, this encoder is inactive and returns None,
    preserving backward compatibility.

    Args:
        edge_type_vocab: Number of distinct edge types.
        edge_type_dim: Embedding dimension for edge types.
    """

    def __init__(self, edge_type_vocab=0, edge_type_dim=0):
        super().__init__()
        self.edge_type_vocab = edge_type_vocab
        self.edge_type_dim = edge_type_dim
        self._active = edge_type_vocab > 0 and edge_type_dim > 0

        if self._active:
            self.embedding = nn.Embedding(edge_type_vocab, edge_type_dim)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    @property
    def active(self):
        return self._active

    def forward(self, edge_types):
        """
        Args:
            edge_types: LongTensor of shape (E,) with edge type indices,
                        or None if no edge types.
        Returns:
            Tensor of shape (E, edge_type_dim) or None if inactive.
        """
        if not self._active or edge_types is None:
            return None
        return self.embedding(edge_types)
