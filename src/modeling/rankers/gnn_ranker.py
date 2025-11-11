"""
Graph Neural Network (GNN) Ranker for Cyclic Graph Structure

This module implements a Graph Attention Network (GAT) architecture for ranking
quantum positions. The 39 positions are modeled as nodes in a cyclic graph (C₃₉),
where each position connects to its neighbors in the ring structure.

Key Concept:
    Graph Neural Networks operate on graph-structured data, learning to propagate
    information between connected nodes. For the C₃₉ cyclic group, positions form
    a ring where each position has 2 neighbors (predecessor and successor).

Architecture Components:
    1. Graph Attention Layer (GAT): Learns attention weights between neighbors
    2. Multi-head Graph Attention: Multiple attention mechanisms in parallel
    3. Graph Encoder: Stack of GAT layers for deep feature learning
    4. Global Pooling: Aggregates node features into graph-level representation
    5. Decoder: Predicts ranking scores for all 39 positions

Mathematical Foundation:
    Graph Attention: α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))
    Node update: h'_i = σ(Σ_j α_ij W h_j)
    where α_ij is attention from node i to neighbor j

Performance Advantages:
    - Models explicit graph structure of C₃₉ cyclic group
    - Learns which neighbor relationships are most important
    - Captures local and global patterns through message passing
    - More interpretable than fully-connected models

Cyclic Graph Structure:
    C₃₉: positions arranged in a ring
    Edges: (i, i+1 mod 39) for i = 0..38 (bidirectional)
    Each node has degree 2 (2 neighbors)

Author: BMad Dev Agent (James)
Date: 2025-10-14
Story: Epic 3, Story 3.5 - Graph-Based Ranker (GNN Architecture)
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_ranker import BaseRanker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default hyperparameters
DEFAULT_GNN_PARAMS = {
    'd_model': 128,  # Node feature dimension
    'num_heads': 4,  # Number of attention heads
    'num_gat_layers': 3,  # Number of GAT layers
    'dropout': 0.2,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'weight_decay': 1e-5,
    'negative_slope': 0.2,  # LeakyReLU negative slope for attention
    'concat_heads': True,  # Concatenate or average multi-head outputs
    'device': 'cpu',
}


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) for learning neighbor importance.

    GAT learns attention weights between connected nodes, allowing the network
    to focus on the most relevant neighbors for each node.

    Args:
        in_features: Input feature dimensionality
        out_features: Output feature dimensionality per head
        num_heads: Number of attention heads
        dropout: Dropout probability
        negative_slope: LeakyReLU negative slope
        concat: Whether to concatenate or average multi-head outputs

    Mathematical Operation:
        e_ij = LeakyReLU(a^T [W h_i || W h_j])  # attention coefficient
        α_ij = softmax_j(e_ij)                   # normalized attention
        h'_i = σ(Σ_j∈N(i) α_ij W h_j)           # aggregated features
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 1,
                 dropout: float = 0.2, negative_slope: float = 0.2, concat: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat = concat

        # Linear transformation for each head
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph attention layer.

        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes) - binary or weighted

        Returns:
            Updated node features (batch, num_nodes, out_features * num_heads) if concat
            or (batch, num_nodes, out_features) if average
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation for all heads: (batch, heads, nodes, out_features)
        h = torch.einsum('bni,hio->bhno', x, self.W)

        # Compute attention coefficients
        # For each edge (i,j), compute attention e_ij
        h_i = h.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1)  # (batch, heads, nodes, nodes, out)
        h_j = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)  # (batch, heads, nodes, nodes, out)

        # Concatenate source and target features
        h_cat = torch.cat([h_i, h_j], dim=-1)  # (batch, heads, nodes, nodes, 2*out)

        # Compute attention scores
        e = torch.einsum('bhnmo,hoi->bhnm', h_cat, self.a).squeeze(-1)  # (batch, heads, nodes, nodes)
        e = self.leakyrelu(e)

        # Mask attention to only neighbors (apply adjacency matrix)
        # adj is (nodes, nodes), expand for batch and heads
        adj_expanded = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, nodes, nodes)
        e = e.masked_fill(adj_expanded == 0, float('-inf'))

        # Softmax to get attention weights
        alpha = F.softmax(e, dim=-1)  # (batch, heads, nodes, nodes)
        alpha = self.dropout_layer(alpha)

        # Apply attention to aggregate neighbor features
        h_prime = torch.einsum('bhnm,bhmo->bhno', alpha, h)  # (batch, heads, nodes, out)

        # Concatenate or average multi-head outputs
        if self.concat:
            h_prime = h_prime.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        else:
            h_prime = h_prime.mean(dim=1)  # Average across heads

        return h_prime


class GATEncoder(nn.Module):
    """
    Graph Attention Network Encoder.

    Stacks multiple GAT layers to learn hierarchical graph representations.
    Each layer allows nodes to aggregate information from their neighbors,
    with deeper layers capturing multi-hop relationships.

    Args:
        in_features: Input node feature dimensionality
        hidden_features: Hidden layer feature dimensionality
        num_heads: Number of attention heads per layer
        num_layers: Number of GAT layers
        dropout: Dropout probability
        negative_slope: LeakyReLU negative slope
    """

    def __init__(self, in_features: int, hidden_features: int, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.2, negative_slope: float = 0.2):
        super().__init__()

        self.num_layers = num_layers

        # First layer
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features,
                hidden_features,
                num_heads=num_heads,
                dropout=dropout,
                negative_slope=negative_slope,
                concat=True
            )
        ])

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GraphAttentionLayer(
                    hidden_features * num_heads,
                    hidden_features,
                    num_heads=num_heads,
                    dropout=dropout,
                    negative_slope=negative_slope,
                    concat=True
                )
            )

        # Last layer (average heads instead of concat)
        if num_layers > 1:
            self.gat_layers.append(
                GraphAttentionLayer(
                    hidden_features * num_heads,
                    hidden_features,
                    num_heads=num_heads,
                    dropout=dropout,
                    negative_slope=negative_slope,
                    concat=False  # Average for final layer
                )
            )

        # Activation and normalization
        self.elu = nn.ELU()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_features * num_heads if i < num_layers - 1 else hidden_features)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GAT encoder.

        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Encoded node features (batch, num_nodes, hidden_features)
        """
        h = x

        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h = gat_layer(h, adj)
            h = layer_norm(h)
            if i < self.num_layers - 1:  # No activation after last layer
                h = self.elu(h)

        return h


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling for graph-level representation.

    Learns to weight nodes by importance when creating a graph-level feature vector.
    More flexible than simple mean/max pooling.

    Args:
        in_features: Node feature dimensionality
    """

    def __init__(self, in_features: int):
        super().__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply global attention pooling.

        Args:
            x: Node features (batch, num_nodes, in_features)

        Returns:
            Graph-level features (batch, in_features)
        """
        # Compute attention scores for each node
        attention_scores = self.attention_net(x)  # (batch, num_nodes, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum of node features
        graph_features = (attention_weights * x).sum(dim=1)  # (batch, in_features)

        return graph_features


class GNNModel(nn.Module):
    """
    Complete GNN architecture for ranking.

    This model operates on the C₃₉ cyclic graph structure:
    1. Node feature embedding
    2. GAT encoder (multiple attention layers)
    3. Global attention pooling
    4. Decoder (MLP to predict 39 position scores)

    Args:
        input_dim: Input feature dimensionality per node
        d_model: Hidden feature dimensionality
        num_heads: Number of attention heads
        num_gat_layers: Number of GAT layers
        output_dim: Number of positions to rank (39)
        dropout: Dropout probability
        negative_slope: LeakyReLU negative slope
    """

    def __init__(self, input_dim: int, d_model: int = 128, num_heads: int = 4,
                 num_gat_layers: int = 3, output_dim: int = 39, dropout: float = 0.2,
                 negative_slope: float = 0.2):
        super().__init__()

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # GAT Encoder
        self.encoder = GATEncoder(
            in_features=d_model,
            hidden_features=d_model,
            num_heads=num_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
            negative_slope=negative_slope
        )

        # Global pooling
        self.pooling = GlobalAttentionPooling(d_model)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN.

        Args:
            x: Node features (batch, num_nodes, input_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Position ranking scores (batch, output_dim)
        """
        # Embed node features
        x = self.input_embedding(x)  # (batch, num_nodes, d_model)

        # Encode graph with GAT
        x = self.encoder(x, adj)  # (batch, num_nodes, d_model)

        # Global pooling
        graph_features = self.pooling(x)  # (batch, d_model)

        # Decode to position scores
        scores = self.decoder(graph_features)  # (batch, output_dim)

        return scores


class GNNRanker(BaseRanker):
    """
    Graph Neural Network ranker for cyclic graph structure.

    This ranker models the 39 quantum positions as nodes in a cyclic graph (C₃₉),
    using Graph Attention Networks to learn which neighbor relationships are most
    important for predicting the next state.

    Why GNNs for C₃₉:
        - **Explicit Structure**: Models the cyclic group structure directly
        - **Interpretability**: Attention weights show which edges are important
        - **Inductive Bias**: Graph structure provides useful prior knowledge
        - **Message Passing**: Learns to propagate information along the ring

    Architecture Highlights:
        - Multi-head graph attention for learning neighbor importance
        - Multiple GAT layers for multi-hop information propagation
        - Global attention pooling for graph-level representation
        - Deep decoder for position score prediction

    Args:
        params: Dict of hyperparameters (uses defaults if None)
        track_time: Whether to track training time (for RunPod decisions per NFR1)

    Attributes:
        model_: GNNModel (set after fit())
        adj_matrix_: torch.Tensor (C₃₉ adjacency matrix)
        training_time_: float (total training time in seconds)
        epoch_times_: List[float] (per-epoch training times)
        train_losses_: List[float] (training losses per epoch)
        val_losses_: List[float] (validation losses per epoch)

    Example:
        >>> # Train with default hyperparameters
        >>> ranker = GNNRanker()
        >>> ranker.fit(X_train, y_train)
        >>> predictions = ranker.predict_top_k(X_test, k=20)

        >>> # Custom hyperparameters for RunPod H200
        >>> ranker = GNNRanker(params={
        ...     'd_model': 256,
        ...     'num_heads': 8,
        ...     'num_gat_layers': 4,
        ...     'batch_size': 64,
        ...     'device': 'cuda'
        ... })
        >>> ranker.fit(X_train, y_train)

    Notes:
        - Requires PyTorch ≥2.0.0
        - GPU recommended for larger models (set device='cuda' for RunPod H200)
        - Explicitly models C₃₉ cyclic graph structure
        - Training time tracked per epoch (NFR1 requirement)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, track_time: bool = True):
        """
        Initialize GNN ranker.

        Args:
            params: Hyperparameters (uses defaults if None)
            track_time: Whether to track training time

        Raises:
            ImportError: If PyTorch is not installed
        """
        super().__init__()

        # Check PyTorch availability
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Please install it with:\n"
                "  pip install torch>=2.0.0\n"
                "See https://pytorch.org/get-started/locally/ for installation instructions."
            )

        # Set hyperparameters (merge with defaults)
        self.params = DEFAULT_GNN_PARAMS.copy()
        if params is not None:
            self.params.update(params)

        self.track_time = track_time

        # Set device (CPU or GPU)
        self.device = self.params['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = 'cpu'

        # Create C₃₉ cyclic graph adjacency matrix
        self.adj_matrix_ = self._create_cyclic_adjacency_matrix(n=39).to(self.device)

        # Model and training artifacts (set during fit)
        self.model_: Optional[GNNModel] = None
        self.input_dim_: Optional[int] = None
        self.training_time_: Optional[float] = None
        self.epoch_times_: Optional[list] = None
        self.train_losses_: Optional[list] = None
        self.val_losses_: Optional[list] = None
        self.best_epoch_: Optional[int] = None

    def _create_cyclic_adjacency_matrix(self, n: int = 39) -> torch.Tensor:
        """
        Create adjacency matrix for C_n cyclic graph.

        Each node i is connected to (i-1) mod n and (i+1) mod n.
        Also includes self-loops for message passing.

        Args:
            n: Number of nodes in the cycle

        Returns:
            Adjacency matrix (n, n) with 1s for edges, 0s elsewhere
        """
        adj = torch.zeros((n, n))

        for i in range(n):
            # Connect to neighbors in the ring
            adj[i, (i - 1) % n] = 1  # Predecessor
            adj[i, (i + 1) % n] = 1  # Successor
            adj[i, i] = 1  # Self-loop

        return adj

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None):
        """
        Train the GNN ranker on training data.

        This method:
        1. Validates input data
        2. Extracts active positions and creates node features
        3. Creates GNN model architecture
        4. Trains the model with early stopping
        5. Tracks training time per epoch (NFR1)

        Training Process:
            - Data is split into train/val (90/10) for early stopping
            - Node features represent which positions are active
            - Model learns to predict which positions are active via graph structure
            - Binary cross-entropy loss for multi-label classification
            - Early stopping prevents overfitting

        Args:
            X_train: Training features from imputed data
                Shape: (n_events, n_features)
                Expected: Imputed data from Epic 2 with position columns
            y_train: Not used (labels extracted from X_train)
                Kept for consistency with BaseRanker interface

        Returns:
            self: Returns self for method chaining

        Raises:
            ValueError: If X_train is empty or has wrong format
            RuntimeError: If training fails

        Notes:
            - Training time is logged per epoch if track_time=True
            - If total training exceeds 1 hour, suggests using RunPod (per NFR1)
        """
        # Validate input
        if X_train.empty:
            raise ValueError("X_train cannot be empty")

        if len(X_train) < 20:
            raise ValueError(
                f"X_train too small: {len(X_train)} samples. "
                "Need at least 20 samples for meaningful training."
            )

        logger.info(f"Training GNNRanker on {len(X_train)} events...")

        # Start timing (if enabled)
        start_time = time.time() if self.track_time else None

        try:
            # Step 1: Extract active positions and prepare graph data
            X_tensor, y_tensor = self._prepare_training_data(X_train)

            # Step 2: Create train/val split for early stopping
            X_train_split, y_train_split, X_val, y_val = self._create_val_split(
                X_tensor, y_tensor
            )

            # Step 3: Create data loaders
            train_loader = self._create_dataloader(X_train_split, y_train_split, shuffle=True)
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

            # Step 4: Initialize model
            self.input_dim_ = X_tensor.shape[2]  # Feature dimension per node
            self.model_ = GNNModel(
                input_dim=self.input_dim_,
                d_model=self.params['d_model'],
                num_heads=self.params['num_heads'],
                num_gat_layers=self.params['num_gat_layers'],
                output_dim=39,
                dropout=self.params['dropout'],
                negative_slope=self.params['negative_slope']
            ).to(self.device)

            logger.info(f"GNN architecture: d_model={self.params['d_model']}, "
                       f"num_heads={self.params['num_heads']}, "
                       f"num_gat_layers={self.params['num_gat_layers']}")
            logger.info(f"Device: {self.device}")
            logger.info(f"C₃₉ cyclic graph: 39 nodes, each with 2 neighbors + self-loop")

            # Step 5: Train model
            self._train_model(train_loader, val_loader)

            # Step 6: Track training time
            if self.track_time:
                self.training_time_ = time.time() - start_time
                logger.info(f"✓ Training completed in {self.training_time_:.2f} seconds")

                # Warn if training exceeded 1 hour (RunPod suggestion per NFR1)
                if self.training_time_ > 3600:
                    logger.warning(
                        f"⚠️ Training exceeded 1 hour ({self.training_time_/3600:.2f} hours). "
                        "Consider using RunPod H200 for Epic 5 experiments (see docs/architecture.md section 5)."
                    )

            self.is_fitted_ = True
            logger.info("✓ GNNRanker training successful")

            return self

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(
                f"GNNRanker training failed: {e}. "
                "Check data format, hyperparameters, and available memory."
            ) from e

    def predict_top_k(self, X_test: pd.DataFrame, k: int = 20) -> np.ndarray:
        """
        Predict top-k ranked positions for each test event.

        For each test event:
        1. Prepare node features (all 39 nodes with active/inactive status)
        2. Forward pass through GNN model with C₃₉ graph structure
        3. Rank all 39 positions by predicted scores
        4. Return top-k positions

        Args:
            X_test: Test features (same format as X_train)
                Shape: (n_events, n_features)
            k: Number of top predictions to return (default: 20)
                Must be in range [1, 39]

        Returns:
            predictions: Top-k ranked position predictions
                Shape: (n_events, k)
                Each row contains k position indices (1-39) ranked by likelihood

        Raises:
            RuntimeError: If called before fit()
            ValueError: If k not in range [1, 39]
            ValueError: If X_test has wrong format

        Example:
            >>> predictions = ranker.predict_top_k(X_test, k=20)
            >>> predictions.shape
            (1000, 20)  # 1000 test events, 20 predictions each
            >>> predictions[0]  # First event's top-20 predictions
            array([ 5, 12, 33, 7, 20, ...])
        """
        # Validation
        self._check_is_fitted()
        self._validate_k(k)

        if X_test.empty:
            raise ValueError("X_test cannot be empty")

        n_events = len(X_test)
        logger.info(f"Predicting top-{k} for {n_events} test events...")

        # Prepare test data
        X_test_tensor, _ = self._prepare_training_data(X_test)
        X_test_tensor = X_test_tensor.to(self.device)

        # Make predictions
        self.model_.eval()
        with torch.no_grad():
            scores = self.model_(X_test_tensor, self.adj_matrix_)  # (n_events, 39)
            scores = scores.cpu().numpy()

        # Rank positions by score (descending)
        ranked_indices = np.argsort(-scores, axis=1)

        # Convert 0-based indices to 1-based position numbers and take top-k
        predictions = ranked_indices[:, :k] + 1

        logger.info(f"✓ Predictions generated for {n_events} events")

        return predictions

    def _prepare_training_data(self, X: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data in GNN format (DATA LEAKAGE FIX).

        Creates node features for all 39 positions in the graph. Each node combines:
        - Binary active status (0 or 1)
        - One-hot position identity (39-dim)
        - Imputed features (learned patterns from Epic 2)

        CRITICAL: Does NOT use QV columns (they contain the answer!).
        Targets are extracted from q_1...q_5 columns.

        Returns:
            X_tensor: Node features (n_samples, 39 nodes, input_dim)
            y_tensor: Target labels (n_samples, 39)
        """
        # Extract active positions from target columns (DATA LEAKAGE FIX)
        active_positions_list = self._extract_active_positions(X)

        # DATA LEAKAGE FIX: Get imputed feature columns (exclude targets and metadata only)
        # CRITICAL: QV columns are NOT in the data and should NOT be referenced
        # Only exclude target columns (q_*) and metadata (event-ID)
        position_cols = [f'q_{i}' for i in range(1, 6)]
        exclude_cols = set(position_cols + ['event-ID'])
        imputed_cols = [col for col in X.columns if col not in exclude_cols]

        n_samples = len(active_positions_list)

        X_list = []
        y_list = []

        for idx, (_, row) in enumerate(X.iterrows()):
            if idx >= len(active_positions_list):
                break

            positions = active_positions_list[idx]

            # PATH A FIX: Extract imputed features for this event
            if len(imputed_cols) > 0:
                imputed_features = row[imputed_cols].values.astype(np.float32)
            else:
                imputed_features = np.array([], dtype=np.float32)

            # Create node features for all 39 nodes
            node_features = []

            for node_id in range(1, 40):  # Nodes 1-39
                # Binary feature: is this node active?
                is_active = 1.0 if node_id in positions else 0.0

                # One-hot encoding of node ID (position in ring)
                node_onehot = np.zeros(39, dtype=np.float32)
                node_onehot[node_id - 1] = 1.0

                # PATH A FIX: Combine is_active, node_onehot, AND imputed features
                # This gives the GNN BOTH graph structure AND learned patterns
                if len(imputed_features) > 0:
                    features = np.concatenate([[is_active], node_onehot, imputed_features])
                else:
                    features = np.concatenate([[is_active], node_onehot])

                node_features.append(features)

            X_list.append(np.array(node_features))

            # Create target: binary labels for all 39 positions
            y_binary = np.zeros(39, dtype=np.float32)
            for pos in positions:
                if 1 <= pos <= 39:
                    y_binary[pos - 1] = 1.0
            y_list.append(y_binary)

        # Convert to tensors
        X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)

        return X_tensor, y_tensor

    def _extract_active_positions(self, X: pd.DataFrame) -> List[List[int]]:
        """
        Extract active position indices from target columns (DATA LEAKAGE FIX).

        Uses q_1...q_5 columns which contain the 5 winning positions for each event.
        This replaces the previous approach that used QV columns (which contained
        the answer and caused data leakage).

        Args:
            X: Feature DataFrame with q_1, q_2, q_3, q_4, q_5 columns

        Returns:
            List of lists, each containing active position indices (1-39) for one event
            Example: [[8, 21, 22, 28, 38], [19, 29, 35, 37, 39], ...]

        Raises:
            ValueError: If target columns are missing

        Notes:
            - Target columns must contain winning positions (not binary indicators)
            - Position indices are 1-based (1 to 39)
            - DATA LEAKAGE FIX: Uses explicit targets, not QV feature columns
        """
        # Use explicit target columns (q_1 through q_5)
        target_cols = ['q_1', 'q_2', 'q_3', 'q_4', 'q_5']

        # Verify target columns exist
        missing_cols = [col for col in target_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Missing target columns in data. Expected q_1 through q_5. "
                f"Missing: {missing_cols}. "
                f"These columns should contain the 5 winning positions per event."
            )

        active_positions_list = []

        # Extract winning positions from target columns
        for _, row in X[target_cols].iterrows():
            # q_* columns contain position numbers (1-39)
            positions = [int(row[col]) for col in target_cols]
            active_positions_list.append(positions)

        return active_positions_list

    def _create_val_split(self, X: torch.Tensor, y: torch.Tensor,
                          val_ratio: float = 0.1) -> Tuple[torch.Tensor, ...]:
        """Create train/validation split for early stopping."""
        n_samples = X.shape[0]
        n_val = int(n_samples * val_ratio)
        n_train = n_samples - n_val

        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:]
        y_val = y[n_train:]

        logger.info(f"Train/val split: {n_train}/{n_val} samples")

        return X_train, y_train, X_val, y_val

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor,
                           shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader for batching."""
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.params['batch_size'],
            shuffle=shuffle
        )
        return loader

    def _train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train GNN model with early stopping."""
        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay']
        )

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.params['early_stopping_patience']

        # Training history
        self.train_losses_ = []
        self.val_losses_ = []
        self.epoch_times_ = []

        logger.info(f"Starting training for up to {self.params['epochs']} epochs...")

        for epoch in range(self.params['epochs']):
            epoch_start = time.time() if self.track_time else None

            # Training phase
            self.model_.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model_(X_batch, self.adj_matrix_)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)
            self.train_losses_.append(train_loss)

            # Validation phase
            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model_(X_batch, self.adj_matrix_)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)
            self.val_losses_.append(val_loss)

            # Track epoch time
            if self.track_time:
                epoch_time = time.time() - epoch_start
                self.epoch_times_.append(epoch_time)

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{self.params['epochs']}: "
                           f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_epoch_ = epoch + 1
                patience_counter = 0
                self.best_model_state_ = self.model_.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {self.best_epoch_}")
                break

        # Restore best model
        self.model_.load_state_dict(self.best_model_state_)
        logger.info(f"✓ Training completed. Best epoch: {self.best_epoch_}")

    def save_model(self, model_path: Path, save_metadata: bool = True):
        """Save trained model to disk."""
        self._check_is_fitted()

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {model_path}")
        torch.save({
            'model_state_dict': self.model_.state_dict(),
            'params': self.params,
            'input_dim': self.input_dim_
        }, model_path)
        logger.info(f"✓ Model saved successfully")

        if save_metadata:
            metadata_path = Path(str(model_path) + '.meta.json')
            metadata = {
                'ranker_type': 'gnn',
                'pytorch_version': torch.__version__,
                'hyperparameters': self.params,
                'training_time_seconds': self.training_time_,
                'total_epochs_trained': len(self.train_losses_),
                'best_epoch': self.best_epoch_,
                'best_val_loss': self.val_losses_[self.best_epoch_ - 1] if self.best_epoch_ else None,
                'final_train_loss': self.train_losses_[-1] if self.train_losses_ else None,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            logger.info(f"Saving metadata to {metadata_path}")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Metadata saved successfully")

    @classmethod
    def load_model(cls, model_path: Path) -> 'GNNRanker':
        """Load trained model from disk."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please train and save model first using save_model()."
            )

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')

        instance = cls(params=checkpoint['params'])
        instance.input_dim_ = checkpoint['input_dim']
        instance.model_ = GNNModel(
            input_dim=instance.input_dim_,
            d_model=instance.params['d_model'],
            num_heads=instance.params['num_heads'],
            num_gat_layers=instance.params['num_gat_layers'],
            output_dim=39,
            dropout=instance.params['dropout'],
            negative_slope=instance.params['negative_slope']
        )

        instance.model_.load_state_dict(checkpoint['model_state_dict'])
        instance.model_.to(instance.device)
        instance.is_fitted_ = True

        # Try to load metadata
        metadata_path = Path(str(model_path) + '.meta.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                instance.training_time_ = metadata.get('training_time_seconds')
                instance.best_epoch_ = metadata.get('best_epoch')
                logger.info("✓ Metadata loaded")

        logger.info(f"✓ Model loaded successfully")
        return instance

    def __repr__(self) -> str:
        """String representation with training status and time."""
        fitted_status = "fitted" if self.is_fitted_ else "not fitted"
        if self.training_time_ is not None:
            return f"GNNRanker({fitted_status}, trained in {self.training_time_:.2f}s, best_epoch={self.best_epoch_})"
        else:
            return f"GNNRanker({fitted_status})"
