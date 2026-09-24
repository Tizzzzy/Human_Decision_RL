"""
Probe model definitions: ShallowMLP and DeepEnsemble.

Duplicated locally (not cross-imported from simulator.probe_training)
to avoid fragile cross-directory imports and keep the RL module self-contained.

ShallowMLP: Single model, Linear(5120 -> hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim -> 1)
DeepEnsemble: 5-model ensemble of ShallowMLP, averaging predictions for robustness.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List


class ShallowMLP(nn.Module):
    """
    Simple MLP probe: 5120-dim input (Qwen hidden state) -> hidden_dim -> 1-dim logit output.
    """

    def __init__(self, input_dim: int = 5120, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            # Note: No sigmoid here. Apply sigmoid(output) to get P(AI) in [0,1].
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (..., input_dim) e.g. [batch_size, 5120]

        Returns:
            Tensor of shape (..., 1) e.g. [batch_size, 1] (logits, not probabilities)
        """
        return self.network(x)


class DeepEnsemble:
    """
    5-model ensemble of ShallowMLP for robust human-simulator predictions.

    Averages probabilities across all 5 models to get ensemble prediction.
    Can be loaded from a checkpoint file saved during training.
    """

    def __init__(self, num_models: int = 5, input_dim: int = 5120, device: str = None):
        """
        Initialize the ensemble.

        Args:
            num_models: Number of models in the ensemble (typically 5)
            input_dim: Input dimension (5120 for Qwen hidden states)
            device: Device to load models on ('cuda' or 'cpu'). Defaults to CUDA if available.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_models = num_models
        self.device = device
        self.models: List[ShallowMLP] = [
            ShallowMLP(input_dim=input_dim).to(device)
            for _ in range(num_models)
        ]

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """
        Get ensemble predictions (average across all models).

        Args:
            X: Input tensor of shape [batch_size, 5120]

        Returns:
            numpy array of shape [batch_size, 1] with P(AI) in [0, 1]
        """
        self.set_eval()
        X = X.to(self.device)

        with torch.no_grad():
            ensemble_probs = [torch.sigmoid(model(X)) for model in self.models]
            avg_probs = torch.stack(ensemble_probs).mean(dim=0)

        return avg_probs.cpu().numpy()

    def set_eval(self):
        """Set all models to eval mode."""
        for model in self.models:
            model.eval()

    def save(self, filepath: str):
        """
        Save the state_dicts of all models in the ensemble.

        Args:
            filepath: Path to save the checkpoint to
        """
        state_dicts = {
            f'model_{i}': model.state_dict() for i, model in enumerate(self.models)
        }
        torch.save(state_dicts, filepath)
        print(f"Ensemble saved successfully to {filepath}")

    def load(self, filepath: str):
        """
        Load the state_dicts from a checkpoint.

        Args:
            filepath: Path to load the checkpoint from
        """
        state_dicts = torch.load(filepath, map_location=self.device)
        for i, model in enumerate(self.models):
            model.load_state_dict(state_dicts[f'model_{i}'])
        print(f"Ensemble loaded successfully from {filepath}")
