import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import pearsonr

# ==========================================
# 1. DATA PARSING & UNROLLING
# ==========================================
def split_train_val_by_group(jsonl_file, val_ratio=0.15):
    """Reads JSONL and splits by text_id to prevent data leakage."""
    data = []
    groups = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
            groups.append(obj['text_id'])
            
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(gss.split(data, groups=groups))
    
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    return train_data, val_data

class UnrolledRepresentationDataset(Dataset):
    def __init__(self, data_source):
        self.X_unrolled = []
        self.Y_unrolled = []
        self.text_ids = []
        
        # For aggregated metrics (Correlation)
        self.X_aggregated = []
        
        # Accept either a file path (for test set) or a list of dicts (for train/val splits)
        if isinstance(data_source, str):
            with open(data_source, 'r') as f:
                data_list = [json.loads(line) for line in f]
        else:
            data_list = data_source
            
        for data in data_list:
            rep = data['last_token_residual_stream']
            raw_guesses = data['raw_guesses_numeric']
            soft_label = data['probability_label_1']
            text_id = data['text_id']
            
            # Store aggregated data for post-evaluation
            self.X_aggregated.append(rep)
            
            # Unroll the data for training
            for guess in raw_guesses:
                self.X_unrolled.append(rep)
                self.Y_unrolled.append(guess)
                self.text_ids.append(text_id)

        self.X_unrolled = torch.tensor(self.X_unrolled, dtype=torch.float32)
        self.Y_unrolled = torch.tensor(self.Y_unrolled, dtype=torch.float32).unsqueeze(1)
        self.X_aggregated = torch.tensor(self.X_aggregated, dtype=torch.float32)

    def __len__(self):
        return len(self.X_unrolled)

    def __getitem__(self, idx):
        return self.X_unrolled[idx], self.Y_unrolled[idx]

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class ShallowMLP(nn.Module):
    def __init__(self, input_dim=5120, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
            # Note: No Sigmoid here. We use BCEWithLogitsLoss for numerical stability.
        )

    def forward(self, x):
        return self.network(x)

class DeepEnsemble:
    def __init__(self, num_models=5, input_dim=5120, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_models = num_models
        self.device = device
        self.models = [ShallowMLP(input_dim=input_dim).to(device) for _ in range(num_models)]
        self.optimizers = [optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) for model in self.models]
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_criterion = nn.BCELoss() # Used for ensemble probabilities

    def train(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            train_losses = [0.0] * self.num_models
            for model in self.models:
                model.train()
                
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                
                for i, model in enumerate(self.models):
                    self.optimizers[i].zero_grad()
                    
                    logits = model(X_batch)
                    loss = self.criterion(logits, Y_batch)
                    
                    loss.backward()
                    self.optimizers[i].step()
                    train_losses[i] += loss.item()
            
            avg_train_losses = [l / len(train_loader) for l in train_losses]
            
            # --- VALIDATION PHASE ---
            self.set_eval()
            val_losses = [0.0] * self.num_models
            ens_val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                    
                    batch_probs = []
                    for i, model in enumerate(self.models):
                        logits = model(X_batch)
                        # Track individual model validation loss
                        loss = self.criterion(logits, Y_batch)
                        val_losses[i] += loss.item()
                        
                        # Store probabilities for ensemble calculation
                        batch_probs.append(torch.sigmoid(logits))
                    
                    # Calculate aggregated ensemble validation loss
                    avg_batch_probs = torch.stack(batch_probs).mean(dim=0)
                    bce_loss = self.val_criterion(avg_batch_probs, Y_batch)
                    ens_val_loss += bce_loss.item()
                    
            avg_val_losses = [l / len(val_loader) for l in val_losses]
            avg_ens_val_loss = ens_val_loss / len(val_loader)
            
            # Print Formatted Results
            train_str = ", ".join([f"{l:.4f}" for l in avg_train_losses])
            val_str = ", ".join([f"{l:.4f}" for l in avg_val_losses])
            print(f"Epoch {epoch+1:02d}/{epochs} | Train: [{train_str}] | Val: [{val_str}] | Ens Val: {avg_ens_val_loss:.4f}")

    def predict_proba(self, X):
        self.set_eval()
        X = X.to(self.device)
        with torch.no_grad():
            ensemble_probs = [torch.sigmoid(model(X)) for model in self.models]
            avg_probs = torch.stack(ensemble_probs).mean(dim=0)
        return avg_probs.cpu().numpy()
        
    def set_eval(self):
        for model in self.models:
            model.eval()

    def save(self, filepath):
        """Saves the state_dicts of all models in the ensemble to a single file."""
        state_dicts = {
            f'model_{i}': model.state_dict() for i, model in enumerate(self.models)
        }
        torch.save(state_dicts, filepath)
        print(f"Ensemble saved successfully to {filepath}")

    def load(self, filepath):
        """Loads the state_dicts into the ensemble's models."""
        state_dicts = torch.load(filepath, map_location=self.device)
        for i, model in enumerate(self.models):
            model.load_state_dict(state_dicts[f'model_{i}'])
        print(f"Ensemble loaded successfully from {filepath}")

# ==========================================
# 3. METRICS & EVALUATION
# ==========================================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def evaluate_ensemble(ensemble, dataset):
    y_true_unrolled = dataset.Y_unrolled.numpy().flatten()
    y_prob_unrolled = ensemble.predict_proba(dataset.X_unrolled).flatten()
    
    loss = log_loss(y_true_unrolled, y_prob_unrolled)
    brier = brier_score_loss(y_true_unrolled, y_prob_unrolled)
    ece = expected_calibration_error(y_true_unrolled, y_prob_unrolled)
    
    global_pos_ratio = y_true_unrolled.mean()
    brier_baseline = brier_score_loss(y_true_unrolled, np.full_like(y_true_unrolled, global_pos_ratio))
    inherent_variance = 0.0303 
    bss = 1 - ((brier - inherent_variance) / (brier_baseline - inherent_variance))
    
    y_prob_agg = ensemble.predict_proba(dataset.X_aggregated).flatten()
    
    print("\n=== ENSEMBLE PERFORMANCE ===")
    print(f"Log Loss:             {loss:.4f}  (Target: ~0.0884)")
    print(f"Brier Score:          {brier:.4f}  (Target: ~0.0303)")
    print(f"Brier Skill Score:    {bss:.4f}  (1.0 is perfect)")
    print(f"ECE:                  {ece:.4f}  (0.0 is perfect)")

    # MAXIMUM ACCURACY:    0.9533
    # (If the model perfectly guessed the majority human vote for every text)

    # MINIMUM LOG LOSS:    0.0884
    # (If the model perfectly output the exact human agreement probability)

    # MINIMUM BRIER SCORE: 0.0303
    # (The inherent, irreducible variance of human disagreement)


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Split train file by text_id group to prevent leakage
    train_file = '/projects/p32143/RL_human_decision/baseline/LLM_decision/data/gpt_train_set.jsonl'
    train_data, val_data = split_train_val_by_group(train_file, val_ratio=0.1)
    
    # Initialize datasets using the split data lists
    train_dataset = UnrolledRepresentationDataset(train_data)
    val_dataset = UnrolledRepresentationDataset(val_data)
    test_dataset = UnrolledRepresentationDataset('/projects/p32143/RL_human_decision/baseline/LLM_decision/data/gpt_test_set.jsonl')
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize and Train Ensemble
    ensemble = DeepEnsemble(num_models=5, input_dim=5120)
    print("Training Shallow Deep Ensemble...")
    ensemble.train(train_loader, val_loader, epochs=20)
    
    # Evaluate
    evaluate_ensemble(ensemble, test_dataset)
    
    # --- ADD THIS TO SAVE THE MODELS ---
    save_path = '/projects/p32143/RL_human_decision/baseline/LLM_decision/probe_training/ensemble_models.pth'
    ensemble.save(save_path)