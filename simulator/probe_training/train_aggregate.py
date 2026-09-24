import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, mean_squared_error
from scipy.stats import pearsonr

# ==========================================
# 1. AGGREGATED DATASET LOADER
# ==========================================
class AggregatedRepresentationDataset(Dataset):
    def __init__(self, jsonl_file):
        self.X = []
        self.Y_soft = []
        self.raw_guesses = []
        self.text_ids = []
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.X.append(data['last_token_residual_stream'])  #
                self.Y_soft.append(data['probability_label_1'])     #[cite: 1]
                self.raw_guesses.append(data['raw_guesses_numeric'])  #[cite: 1]
                self.text_ids.append(data['text_id'])  #[cite: 1]

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y_soft = torch.tensor(self.Y_soft, dtype=torch.float32).unsqueeze(1)
        self.Y_soft_numpy = np.array(self.Y_soft.squeeze(1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_soft[idx]

    def get_unrolled_eval_data(self):
        """Unrolls raw binary decisions for noise-ceiling evaluation."""
        X_unrolled = []
        Y_unrolled = []
        for i, guesses in enumerate(self.raw_guesses):
            rep = self.X[i].numpy()
            for g in guesses:
                X_unrolled.append(rep)
                Y_unrolled.append(g)
        return torch.tensor(X_unrolled, dtype=torch.float32), np.array(Y_unrolled)

# ==========================================
# 2. SHALLOW ENSEMBLE ARCHITECTURE
# ==========================================
class ShallowMLP(nn.Module):
    def __init__(self, input_dim=5120, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

class SoftLabelDeepEnsemble:
    def __init__(self, num_models=5, input_dim=5120, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_models = num_models
        self.device = device
        self.models = [ShallowMLP(input_dim=input_dim).to(device) for _ in range(num_models)]
        self.optimizers = [
            optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) 
            for model in self.models
        ]
        # PyTorch BCEWithLogitsLoss natively handles continuous targets in [0, 1]
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, dataloader, epochs=15):
        for epoch in range(epochs):
            losses = [0.0] * self.num_models
            for X_batch, Y_soft_batch in dataloader:
                X_batch = X_batch.to(self.device)
                Y_soft_batch = Y_soft_batch.to(self.device)
                
                for i, model in enumerate(self.models):
                    model.train()
                    self.optimizers[i].zero_grad()
                    
                    logits = model(X_batch)
                    loss = self.criterion(logits, Y_soft_batch)
                    
                    loss.backward()
                    self.optimizers[i].step()
                    losses[i] += loss.item()
            
            avg_losses = [round(l / len(dataloader), 4) for l in losses]
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Soft-Label BCE Losses: {avg_losses}")

    def predict_proba(self, X_tensor):
        self.set_eval()
        X_tensor = X_tensor.to(self.device)
        with torch.no_grad():
            ensemble_probs = [torch.sigmoid(model(X_tensor)) for model in self.models]
            avg_probs = torch.stack(ensemble_probs).mean(dim=0)
        return avg_probs.cpu().numpy().flatten()

    def set_eval(self):
        for model in self.models:
            model.eval()

# ==========================================
# 3. METRICS & CALIBRATION EVALUATION
# ==========================================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def evaluate_soft_ensemble(ensemble, dataset):
    # ----------------------------------------------------
    # Metric Group A: Aggregated Soft Target Evaluation
    # ----------------------------------------------------
    y_pred_soft = ensemble.predict_proba(dataset.X)
    y_true_soft = dataset.Y_soft_numpy
    
    soft_mse = mean_squared_error(y_true_soft, y_pred_soft)
    # Clip probabilities slightly to avoid log(0)
    clipped_pred = np.clip(y_pred_soft, 1e-7, 1 - 1e-7)
    soft_bce = -(y_true_soft * np.log(clipped_pred) + (1 - y_true_soft) * np.log(1 - clipped_pred)).mean()
    pearson_corr, _ = pearsonr(y_pred_soft, y_true_soft)

    # ----------------------------------------------------
    # Metric Group B: Raw Guess Evaluation (Noise Ceilings)
    # ----------------------------------------------------
    X_unrolled, y_true_unrolled = dataset.get_unrolled_eval_data()
    y_pred_unrolled = ensemble.predict_proba(X_unrolled)
    
    unrolled_log_loss = log_loss(y_true_unrolled, y_pred_unrolled)
    unrolled_brier = brier_score_loss(y_true_unrolled, y_pred_unrolled)
    ece = expected_calibration_error(y_true_unrolled, y_pred_unrolled)
    
    global_prior = y_true_unrolled.mean()
    baseline_brier = brier_score_loss(y_true_unrolled, np.full_like(y_true_unrolled, global_prior))
    inherent_variance = 0.1713  # Target noise floor
    bss = 1.0 - ((unrolled_brier - inherent_variance) / (baseline_brier - inherent_variance))

    print("\n" + "="*50)
    print("      SOFT-LABEL ENSEMBLE EVALUATION REPORT      ")
    print("="*50)
    print("1. Aggregated Target Metrics (probability_label_1):")
    print(f"   - Soft Target BCE Loss:  {soft_bce:.4f}")
    print(f"   - Soft Target MSE:       {soft_mse:.4f}")
    print(f"   - Pearson Correlation:   {pearson_corr:.4f}")
    print("\n2. Benchmark Noise-Ceiling Metrics (raw_guesses_numeric):")
    print(f"   - Log Loss:              {unrolled_log_loss:.4f}  (Ceiling Minimum: 0.4932)")
    print(f"   - Brier Score:           {unrolled_brier:.4f}  (Ceiling Minimum: 0.1713)")
    print(f"   - Brier Skill Score:     {bss:.4f}  (1.0 is optimal)")
    print(f"   - Calibration Error ECE: {ece:.4f}  (0.0 is optimal)")
    print("="*50)

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    train_dataset = AggregatedRepresentationDataset('/projects/p32143/RL_human_decision/simulator/data/train_set.jsonl')
    test_dataset = AggregatedRepresentationDataset('/projects/p32143/RL_human_decision/simulator/data/test_set.jsonl')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    ensemble = SoftLabelDeepEnsemble(num_models=5, input_dim=5120)
    
    print("Training Shallow Deep Ensemble on probability_label_1...")
    ensemble.train(train_loader, epochs=15)
    
    evaluate_soft_ensemble(ensemble, test_dataset)