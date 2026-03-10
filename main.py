from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "PubChem_compound_anticancer.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# unique per run (microseconds)
RUN_DIR = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# remove these lines to allow randomness each run
# np.random.seed(42)
# torch.manual_seed(42)

print("Loading PubChem anticancer drug dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"CSV not found at {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
print("Dataset loaded with", len(df), "rows")
print("Columns:", list(df.columns))

smiles_col_candidates = ["CanonicalSMILES", "SMILES", "canonical_smiles", "smiles"]
smiles_col = next((c for c in smiles_col_candidates if c in df.columns), None)
if smiles_col is None:
    raise KeyError(f"SMILES column not found. Tried: {smiles_col_candidates}")

df = df.rename(columns={smiles_col: "smiles"}).dropna(subset=["smiles"])

def get_features(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        ]
    except Exception:
        return None

print("Extracting molecular descriptors using RDKit...")
df["features"] = df["smiles"].apply(get_features)
df = df.dropna(subset=["features"])
print("Valid molecules:", len(df))

# Placeholder target (simulated solubility)
df["solubility"] = np.random.uniform(0, 1, size=len(df))

X = torch.tensor(list(df["features"]), dtype=torch.float32)
y = torch.tensor(df["solubility"].values, dtype=torch.float32).view(-1, 1)

class SolubilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.fc(x)

model = SolubilityModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("Training neural network model...")
for epoch in range(1, 201):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

print("Training completed!")

test_feat = torch.tensor([get_features(df["smiles"].iloc[0])], dtype=torch.float32)
prediction = model(test_feat)
print("Predicted Water Solubility (sample):", prediction.item())

with torch.no_grad():
    preds = model(X).detach().numpy().flatten()

# Save predictions CSV
out_df = df[["smiles"]].copy()
out_df["actual_solubility"] = y.detach().numpy().flatten()
out_df["predicted_solubility"] = preds
preds_path = RUN_DIR / "predictions.csv"
out_df.to_csv(preds_path, index=False)
print(f"Predictions saved to: {preds_path}")

# Save model weights
model_path = RUN_DIR / "model.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Plot and save
plt.scatter(y.detach().numpy().flatten(), preds, alpha=0.6)
plt.xlabel("Actual Solubility (simulated)")
plt.ylabel("Predicted Solubility")
plt.title("Anticancer Drug Water Solubility Prediction")
plot_path = RUN_DIR / "solubility_plot.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_path}")