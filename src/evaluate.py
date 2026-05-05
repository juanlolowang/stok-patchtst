"""
evaluate.py
Evaluasi model PatchTST pada test set: MAE, RMSE, MAPE + plot prediksi vs aktual.
"""

import os
import sys
import json
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import PatchTST, VanillaLSTM

BASE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE, "..", "data", "processed")
MODELS_DIR  = os.path.join(BASE, "..", "models")
OUT_DIR     = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(model_path: str, model_type: str, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    
    if model_type == "patchtst":
        model = PatchTST(
            seq_len    = cfg["seq_len"],
            pred_len   = cfg["pred_len"],
            patch_len  = cfg["patch_len"],
            stride     = cfg["stride"],
            d_model    = cfg["d_model"],
            n_heads    = cfg["n_heads"],
            n_layers   = cfg["n_layers"],
            d_ff       = cfg["d_ff"],
            dropout    = cfg["dropout"],
            n_channels = cfg["n_channels"],
        ).to(device)
    else:
        model = VanillaLSTM(
            seq_len    = cfg["seq_len"],
            pred_len   = cfg["pred_len"],
            hidden_dim = cfg["d_model"],
            n_layers   = cfg["n_layers"],
            dropout    = cfg["dropout"],
            n_channels = cfg["n_channels"],
        ).to(device)
        
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_files = {
        "PatchTST": os.path.join(MODELS_DIR, "best_model_patchtst.pt"),
        "LSTM": os.path.join(MODELS_DIR, "best_model_lstm.pt")
    }

    # Load test data (normalized)
    X_test_path = os.path.join(DATA_DIR, "X_test.npy")
    y_test_path = os.path.join(DATA_DIR, "y_test.npy")
    
    if not os.path.exists(X_test_path):
        print("❌ Data test tidak ditemukan.")
        return

    X_test = torch.from_numpy(np.load(X_test_path)).to(device)
    y_test = np.load(y_test_path)

    all_metrics = {}
    
    print(f"\n{'='*60}")
    print(f"  {'Model':<15} | {'MAE':<12} | {'MSE':<12}")
    print(f"{'-'*60}")

    for name, path in model_files.items():
        if os.path.exists(path):
            m_type = "patchtst" if "patchtst" in os.path.basename(path) else "lstm"
            model, cfg = load_model(path, m_type, device)
            
            with torch.no_grad():
                y_pred_norm = model(X_test).cpu().numpy()
            
            mae_v = np.mean(np.abs(y_test - y_pred_norm))
            mse_v = np.mean((y_test - y_pred_norm) ** 2)
            
            all_metrics[name] = {"MAE": float(mae_v), "MSE": float(mse_v)}
            print(f"  {name:<15} | {mae_v:.6f}     | {mse_v:.6f}")
        else:
            print(f"  {name:<15} | (Belum ditraining)")

    print(f"{'='*60}")

    # Simpan perbandingan
    if all_metrics:
        with open(os.path.join(OUT_DIR, "comparison_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
            
        # Plot Perbandingan
        plt.style.use("dark_background")
        names = list(all_metrics.keys())
        maes = [all_metrics[n]["MAE"] for n in names]
        mses = [all_metrics[n]["MSE"] for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, maes, width, label='MAE', color='#7c5cfc')
        rects2 = ax.bar(x + width/2, mses, width, label='MSE', color='#ff8a65')
        
        ax.set_ylabel('Error Score')
        ax.set_title('Perbandingan Performa: PatchTST vs LSTM')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "comparison.png"), dpi=150)
        plt.close()
        print(f"✅ Plot perbandingan disimpan ke outputs/comparison.png")

    return all_metrics


if __name__ == "__main__":
    evaluate()
