"""
Neural Network for Heat Transfer Prediction (from scratch with numpy)
=====================================================================
MEP311s Heat Transfer — Bonus Task, Ain Shams University

Dataset: "A Simple Heat Transfer Model" — 183,750 CFD simulations (Star-CCM+)
Source: https://www.kaggle.com/datasets/usnyccc/a-simple-heat-transfer-model

Inputs:  Plate geometry (H, L, W), heat source geometry (H2, L2),
         heat generation (heat), air velocity (v)
Outputs: Heat source temperature (heatT), heat removed by air (energy_out),
         air temperature rise (T_d)

Author: Omar Keshk
"""

import numpy as np
import pandas as pd
import json
import time
import os

np.random.seed(42)

# =============================================================================
# 1. LOAD AND PREPROCESS DATA
# =============================================================================

def load_and_preprocess(csv_path="dataset.csv"):
    """Load dataset, clean columns, select features/targets."""
    df = pd.read_csv(csv_path)

    # Clean column names (remove leading spaces and quotes)
    df.columns = [c.strip().strip('"').strip() for c in df.columns]
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Input features
    input_cols = ['H (mm)', 'L (mm)', 'W (mm)', 'H2 (mm)', 'L2 (mm)', 'heat', 'v']
    # Output targets
    output_cols = ['heatT (C)', 'energy_out', 'T_d']

    X = df[input_cols].values.astype(np.float64)
    Y = df[output_cols].values.astype(np.float64)

    print(f"\nInputs shape:  {X.shape}")
    print(f"Outputs shape: {Y.shape}")

    # Input statistics
    print("\n--- INPUT RANGES ---")
    for i, col in enumerate(input_cols):
        print(f"  {col:12s}: [{X[:, i].min():.1f}, {X[:, i].max():.1f}]  mean={X[:, i].mean():.1f}")

    print("\n--- OUTPUT RANGES ---")
    for i, col in enumerate(output_cols):
        print(f"  {col:12s}: [{Y[:, i].min():.2f}, {Y[:, i].max():.2f}]  mean={Y[:, i].mean():.2f}")

    return X, Y, input_cols, output_cols


# =============================================================================
# 2. STANDARD SCALER
# =============================================================================

class StandardScaler:
    """Standardize features: zero mean, unit variance."""
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean

    def to_dict(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}


# =============================================================================
# 3. NEURAL NETWORK (from scratch)
# =============================================================================

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)


class NeuralNetwork:
    """
    Feedforward NN with Adam optimizer, built from scratch.
    Architecture: input → 128 → 64 → 32 → output
    """

    def __init__(self, input_size, output_size, hidden_sizes=(128, 64, 32)):
        self.layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.n_layers = len(self.layer_sizes) - 1

        # He initialization
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

        # Adam state
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, X):
        """Forward pass with cached activations for backprop."""
        self.activations = [X]
        self.pre_activations = []
        a = X
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            a = relu(z) if i < self.n_layers - 1 else z
            self.activations.append(a)
        return a

    def backward(self, y_true, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """Backward pass with Adam optimizer."""
        m = y_true.shape[0]
        self.t += 1

        delta = (self.activations[-1] - y_true) * (2.0 / m)

        for i in range(self.n_layers - 1, -1, -1):
            dw = self.activations[i].T @ delta
            db = delta.sum(axis=0)

            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_deriv(self.pre_activations[i - 1])

            # Adam
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dw
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * dw**2
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * db**2

            m_w_hat = self.m_w[i] / (1 - beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - beta2**self.t)
            m_b_hat = self.m_b[i] / (1 - beta1**self.t)
            v_b_hat = self.v_b[i] / (1 - beta2**self.t)

            self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def predict(self, X):
        """Inference only (no caching)."""
        a = X
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            a = relu(z) if i < self.n_layers - 1 else z
        return a

    def export_weights(self):
        return {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'architecture': self.layer_sizes
        }


# =============================================================================
# 4. METRICS
# =============================================================================

def compute_metrics(y_true, y_pred, output_names):
    metrics = {}
    for i, name in enumerate(output_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        ss_res = np.sum((yt - yp)**2)
        ss_tot = np.sum((yt - yt.mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        rmse = np.sqrt(np.mean((yt - yp)**2))
        mae = np.mean(np.abs(yt - yp))
        mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100
        metrics[name] = {
            'R2': round(float(r2), 6),
            'RMSE': round(float(rmse), 6),
            'MAE': round(float(mae), 6),
            'MAPE': round(float(mape), 4)
        }
    return metrics


# =============================================================================
# 5. MAIN TRAINING PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print("  NEURAL NETWORK FOR HEAT TRANSFER PREDICTION")
    print("  Dataset: Kaggle — A Simple Heat Transfer Model (183,750 CFD)")
    print("=" * 65)

    # Load data
    X, Y, input_cols, output_cols = load_and_preprocess("dataset.csv")

    # 70/30 split
    n = len(X)
    n_train = int(0.7 * n)
    idx = np.random.permutation(n)
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    Y_train, Y_test = Y[idx[:n_train]], Y[idx[n_train:]]
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Normalize
    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)
    X_train_n = scaler_X.transform(X_train)
    X_test_n = scaler_X.transform(X_test)
    Y_train_n = scaler_Y.transform(Y_train)
    Y_test_n = scaler_Y.transform(Y_test)

    # Use a subsample for faster training (50k samples)
    # Full dataset is 128k training — too slow for pure numpy
    TRAIN_SIZE = 50000
    if X_train_n.shape[0] > TRAIN_SIZE:
        sub_idx = np.random.choice(X_train_n.shape[0], TRAIN_SIZE, replace=False)
        X_sub = X_train_n[sub_idx]
        Y_sub = Y_train_n[sub_idx]
        print(f"Using {TRAIN_SIZE} subsample for training (full dataset too large for numpy)")
    else:
        X_sub = X_train_n
        Y_sub = Y_train_n

    # Initialize model
    model = NeuralNetwork(
        input_size=7,
        output_size=3,
        hidden_sizes=(128, 64, 32)
    )

    # Training config
    epochs = 300
    batch_size = 512
    lr = 0.001

    history = {'train_loss': [], 'test_loss': [], 'epoch': []}

    print(f"\nTraining: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Architecture: {model.layer_sizes}")
    print("-" * 65)

    start = time.time()

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(X_sub.shape[0])
        X_shuf = X_sub[perm]
        Y_shuf = Y_sub[perm]

        # Mini-batch SGD
        epoch_loss = 0
        n_batches = 0
        for s in range(0, X_sub.shape[0], batch_size):
            e = min(s + batch_size, X_sub.shape[0])
            xb, yb = X_shuf[s:e], Y_shuf[s:e]
            pred = model.forward(xb)
            epoch_loss += float(np.mean((pred - yb)**2))
            n_batches += 1
            model.backward(yb, lr=lr)

        epoch_loss /= n_batches

        # Test loss
        test_pred = model.predict(X_test_n)
        test_loss = float(np.mean((test_pred - Y_test_n)**2))

        history['train_loss'].append(round(epoch_loss, 8))
        history['test_loss'].append(round(test_loss, 8))
        history['epoch'].append(epoch + 1)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {epoch_loss:.6f} | "
                  f"Test: {test_loss:.6f} | "
                  f"{elapsed:.1f}s")

    total_time = time.time() - start
    print(f"\nDone in {total_time:.1f}s")

    # =========================================================================
    # 6. EVALUATION
    # =========================================================================

    Y_pred_train = scaler_Y.inverse_transform(model.predict(X_train_n))
    Y_pred_test = scaler_Y.inverse_transform(model.predict(X_test_n))

    train_metrics = compute_metrics(Y_train, Y_pred_train, output_cols)
    test_metrics = compute_metrics(Y_test, Y_pred_test, output_cols)

    print("\n" + "=" * 65)
    print("TRAIN METRICS")
    print("=" * 65)
    for name, m in train_metrics.items():
        print(f"  {name:15s} | R²: {m['R2']:.4f} | RMSE: {m['RMSE']:.4f} | MAPE: {m['MAPE']:.2f}%")

    print("\n" + "=" * 65)
    print("TEST METRICS (30%)")
    print("=" * 65)
    for name, m in test_metrics.items():
        print(f"  {name:15s} | R²: {m['R2']:.4f} | RMSE: {m['RMSE']:.4f} | MAPE: {m['MAPE']:.2f}%")

    # =========================================================================
    # 7. EXPORT FOR WEB APP
    # =========================================================================

    # Sample predictions (100 test points for visualization)
    sample_idx = np.random.choice(X_test.shape[0], 100, replace=False)
    samples = {
        'inputs': X_test[sample_idx].tolist(),
        'actual': Y_test[sample_idx].tolist(),
        'predicted': Y_pred_test[sample_idx].tolist(),
    }

    export = {
        'model': model.export_weights(),
        'scaler_X': scaler_X.to_dict(),
        'scaler_Y': scaler_Y.to_dict(),
        'input_names': input_cols,
        'output_names': output_cols,
        'input_units': ['mm', 'mm', 'mm', 'mm', 'mm', 'W', 'm/s'],
        'output_units': ['°C', 'W', '°C'],
        'input_ranges': {col: [float(X[:, i].min()), float(X[:, i].max())]
                        for i, col in enumerate(input_cols)},
        'output_ranges': {col: [float(Y[:, i].min()), float(Y[:, i].max())]
                         for i, col in enumerate(output_cols)},
        'training': {
            'history': history,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'n_total': int(n),
            'n_train': int(n_train),
            'n_test': int(n - n_train),
            'n_subsample': TRAIN_SIZE if X_train_n.shape[0] > TRAIN_SIZE else int(n_train),
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'time_s': round(total_time, 2)
        },
        'sample_predictions': samples,
        'dataset_source': {
            'name': 'A Simple Heat Transfer Model',
            'url': 'https://www.kaggle.com/datasets/usnyccc/a-simple-heat-transfer-model',
            'author': 'Xue Chao',
            'cfd_software': 'Star-CCM+',
            'total_simulations': 183750,
            'description': 'CFD simulations of air flowing over a heated plate with copper heat source on iron plate'
        }
    }

    out_path = 'model_export.json'
    with open(out_path, 'w') as f:
        json.dump(export, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nExported to {out_path} ({size_kb:.0f} KB)")
    print("Ready for web app.")


if __name__ == '__main__':
    main()
