"""
This script was largely inspired by the following source:https://github.com/ronantakizawa/kanomaly/blob/main/kananomaly.ipynb
and is only used as a baseline for the evaluation and is not submitted as a final.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from anomaly_detection import AnomalyDetection


# Set random seeds for reproducibility
class Args:
    def __init__(self):
        self.device = None

    path = "./data/"
    dropout = 0.3
    hidden_size = 128
    grid_size = 50
    n_layers = 2
    epochs = 200
    early_stopping = 30  # Increased patience
    seed = 42
    lr = 1e-3  # Increased learning rate
    window_size = 20
    step_size = 10
    batch_size = 32
    anomaly_fraction = 0.1


# Define the custom dataset with overlapping windows
class TimeSeriesAnomalyDataset(torch.utils.data.Dataset):
    def __init__(
            self, time_series, labels, window_size=20, step_size=10, transform=None
    ):
        self.time_series = time_series
        self.labels = labels
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.sample_indices = list(
            range(0, len(time_series) - window_size + 1, step_size)
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        if idx >= len(self.sample_indices) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for sample_indices of length {len(self.sample_indices)}"
            )
        i = self.sample_indices[idx]
        window = self.time_series[i: i + self.window_size]
        window_labels = self.labels[i: i + self.window_size]

        # Input features: window values
        x = torch.tensor(window, dtype=torch.float).unsqueeze(-1)  # Shape: [window_size, 1]

        # Label: 1 if any point in the window is an anomaly, else 0
        y = torch.tensor(1.0 if window_labels.any() else 0.0, dtype=torch.float)

        return x, y

    def indices(self):
        return self.sample_indices


# Create a new dataset with the resampled data
class ResampledDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = [torch.tensor(x, dtype=torch.float).view(-1, 1) for x in X]
        self.y = [torch.tensor(label, dtype=torch.float) for label in y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Implement Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss.sum()


# Corrected NaiveFourierKANLayer class
class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=50, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(
            torch.randn(2 * gridsize, inputdim, outdim)
            / (np.sqrt(inputdim) * np.sqrt(gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(outdim))

    def forward(self, x):
        # x shape: [batch_size, window_size, inputdim]
        batch_size, window_size, inputdim = x.size()
        k = torch.arange(1, self.gridsize + 1, device=x.device).float()
        k = k.view(1, 1, 1, self.gridsize)
        x_expanded = x.unsqueeze(-1)  # [batch_size, window_size, inputdim, 1]
        angles = x_expanded * k * np.pi  # [batch_size, window_size, inputdim, gridsize]
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        features = torch.cat([sin_features, cos_features], dim=-1)  # Concatenate on gridsize dimension
        features = features.view(batch_size * window_size, inputdim, -1)  # Flatten for matmul
        coeffs = self.fouriercoeffs  # [2 * gridsize, inputdim, outdim]
        y = torch.einsum('bik,kio->bo', features, coeffs)
        y = y.view(batch_size, window_size, self.outdim)
        if self.addbias:
            y += self.bias
        return y


# Define the KAN model
class KAN(nn.Module):
    def __init__(
            self,
            in_feat,
            hidden_feat,
            out_feat,
            grid_feat,
            num_layers,
            use_bias=True,
            dropout=0.3,
    ):
        super(KAN, self).__init__()
        self.num_layers = num_layers
        self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.bn_in = nn.BatchNorm1d(hidden_feat)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                NaiveFourierKANLayer(
                    hidden_feat, hidden_feat, grid_feat, addbias=use_bias
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_feat))
        self.lin_out = nn.Linear(hidden_feat, out_feat, bias=use_bias)

    def forward(self, x):
        # x shape: [batch_size, window_size, 1]
        batch_size, window_size, _ = x.size()
        x = self.lin_in(x)  # [batch_size, window_size, hidden_feat]
        x = self.bn_in(x.view(-1, x.size(-1))).view(batch_size, window_size, -1)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x.view(-1, x.size(-1))).view(batch_size, window_size, -1)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.dropout(x)
        # Global average pooling over the window dimension
        x = x.mean(dim=1)  # [batch_size, hidden_feat]
        x = self.lin_out(x).squeeze()  # [batch_size]
        return x


class KANBaseline:
    def __init__(self):
        self.setup()
    # Stratified Splitting
    def stratified_split(self,
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    ):
        # Extract labels
        labels = [y.item() for _, y in dataset]
    
        # Split into train+val and test
        train_val_indices, test_indices = train_test_split(
            np.arange(len(labels)),
            test_size=test_ratio,
            stratify=labels,
            random_state=seed,
        )
    
        # Calculate validation size relative to train_val
        val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    
        # Split train_val into train and val
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_relative_ratio,
            stratify=[labels[i] for i in train_val_indices],
            random_state=seed,
        )
    
        return train_indices, val_indices, test_indices
    
    
    # Set up the arguments
    def setup(self=None):
        self.args = Args()
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
    
    
    # Define evaluation metrics
    def evaluate_metrics(self, true_labels, pred_labels, pred_probs):
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        roc_auc_val = roc_auc_score(true_labels, pred_probs)
        return precision, recall, f1, roc_auc_val
    
    
    # Function to determine optimal threshold based on validation set
    def find_optimal_threshold(self, probs, labels):
        precision_vals, recall_vals, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        if optimal_idx < len(thresholds):
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 0.5  # Default threshold
        optimal_f1 = f1_scores[optimal_idx]
        return optimal_threshold, optimal_f1
    
    
    # Training loop
    def train_model(self, scheduler, train_loader, train_set, val_set, val_loader, model, criterion, optimizer):
        # Training and validation loop with early stopping
        best_val_f1 = 0
        patience = self.args.early_stopping
        patience_counter = 0
        optimal_threshold = 0.5  # Initialize with default threshold
    
        for epoch in range(self.args.epochs):
            # Training Phase
            model.train()
            total_loss = 0
            total_acc = 0
            total_preds_pos = 0  # Monitor number of positive predictions
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                optimizer.zero_grad()
                out = model(x_batch)  # Output shape: [batch_size]
                loss = criterion(out, y_batch)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).float()
                acc = (preds == y_batch).float().mean().item()
                total_acc += acc * x_batch.size(0)
                total_preds_pos += preds.sum().item()
            avg_loss = total_loss / len(train_set)
            avg_acc = total_acc / len(train_set)
    
            # print(f"Epoch {epoch + 1}, Training Positive Predictions: {total_preds_pos}")
    
            # Validation Phase
            model.eval()
            val_loss = 0
            val_acc = 0
            all_true = []
            all_preds = []
            all_probs = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.args.device)
                    y_batch = y_batch.to(self.args.device)
                    out = model(x_batch)
                    loss = criterion(out, y_batch)
                    val_loss += loss.item() * x_batch.size(0)
                    probs = torch.sigmoid(out)
                    preds = (probs > 0.5).float()
                    acc = (preds == y_batch).float().mean().item()
                    val_acc += acc * x_batch.size(0)
                    all_true.extend(y_batch.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            avg_val_loss = val_loss / len(val_set)
            avg_val_acc = val_acc / len(val_set)
            precision, recall, f1, roc_auc_val = self.evaluate_metrics(all_true, all_preds, all_probs)
    
            # Find Optimal Threshold
            current_threshold, current_f1 = self.find_optimal_threshold(all_probs, all_true)
    
            # Step the scheduler
            scheduler.step(avg_val_loss)
    
            # Early Stopping
            if f1 > best_val_f1:
                best_val_f1 = f1
                patience_counter = 0
                optimal_threshold = current_threshold  # Update optimal threshold
                # Save the best model
                torch.save(model.state_dict(), "best_kan_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
    
        return optimal_threshold, best_val_f1
    
    
    # Test the model
    def run_model(self, criterion, test_loader, optimal_threshold, test_set, model, threshold=0.5):
        # Load the best model
        model.load_state_dict(torch.load("best_kan_model.pth", weights_only=True))
    
        # Test the model using the optimal threshold
        model.eval()
        test_loss = 0
        test_acc = 0
        all_true_test = []
        all_preds_test = []
        all_probs_test = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                out = model(x_batch)
                loss = criterion(out, y_batch)
                test_loss += loss.item() * x_batch.size(0)
                probs = torch.sigmoid(out)
                preds = (probs > optimal_threshold).float()
                acc = (preds == y_batch).float().mean().item()
                test_acc += acc * x_batch.size(0)
                all_true_test.extend(y_batch.cpu().numpy())
                all_preds_test.extend(preds.cpu().numpy())
                all_probs_test.extend(probs.cpu().numpy())
        avg_test_loss = test_loss / len(test_set)
        avg_test_acc = test_acc / len(test_set)
        precision, recall, f1, roc_auc_val = self.evaluate_metrics(
            all_true_test, all_preds_test, all_probs_test
        )
    
        return all_true_test, all_preds_test, all_probs_test
    
    
    # Visualization of anomalies
    def plot_anomalies(time_series, labels, preds, start=0, end=1000):
        plt.figure(figsize=(15, 5))
        plt.plot(time_series[start:end], label="Time Series")
        plt.scatter(
            np.arange(start, end)[labels[start:end] == 1],
            time_series[start:end][labels[start:end] == 1],
            color="red",
            label="True Anomalies",
        )
        plt.scatter(
            np.arange(start, end)[preds[start:end] == 1],
            time_series[start:end][preds[start:end] == 1],
            color="orange",
            marker="x",
            label="Predicted Anomalies",
        )
        plt.legend()
        plt.title("Anomaly Detection")
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Value")
        plt.show()
    
    
    # Aggregate predictions on the test set
    def aggregate_predictions(self,indices, preds, window_size, total_length):
        aggregated = np.zeros(total_length, dtype=float)
        counts = np.zeros(total_length, dtype=float)
        for idx, pred in zip(indices, preds):
            start = idx
            end = idx + window_size
            if end > total_length:
                end = total_length
            aggregated[start:end] += pred
            counts[start:end] += 1
        counts[counts == 0] = 1
        averaged = aggregated / counts
        return (averaged > 0.5).astype(int)
    
    
    # Additional Visualization: ROC and Precision-Recall Curves
    def plot_metrics(self,true_labels, pred_probs):
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        roc_auc_val = auc(fpr, tpr)
    
        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_probs)
        pr_auc_val = auc(recall_vals, precision_vals)
    
        plt.figure(figsize=(12, 5))
    
        # ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_val:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
    
        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        plt.plot(recall_vals, precision_vals, label=f"PR Curve (AUC = {pr_auc_val:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (PR) Curve")
        plt.legend()
    
        plt.tight_layout()
        plt.show()
    
    
    def detect_anomalies(self,time_series, labels):
        # Create the dataset
        dataset = TimeSeriesAnomalyDataset(
            time_series,
            labels,
            window_size=self.args.window_size,
            step_size=self.args.step_size,
        )
    
        train_indices, val_indices, test_indices = self.stratified_split(dataset, seed=self.args.seed)
    
        print(
            f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}"
        )
    
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
        # Prepare data for SMOTE
        X_train = [x.numpy().flatten() for x, _ in train_dataset]
        y_train = [int(y.item()) for _, y in train_dataset]
    
        # Implementing SMOTE for oversampling
        smote = SMOTE(random_state=self.args.seed)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
        balanced_train_dataset = ResampledDataset(X_resampled, y_resampled)
    
        # Update the DataLoader
        train_loader = torch.utils.data.DataLoader(
            balanced_train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
    
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False
        )
    
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=False
        )
    
        criterion = FocalLoss(alpha=0.25, gamma=2)
    
        # Initialize the model
        model = KAN(
            in_feat=1,
            hidden_feat=self.args.hidden_size,
            out_feat=1,
            grid_feat=self.args.grid_size,
            num_layers=self.args.n_layers,
            use_bias=True,
            dropout=self.args.dropout,
        ).to(self.args.device)
    
        # Define optimizer with weight decay for regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-5)
    
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
        # Train the model
        optimized_th, _ = self.train_model(
            scheduler, train_loader, balanced_train_dataset, val_dataset, val_loader, model, criterion, optimizer
        )
    
        # Test the model
        all_true_test, all_preds_test, all_probs_test = self.run_model(
            criterion, test_loader, 0.5, test_dataset, model
        )
    
        # Aggregate predictions
        test_sample_indices = [dataset.sample_indices[i] for i in test_indices]
        aggregated_preds = self.aggregate_predictions(
            test_sample_indices, all_preds_test, self.args.window_size, len(time_series)
        )
    
        # Plot anomalies    # Plot anomalies on the test set
        # test_start = min(test_sample_indices)
        # test_end = max(test_sample_indices) + args.window_size
        # plot_anomalies(time_series, labels, aggregated_preds, start=test_start, end=test_end)
        # plot_metrics(all_true_test, all_probs_test)
    
        return aggregated_preds



kan = KANBaseline()
ad = AnomalyDetection('baseline-kan', kan.detect_anomalies)
ad.test()
