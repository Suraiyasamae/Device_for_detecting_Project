import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
# from data_utils_cross_ab import prepare_train_test_data, prepare_cross_validation_data
from data_utils_cross_ab_noise import prepare_train_test_data, prepare_cross_validation_data


class CNN(nn.Module):
    def __init__(self, input_channels=1, dropout_rate=None):
        super().__init__()
        self.feature_reduction = nn.Sequential(
            # Layer 1: Input shape: (batch_size, 1, 24, 32)
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 64, 12, 16)
            nn.Dropout2d(dropout_rate) if dropout_rate else nn.Identity(),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 128, 6, 8)
            nn.Dropout2d(dropout_rate) if dropout_rate else nn.Identity(),
        )

        # Calculate output features for LSTM
        self.output_features = 128 * 6 * 8  # Based on final conv layer output size

    def forward(self, x):
        # x shape: (batch_size, sequence_length, height, width)
        batch_size, seq_len = x.size(0), x.size(1)

        # Reshape for 2D convolution
        x = x.view(batch_size * seq_len, 1, 24, 32)  # Add channel dimension

        # Apply CNN layers
        x = self.feature_reduction(x)

        # Reshape back to sequence format
        x = x.view(batch_size, seq_len, self.output_features)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, sequence_length=20, input_size=None, hidden_size=256,
                 num_classes=4, num_lstm_layers=4, dropout_rate=None):
        super().__init__()

        # CNN feature extractor
        self.cnn = CNN(input_channels=1, dropout_rate=dropout_rate)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.cnn.output_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Extract CNN features
        x = self.cnn(x)

        # Process through LSTM
        lstm_out, _ = self.lstm(x)

        # Use the last time step output for classification
        x = self.classifier(lstm_out[:, -1, :])
        return x


def plot_fold_densities(fold_predictions, save_path):
    """Plot overlapping density plots for all folds."""
    plt.figure(figsize=(10, 6))

    # Create density plot for each fold
    for fold_idx, preds in enumerate(fold_predictions, 1):
        sns.kdeplot(data=preds, label=f'Fold{fold_idx}')

    plt.title('Prediction Distribution Across Folds')
    plt.xlabel('Predicted Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig(os.path.join(save_path, 'fold_densities.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, fold: int, save_path: str):
    """
    Plot and save confusion matrix for a specific fold

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (list): List of class names
        fold (int): Current fold number
        save_path (str): Directory to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - Trial {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f'confusion_matrix_trial_{fold}.png'))
    plt.close()


def plot_learning_curves(train_losses, val_losses, val_accuracies, val_f1s, fold, save_path):
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='#2ecc71', label='Training Loss', linewidth=2)
    plt.plot(val_losses, color='#e74c3c', label='Validation Loss', linewidth=2)
    plt.title(f'Learning Curves - Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, color='#3498db', label='Validation Accuracy', linewidth=2)
    plt.plot(val_f1s, color='#9b59b6', label='Validation F1 Score', linewidth=2)
    plt.title(f'Validation Metrics - Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'fold_{fold + 1}_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_output_directory(dropout_rate):
    """Create output directory for saving results with dropout-specific naming"""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create 'outputs' directory in the same folder as the script
    # Use dropout rate in the directory name
    dropout_str = f"dropout_{str(dropout_rate).replace('.', '_')}"
    output_dir = os.path.join(script_dir, 'outputs', dropout_str)

    # Create subdirectories for different types of outputs
    model_dir = os.path.join(output_dir, 'models')
    plot_dir = os.path.join(output_dir, 'plots')

    # Create all directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    return output_dir, model_dir, plot_dir


def train_and_evaluate_cv_with_ensemble(X_train, y_train, n_splits=5):
    # Create output directories

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #***** Updated configuration*****
    # config = {
    #     'sequence_length': 20,
    #     'input_size': X_train.shape[2],
    #     'hidden_size': 256,
    #     'batch_size': 16,  # Reduced batch size
    #     'learning_rate': 0.0001,  # Reduced learning rate
    #     'num_epochs': 200,  # Increased epochs
    #     'dropout_rate': 0.4,  # Increased dropout
    #     'patience': 20,  # Added patience for early stopping
    #     'min_delta': 0.001  # Minimum improvement required
    # }
    config = {
        'sequence_length': 20,
        'input_size': X_train.shape[2],
        'hidden_size': 58,
        'batch_size': 79,
        'learning_rate': 0.000122,
        'num_epochs': 131,  # Update this
        'num_lstm_layers': 5,  # Add this
        #*************************************
        # 'dropout_rate': 0.1,  # Increased dropout
        # 'dropout_rate': 0.2,
        # 'dropout_rate': 0.3,
        # 'dropout_rate': 0.4,
        # 'dropout_rate': 0.5,
        'dropout_rate': 0.6,
        # 'dropout_rate': 0.7,
        # 'dropout_rate': 0.8,

    }

    # Create output directories after config is defined
    output_dir, model_dir, plot_dir = create_output_directory(config['dropout_rate'])

    print(f"Using device: {device}")
    print(f"Saving outputs to: {output_dir}")

    print(f"Initializing CNN_LSTM with dropout: {config['dropout_rate']}")


    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    best_models = []
    fold_metrics = []
    cv_folds = prepare_cross_validation_data(X_train.numpy(), y_train.numpy(), n_splits)

    class_names = ['Non-request', 'Both hands', 'Left hand', 'Right hand']

    fold_predictions = []

    # ทำ Cross-validation เหมือนเดิม
    # Cross-validation loop
    for fold, (X_tr, y_tr, X_val, y_val) in enumerate(cv_folds):
        print(f"\nFold {fold + 1}/{n_splits}")
        print("-" * 50)

        # Data preparation
        scaler = StandardScaler()
        X_tr_reshaped = X_tr.reshape(-1, X_tr.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

        X_tr_normalized = scaler.fit_transform(X_tr_reshaped)
        X_val_normalized = scaler.transform(X_val_reshaped)

        X_tr_normalized = X_tr_normalized.reshape(X_tr.shape)
        X_val_normalized = X_val_normalized.reshape(X_val.shape)

        # Convert to tensors
        X_tr_tensor = torch.FloatTensor(X_tr_normalized).to(device)
        y_tr_tensor = torch.LongTensor(y_tr).to(device)
        X_val_tensor = torch.FloatTensor(X_val_normalized).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_tr),
            y=y_tr
        )
        temperature = 1.5
        class_weights = np.exp(np.log(class_weights) / temperature)

        # Model initialization
        model = CNN_LSTM(
            sequence_length=config['sequence_length'],
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate'],
            num_lstm_layers=config['num_lstm_layers']
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.000164
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )

        # Create data loaders
        train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True
        )
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        # Training loop
        best_val_f1 = 0
        best_model_state = None
        train_losses, val_losses = [], []
        val_accuracies, val_f1s = [], []

        for epoch in range(config['num_epochs']):
            # Training phase
            model.train()
            total_train_loss = 0

            for sequences, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()

            # Validation phase
            model.eval()
            total_val_loss = 0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for sequences, labels in val_loader:
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # Calculate metrics
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            val_f1s.append(val_f1)

            scheduler.step(val_f1)

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()

            # Print epoch results
            print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], '
                  f'Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

        # Evaluate best model for this fold
        model.load_state_dict(best_model_state)
        model.eval()
        final_val_preds, final_val_labels = [], []

        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                final_val_preds.extend(predicted.cpu().numpy())
                final_val_labels.extend(labels.cpu().numpy())

        model.eval()
        fold_preds = []
        with torch.no_grad():
            for sequences, _ in val_loader:
                outputs = model(sequences)
                probabilities = torch.softmax(outputs, dim=1)
                fold_preds.extend(probabilities.cpu().numpy())

        fold_predictions.append(np.array(fold_preds).flatten())

        # Calculate final fold metrics
        fold_accuracy = accuracy_score(final_val_labels, final_val_preds)
        fold_f1 = f1_score(final_val_labels, final_val_preds, average='weighted')
        fold_report = classification_report(final_val_labels, final_val_preds, digits=4)

        # Store fold results
        fold_metrics.append({
            'accuracy': fold_accuracy,
            'f1': fold_f1,
            'precision': precision_score(final_val_labels, final_val_preds, average='weighted'),
            'recall': recall_score(final_val_labels, final_val_preds, average='weighted'),
            'classification_report': fold_report
        })

        # Print fold results
        print(f"\nFold {fold + 1} Final Results:")
        print("=" * 50)
        print(f"Accuracy: {fold_accuracy:.4f}")
        print(f"F1 Score: {fold_f1:.4f}")
        print("\nClassification Report:")
        print(fold_report)
        print("=" * 50)

        # Plot learning curves
        plot_learning_curves(train_losses, val_losses, val_accuracies, val_f1s, fold, plot_dir)
        plot_confusion_matrix(final_val_labels, final_val_preds, class_names, fold, plot_dir)
        plot_fold_densities(fold_predictions, plot_dir)

        # Save best model from this fold
        best_models.append({
            'model': best_model_state,
            'scaler': scaler
        })

        # Save fold model
        fold_model_path = os.path.join(model_dir, f'fold_{fold + 1}_model.pth')
        torch.save(best_model_state, fold_model_path)

    # Print cross-validation summary
    print("\nCross-validation Summary:")
    print("=" * 50)
    print("Results for each fold:")
    for i, metrics in enumerate(fold_metrics):
        print(f"\nFold {i + 1}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

    print("\nAverage Metrics Across All Folds:")
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    std_f1 = np.std([m['f1'] for m in fold_metrics])
    avg_precision = np.mean([m['precision'] for m in fold_metrics])
    std_precision = np.std([m['precision'] for m in fold_metrics])
    avg_recall = np.mean([m['recall'] for m in fold_metrics])
    std_recall = np.std([m['recall'] for m in fold_metrics])

    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    print("=" * 50)

    # Save metrics summary
    with open(os.path.join(output_dir, 'cross_validation_summary.txt'), 'w') as f:
        f.write("Cross-validation Results:\n")
        f.write("=" * 50 + "\n")
        f.write("Results for each fold:\n")

        for i, metrics in enumerate(fold_metrics):
            f.write(f"\nFold {i + 1}:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(metrics['classification_report'] + "\n")

        f.write("\nAverage Metrics Across All Folds:\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}\n")

    return fold_metrics, best_models, scaler




if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_train_test_data()

    # Perform cross validation with ensemble learning
    fold_metrics, final_model, final_scaler = train_and_evaluate_cv_with_ensemble(
        X_train, y_train, n_splits=5
    )