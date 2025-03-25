import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_score,
    recall_score,
    confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from data_utils_cross_ab_noise import prepare_train_test_data
from cnn2lstm5 import CNN_LSTM


def plot_f1_scores(results, save_dir=None):
    """
    สร้างกราฟแท่งแสดง F1 scores สำหรับแต่ละระดับ noise ด้วยการแสดงผลที่สวยงาม

    Parameters:
    results (dict): dictionary ที่มี noise levels เป็น keys และมี metrics เป็น values
    save_dir (str): path ที่ต้องการบันทึกกราฟ (ถ้าไม่ระบุจะบันทึกในโฟลเดอร์ปัจจุบัน)
    """
    # เตรียมข้อมูลสำหรับพล็อต
    noise_levels = [k * 100 for k in results.keys()]
    f1_scores = [v['f1_score'] for v in results.values()]

    # สร้างรายการ labels สำหรับแกน x
    x_labels = [f'noise-{int(level)}%' for level in noise_levels]

    plt.figure(figsize=(12, 6))

    # สร้าง x positions สำหรับแท่งกราฟที่ชิดกัน
    x_pos = np.arange(len(noise_levels))

    # สร้างกราฟแท่ง
    bars = plt.bar(x_pos, f1_scores, width=0.6, color='royalblue', alpha=0.7)

    # ตกแต่งกราฟ
    plt.title('F1 Score vs Noise Level', fontsize=14, pad=20)
    plt.xlabel('Noise Level', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # ใส่ค่า F1 score บนแท่งกราฟ
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    # ปรับขอบเขตแกน y
    plt.ylim(0, max(f1_scores) * 1.1)

    # กำหนด x ticks และ labels
    plt.xticks(x_pos, x_labels)

    # กำหนด path สำหรับบันทึกไฟล์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'f1_scores_bar_comparison_{timestamp}.png')
    else:
        save_path = f'f1_scores_bar_comparison_{timestamp}.png'

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"บันทึกกราฟที่: {save_path}")
    plt.close()

def test_ensemble_models_with_noise(model_dir, X_test, y_test, noise_levels=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    batch_size=32, device='cuda'):
    results = {}
    fold_models = []

    # 1. โหลดทุกโมเดลจาก cross validation
    for fold_file in sorted(os.listdir(model_dir)):
        if fold_file.endswith('.pth'):
            model = CNN_LSTM(
                sequence_length=20,
                input_size=X_test.shape[2],
                hidden_size=132,
                num_classes=4,
                num_lstm_layers=5,
                dropout_rate=0.4
            ).to(device)

            model_path = os.path.join(model_dir, fold_file)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            fold_models.append(model)

    print(f"Loaded {len(fold_models)} models from {model_dir}")

    # 2. Normalize ข้อมูล
    scaler = StandardScaler()
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_normalized = scaler.fit_transform(X_test_reshaped)
    X_test = X_test_normalized.reshape(X_test.shape)

    for noise_level in noise_levels:
        print(f"\nTesting with {noise_level * 100}% noise level:")

        # 3. เพิ่ม noise
        X_test_noisy = X_test.copy()
        noise = np.random.normal(0, np.std(X_test) * noise_level, X_test.shape)
        X_test_noisy = X_test_noisy + noise

        # 4. แปลงข้อมูลเป็น tensor
        X_test_tensor = torch.FloatTensor(X_test_noisy)
        y_test_tensor = torch.LongTensor(y_test)

        test_dataset = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 5. เก็บ predictions จากทุกโมเดล
        all_fold_predictions = []
        all_fold_probabilities = []
        true_labels = []

        # 6. ทำนายด้วยแต่ละโมเดล
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_probabilities = []

                if len(batch_X.shape) == 3:
                    batch_X = batch_X.view(batch_X.size(0), batch_X.size(1), 24, 32)

                # ทำนายด้วยทุกโมเดล
                for model in fold_models:
                    outputs = model(batch_X)
                    probabilities = torch.softmax(outputs, dim=1)
                    batch_probabilities.append(probabilities.cpu().numpy())

                # เฉลี่ย probabilities จากทุกโมเดล
                avg_probabilities = np.mean(batch_probabilities, axis=0)
                predictions = np.argmax(avg_probabilities, axis=1)

                all_fold_predictions.extend(predictions)
                all_fold_probabilities.extend(avg_probabilities)
                true_labels.extend(batch_y.cpu().numpy())

        # 7. คำนวณ metrics จาก ensemble predictions
        accuracy = accuracy_score(true_labels, all_fold_predictions)
        f1 = f1_score(true_labels, all_fold_predictions, average='weighted')
        precision = precision_score(true_labels, all_fold_predictions, average='weighted')
        recall = recall_score(true_labels, all_fold_predictions, average='weighted')

        print(f"\nEnsemble Results for Noise Level {noise_level * 100}%:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        # 8. แสดง confusion matrix
        cm = confusion_matrix(true_labels, all_fold_predictions)
        print("\nConfusion Matrix:")
        print(cm)

        # 9. แสดง detailed report
        print("\nClassification Report:")
        print(classification_report(true_labels, all_fold_predictions, digits=4))

        results[noise_level] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': all_fold_predictions,
            'true_labels': true_labels
        }

    return results


if __name__ == "__main__":
    # 1. โหลดข้อมูล test
    X_train, X_test, y_train, y_test = prepare_train_test_data()

    # 2. ระบุ path ของโฟลเดอร์ที่เก็บโมเดล
    model_dir = r"C:\model101\venv\Scripts\models"  # โฟลเดอร์ที่มีไฟล์ .pth จาก cross validation


    # 3. ทดสอบ ensemble
    results = test_ensemble_models_with_noise(
        model_dir=model_dir,
        X_test=X_test,
        y_test=y_test,
        noise_levels=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # 4. สร้างกราฟเปรียบเทียบ F1 scores
    plot_f1_scores(results)