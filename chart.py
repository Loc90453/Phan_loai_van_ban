# ============================================================
# TÁI TẠO BIỂU ĐỒ TỪ MODEL ĐÃ TRAIN XONG (không cần train lại)
# ============================================================
import torch
import pandas as pd
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cấu hình (PHẢI khớp với lúc train) ---
MODEL_PATH = './phobert_best'      # đường dẫn tới model đã giải nén
DATA_PATH  = 'data/processed/data_clean.csv'   # cần có lại file data_clean.csv
MAX_LEN    = 256
BATCH_SIZE = 32

LABEL_VI = [
    'Thời sự', 'Thế giới', 'Kinh doanh', 'Khoa học công nghệ',
    'Bất động sản', 'Sức khỏe', 'Thể thao', 'Giải trí',
    'Pháp luật', 'Giáo dục', 'Đời sống',
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load lại model đã train ---
print('⏳ Đang load model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()
print('✅ Load model xong!')

# --- Tái tạo LẠI đúng tập test (dùng cùng random_state=42 như lúc train) ---
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
X = df['text_clean'].values
y_id = df['label_id'].values

X_train, X_temp, yid_train, yid_temp = train_test_split(
    X, y_id, test_size=0.30, stratify=y_id, random_state=42
)
X_val, X_test, yid_val, yid_test = train_test_split(
    X_temp, yid_temp, test_size=0.50, stratify=yid_temp, random_state=42
)
print(f'✅ Tái tạo tập test: {len(X_test)} mẫu (phải khớp số lượng lúc train)')

# --- Dataset + DataLoader cho tập test ---
class VNTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tok = texts, labels, tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return {'text': str(self.texts[idx]), 'label': int(self.labels[idx])}

def collate_fn(batch):
    texts  = [b['text'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    enc = tokenizer(texts, max_length=MAX_LEN, padding=True, truncation=True, return_tensors='pt')
    enc['labels'] = labels
    return enc

test_loader = DataLoader(VNTextDataset(X_test, yid_test, tokenizer),
                          batch_size=BATCH_SIZE, collate_fn=collate_fn)

# --- Dự đoán trên tập test ---
def evaluate(loader):
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device))
            preds += torch.argmax(out.logits, dim=1).cpu().tolist()
            trues += batch['labels'].tolist()
    return np.array(preds), np.array(trues)

print('⏳ Đang dự đoán trên tập test...')
test_preds, test_true = evaluate(test_loader)
print('✅ Xong!')

os.makedirs('results', exist_ok=True)

print('\n' + '='*55)
print(' KẾT QUẢ TRÊN TẬP TEST')
print('='*55)
print(f'Accuracy: {accuracy_score(test_true, test_preds):.4f}\n')
print(classification_report(test_true, test_preds, target_names=LABEL_VI))

# --- Biểu đồ Precision/Recall/F1 theo từng chủ đề ---
precision_pc, recall_pc, f1_pc, _ = precision_recall_fscore_support(
    test_true, test_preds, labels=range(len(LABEL_VI)), zero_division=0
)
acc_overall = accuracy_score(test_true, test_preds)
precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
    test_true, test_preds, average='weighted', zero_division=0
)

x = np.arange(len(LABEL_VI))
width = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, precision_pc, width, label='Precision', color='#4C72B0')
ax.bar(x,          recall_pc,    width, label='Recall',    color='#55A868')
ax.bar(x + width,  f1_pc,        width, label='F1-score',  color='#C44E52')
ax.set_xticks(x); ax.set_xticklabels(LABEL_VI, rotation=30, ha='right')
ax.set_ylim(0, 1.05); ax.set_ylabel('Điểm số')
ax.set_title('Precision / Recall / F1-score theo từng chủ đề')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('results/metrics_per_class.png', dpi=150); plt.show()

# --- Biểu đồ tổng hợp Accuracy/Precision/Recall/F1 ---
metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1-score']
metric_values = [acc_overall, precision_w, recall_w, f1_w]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(metric_names, metric_values, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
ax.set_ylim(0, 1.05); ax.set_ylabel('Điểm số')
ax.set_title('Kết quả tổng hợp trên tập Test (Weighted Average)')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.4f}', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig('results/metrics_summary.png', dpi=150); plt.show()

# --- Confusion Matrix ---
cm = confusion_matrix(test_true, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_VI, yticklabels=LABEL_VI, ax=axes[0])
axes[0].set_title('Confusion Matrix (Số lượng)')
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=LABEL_VI, yticklabels=LABEL_VI, ax=axes[1])
axes[1].set_title('Confusion Matrix (Tỷ lệ %)')
plt.tight_layout(); plt.savefig('results/confusion_matrix.png', dpi=150); plt.show()

# --- Biểu đồ Loss/Accuracy theo epoch — lấy từ checkpoint.pt (không cần train lại) ---
if os.path.exists('models/checkpoint.pt'):
    ckpt = torch.load('models/checkpoint.pt', map_location='cpu')
    df_hist = pd.DataFrame(ckpt['history'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df_hist['epoch'], df_hist['train_loss'], marker='o', color='coral')
    axes[0].set_title('Train Loss theo Epoch')

    axes[1].plot(df_hist['epoch'], df_hist['train_acc'], marker='o', label='Train', color='steelblue')
    axes[1].plot(df_hist['epoch'], df_hist['test_acc'], marker='s', label='Test', color='darkorange')
    axes[1].set_title('Accuracy theo Epoch (Train vs Test)'); axes[1].legend()
    plt.tight_layout(); plt.savefig('results/training_history.png', dpi=150); plt.show()
    print(f"\nBest Val Accuracy đã lưu: {ckpt['best_val_acc']:.4f}")
else:
    print('⚠️ Không tìm thấy checkpoint.pt — bỏ qua biểu đồ Loss/Accuracy theo epoch')