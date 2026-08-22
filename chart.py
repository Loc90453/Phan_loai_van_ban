# ============================================================
# TÁI TẠO BIỂU ĐỒ TỪ MODEL ĐÃ TRAIN XONG (ĐÃ FIX SẠCH LỖI ĐƯỜNG DẪN)
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
import unicodedata
import re

MODEL_PATH = './phobert_best'
MAX_LEN = 256
BATCH_SIZE = 32

LABEL_VI = [
    'Thời sự', 'Thế giới', 'Kinh doanh', 'Khoa học công nghệ',
    'Bất động sản', 'Sức khỏe', 'Thể thao', 'Giải trí',
    'Pháp luật', 'Giáo dục', 'Đời sống',
]

STOPWORDS = set([
    'và','của','là','các','để','với','trong','cho','về','từ',
    'có','được','này','đã','không','một','những','theo','ra',
    'đó','thì','trên','mà','khi','đến','bị','vì','tại','hay',
    'hoặc','như','nhưng','cũng','vào','lên','rằng','lại','sau',
    'trước','qua','hơn','đây','ở','cùng','bởi','chỉ','đang',
    'sẽ','nên','phải','vẫn','đều','rất','nữa','thêm','giữa',
    'đi','lúc','nay','xem','tuy','dù','mới','còn','gì','ai',
    'nào','sao','thế','hội','tôi','bạn','anh','chị','ông','bà',
])

def clean_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'http\S+|www\S+|<[^>]+>|\S+@\S+|[^\w\s]|\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        from underthesea import word_tokenize
        text = word_tokenize(text, format='text')
    except Exception:
        pass
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load model
print(f'⏳ Đang nạp model từ {MODEL_PATH} trên {device}...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()
print('✅ Nạp model thành công!')

# 2. Xác định file dữ liệu khả dụng
data_sources = [
    'data/processed/data_clean.csv',
    'data/test_balanced_5500.csv',
    'data/test_quick_550.csv'
]
data_path = None
for p in data_sources:
    if os.path.exists(p):
        data_path = p
        break

if not data_path:
    raise FileNotFoundError("Không tìm thấy file dữ liệu nào trong data/ để vẽ biểu đồ.")

print(f'📖 Đang đọc dữ liệu từ: {data_path}')
df = pd.read_csv(data_path)

if 'text_clean' in df.columns:
    X_raw = df['text_clean'].values
else:
    print('⏳ Đang tiền xử lý văn bản...')
    X_raw = [clean_text(t) for t in df['text'].values]

y_id = df['label_id'].values if 'label_id' in df.columns else df['label'].values

# Lấy tập test
if len(X_raw) > 2000:
    _, X_test, _, y_test = train_test_split(X_raw, y_id, test_size=0.2, stratify=y_id, random_state=42)
else:
    X_test, y_test = X_raw, y_id

print(f'✅ Tập đánh giá gồm: {len(X_test)} mẫu.')

# 3. Dataset & DataLoader
class VNTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return {'text': str(self.texts[idx]), 'label': int(self.labels[idx])}

def collate_fn(batch):
    texts = [b['text'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    enc = tokenizer(texts, max_length=MAX_LEN, padding=True, truncation=True, return_tensors='pt')
    enc['labels'] = labels
    return enc

test_loader = DataLoader(VNTextDataset(X_test, y_test), batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 4. Dự đoán
preds, trues = [], []
with torch.no_grad():
    for batch in test_loader:
        out = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
        preds += torch.argmax(out.logits, dim=1).cpu().tolist()
        trues += batch['labels'].tolist()

test_preds = np.array(preds)
test_true = np.array(trues)

os.makedirs('results', exist_ok=True)

print('\n' + '='*55)
print(' KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST')
print('='*55)
print(f'Accuracy: {accuracy_score(test_true, test_preds):.4f}\n')
print(classification_report(test_true, test_preds, target_names=LABEL_VI, digits=4))

# 5. Vẽ Confusion Matrix
cm = confusion_matrix(test_true, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_VI, yticklabels=LABEL_VI, ax=axes[0])
axes[0].set_title('Confusion Matrix (Số lượng)')
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=LABEL_VI, yticklabels=LABEL_VI, ax=axes[1])
axes[1].set_title('Confusion Matrix (Tỷ lệ %)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.close()
print('💾 Đã lưu: results/confusion_matrix.png')

# 6. Biểu đồ Precision / Recall / F1 từng lớp
precision_pc, recall_pc, f1_pc, _ = precision_recall_fscore_support(
    test_true, test_preds, labels=range(len(LABEL_VI)), zero_division=0
)
x = np.arange(len(LABEL_VI))
width = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, precision_pc, width, label='Precision', color='#4C72B0')
ax.bar(x, recall_pc, width, label='Recall', color='#55A868')
ax.bar(x + width, f1_pc, width, label='F1-score', color='#C44E52')
ax.set_xticks(x)
ax.set_xticklabels(LABEL_VI, rotation=30, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Điểm số')
ax.set_title('Precision / Recall / F1-score theo từng chủ đề')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/metrics_per_class.png', dpi=150)
plt.close()
print('💾 Đã lưu: results/metrics_per_class.png')
print('🎉 Hoàn tất toàn bộ biểu đồ!')