import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import unicodedata
import re
import numpy as np
import time

MODEL_PATH = './phobert_best'

LABEL_NAMES = [
    'Thời sự', 'Thế giới', 'Kinh doanh', 'Khoa học công nghệ',
    'Bất động sản', 'Sức khỏe', 'Thể thao', 'Giải trí',
    'Pháp luật', 'Giáo dục', 'Đời sống'
]

LABEL_ICONS = {
    'Thời sự': '📰', 'Thế giới': '🌍', 'Kinh doanh': '📈', 'Khoa học công nghệ': '🔬',
    'Bất động sản': '🏢', 'Sức khỏe': '🏥', 'Thể thao': '⚽', 'Giải trí': '🎬',
    'Pháp luật': '⚖️', 'Giáo dục': '📚', 'Đời sống': '🏡'
}

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Thiết bị sử dụng: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

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

def predict(text):
    cleaned = clean_text(text) or text
    enc = tokenizer(cleaned, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = model(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device))
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

# Bộ kiểm thử 22 bài viết đa dạng bao quát 11 chủ đề
TEST_SUITE = [
    # 📰 Thời sự
    ("Chủ tịch nước tiếp đón đoàn đại biểu cấp cao thăm chính thức Việt Nam", "Thời sự"),
    ("Thủ tướng chủ trì cuộc họp khẩn về công tác khắc phục hậu quả thiên tai", "Thời sự"),

    # 🌍 Thế giới
    ("Hội đồng Bảo an Liên Hợp Quốc thông qua nghị quyết kêu gọi ngừng bắn", "Thế giới"),
    ("Mỹ và các nước đồng minh thảo luận về thỏa thuận an ninh hàng hải quốc tế", "Thế giới"),

    # 📈 Kinh doanh
    ("Ngân hàng Nhà nước giảm lãi suất điều hành, bơm thanh khoản cho nền kinh tế", "Kinh doanh"),
    ("Chỉ số chứng khoán VN-Index bứt phá vượt mốc 1300 điểm với thanh khoản lớn", "Kinh doanh"),

    # 🔬 Khoa học công nghệ
    ("Apple ra mắt dòng vi xử lý M3 với kiến trúc bán dẫn 3nm tiên tiến", "Khoa học công nghệ"),
    ("Các nhà khoa học chế tạo thành công pin năng lượng mặt trời hiệu suất kỷ lục", "Khoa học công nghệ"),

    # 🏢 Bất động sản
    ("Giá căn hộ chung cư tại Hà Nội và TP.HCM tiếp tục tăng mạnh trong quý này", "Bất động sản"),
    ("Thị trường đất nền vùng ven ghi nhận thanh khoản phục hồi tích cực", "Bất động sản"),

    # 🏥 Sức khỏe
    ("Bộ Y tế khuyến cáo người dân chủ động tiêm phòng vaccine cúm mùa và sốt xuất huyết", "Sức khỏe"),
    ("Bệnh viện triển khai phẫu thuật nội soi robot điều trị thành công ca bệnh khó", "Sức khỏe"),

    # ⚽ Thể thao
    ("Đội tuyển U23 Việt Nam giành chiến thắng thuyết phục 3-0 trong trận ra quân", "Thể thao"),
    ("Tiền đạo ghi cú đúp giúp câu lạc bộ đoạt cúp vô địch bóng đá quốc gia", "Thể thao"),

    # 🎬 Giải trí
    ("Bộ phim điện ảnh mới của đạo diễn Việt Nam cán mốc 200 tỷ đồng doanh thu phòng vé", "Giải trí"),
    ("Ca sĩ ra mắt MV ca nhạc mới sau thời gian dài vắng bóng trên sân khấu", "Giải trí"),

    # ⚖️ Pháp luật
    ("Tòa án nhân dân tuyên phạt bị cáo mức án tù chung thân về tội lừa đảo chiếm đoạt tài sản", "Pháp luật"),
    ("Cơ quan điều tra khởi tố và bắt tạm giam đối tượng cầm đầu đường dây buôn lậu", "Pháp luật"),

    # 📚 Giáo dục
    ("Bộ Giáo dục công bố điểm chuẩn và phổ điểm thi tốt nghiệp THPT năm nay", "Giáo dục"),
    ("Trường đại học mở thêm ngành đào tạo Trí tuệ nhân tạo và Khoa học dữ liệu", "Giáo dục"),

    # 🏡 Đời sống
    ("Người dân miền Trung khắc phục thiệt hại sau đợt mưa lũ lịch sử", "Thời sự"),
    ("Nhiều gia đình lựa chọn du lịch cắm trại sinh thái ngoài trời dịp cuối tuần", "Đời sống"),
]

print("="*85)
print("🚀 BẮT ĐẦU CHẠY KIỂM THỬ TOÀN DIỆN (COMPREHENSIVE TEST SUITE)")
print("="*85)

correct_count = 0
total = len(TEST_SUITE)
latencies = []

for idx, (text, expected) in enumerate(TEST_SUITE, 1):
    t0 = time.time()
    pred_idx, probs = predict(text)
    latency = (time.time() - t0) * 1000
    latencies.append(latency)
    
    pred_label = LABEL_NAMES[pred_idx]
    conf = probs[pred_idx] * 100
    is_correct = (pred_label == expected)
    
    if is_correct:
        correct_count += 1
        status = "✅ ĐÚNG"
    else:
        status = f"⚠️ LỆCH ({expected})"
        
    icon = LABEL_ICONS[pred_label]
    short_text = text[:48] + "..." if len(text) > 48 else text
    print(f"[{idx:>2}/{total}] {status:20s} | {icon} {pred_label:18s} ({conf:5.1f}%) | {short_text}")

print("\n" + "="*85)
print(f"📊 KẾT QUẢ TỔNG KẾT:")
print(f"🎯 Độ chính xác trên bộ test mẫu: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
print(f"⚡ Độ trễ suy luận trung bình   : {np.mean(latencies):.1f} ms / bài báo")
print("="*85)
