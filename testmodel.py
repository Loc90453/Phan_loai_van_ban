from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = './phobert_best'   # đường dẫn tới thư mục vừa giải nén

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

print('✅ Load model thành công, sẵn sàng dự đoán!')

ID2LABEL = {
    0: 'thoi_su', 1: 'the_gioi', 2: 'kinh_doanh', 3: 'khoa_hoc_cong_nghe',
    4: 'bat_dong_san', 5: 'suc_khoe', 6: 'the_thao', 7: 'giai_tri',
    8: 'phap_luat', 9: 'giao_duc', 10: 'doi_song',
}

def predict(text):
    enc = tokenizer(text, max_length=256, padding='max_length',
                     truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = model(input_ids=enc['input_ids'].to(device),
                     attention_mask=enc['attention_mask'].to(device))
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    pred_id = probs.argmax()
    return ID2LABEL[pred_id], probs[pred_id]

test_texts = [
    # 📰 Thời sự
    'Quốc hội thảo luận về dự thảo luật đất đai sửa đổi',
    'Thủ tướng chủ trì hội nghị tổng kết công tác phòng chống thiên tai',
    'Chủ tịch nước tiếp đón đoàn ngoại giao cấp cao thăm chính thức',

    # 🌍 Thế giới
    'Liên Hợp Quốc kêu gọi các bên đối thoại để giải quyết xung đột',
    'Mỹ và Trung Quốc nối lại đàm phán thương mại sau nhiều tháng căng thẳng',
    'Động đất mạnh xảy ra tại Nhật Bản, nhiều khu vực bị ảnh hưởng',

    # 📈 Kinh doanh
    'Ngân hàng Nhà nước giảm lãi suất điều hành hỗ trợ doanh nghiệp',
    'Chỉ số VN-Index tăng điểm mạnh trong phiên giao dịch cuối tuần',
    'Doanh nghiệp xuất khẩu gạo ghi nhận doanh thu kỷ lục trong năm',

    # 🔬 Khoa học công nghệ
    'Apple ra mắt chip xử lý thế hệ mới cho dòng iPhone',
    'Các nhà khoa học phát triển pin mặt trời hiệu suất cao hơn 30%',
    'Trí tuệ nhân tạo được ứng dụng trong chẩn đoán hình ảnh y tế',

    # 🏢 Bất động sản
    'Giá căn hộ chung cư tại Hà Nội tiếp tục tăng trong quý này',
    'Nhiều dự án bất động sản nghỉ dưỡng ven biển được khởi công',
    'Thị trường đất nền vùng ven TP.HCM ghi nhận giao dịch sôi động',

    # 🏥 Sức khỏe
    'Bộ Y tế khuyến cáo người dân tiêm vaccine phòng bệnh đầy đủ',
    'Bệnh viện triển khai kỹ thuật mổ nội soi tiên tiến điều trị ung thư',
    'Số ca mắc sốt xuất huyết tăng cao tại các tỉnh phía Nam',

    # ⚽ Thể thao
    'Đội tuyển U23 Việt Nam giành chiến thắng trong trận ra quân',
    'Cầu thủ Việt Nam chuyển nhượng sang câu lạc bộ nước ngoài thi đấu',
    'Giải marathon quốc tế thu hút hàng nghìn vận động viên tham gia',

    # 🎬 Giải trí
    'Bộ phim mới của đạo diễn Việt Nam công chiếu cuối tuần này',
    'Nam ca sĩ ra mắt album mới sau ba năm vắng bóng',
    'Chương trình truyền hình thực tế gây sốt mạng xã hội tuần qua',

    # ⚖️ Pháp luật
    'Tòa án tuyên án bị cáo trong vụ lừa đảo chiếm đoạt tài sản',
    'Cơ quan điều tra khởi tố thêm nhiều đối tượng liên quan vụ án',
    'Luật mới quy định chặt chẽ hơn về xử phạt vi phạm giao thông',

    # 📚 Giáo dục
    'Bộ Giáo dục công bố kết quả thi tốt nghiệp THPT năm nay',
    'Trường đại học mở thêm ngành đào tạo trí tuệ nhân tạo',
    'Học sinh giỏi quốc gia được tuyển thẳng vào các trường top đầu',

    # 🏡 Đời sống
    'Người dân miền Trung ứng phó với đợt mưa lũ lớn',
    'Giá thực phẩm tại các chợ truyền thống tăng nhẹ dịp cuối năm',
    'Nhiều gia đình lựa chọn du lịch trong nước dịp nghỉ lễ dài ngày',
]

# --- Lặp qua từng câu và dự đoán ---
for text in test_texts:
    label, conf = predict(text)
    short = text[:55] + '...' if len(text) > 55 else text
    print(f'{short:60s} → {label:20s} ({conf:.1%})')

label, conf = predict("Bộ Y tế khuyến cáo tiêm vaccine phòng bệnh")
print(f'Chủ đề: {label} ({conf:.1%})')