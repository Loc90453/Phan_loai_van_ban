"""
=============================================================================
HỆ THỐNG PHÂN LOẠI VĂN BẢN TIẾNG VIỆT ĐA CHỦ ĐỀ & GIẢI THÍCH MÔ HÌNH (PHOBERT + XAI)
Đề tài: Xử lý ngôn ngữ tự nhiên (NLP) — Bậc Thạc sĩ
Mô hình nòng cốt: PhoBERT (VinAI Research) + Explainable AI (Leave-One-Out Attribution)
=============================================================================
"""

import os
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =============================================================================
# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN HIỆN ĐẠI (PREMIUM DESIGN SYSTEM)
# =============================================================================
st.set_page_config(
    page_title="Phân Loại Văn Bản Tiếng Việt - PhoBERT AI",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        line-height: 1.6;
    }
    
    /* Result Card */
    .result-card-main {
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    /* Metric Badge */
    .stat-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    
    /* Quick chip buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* XAI Tokens */
    .xai-badge {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        cursor: default;
    }
    .xai-badge:hover {
        transform: scale(1.08);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. HẰNG SỐ, NHÃN & MẪU DỮ LIỆU CHUẨN
# =============================================================================
MODEL_PATH = './phobert_best'

LABEL_NAMES: List[str] = [
    'Thời sự', 'Thế giới', 'Kinh doanh', 'Khoa học công nghệ',
    'Bất động sản', 'Sức khỏe', 'Thể thao', 'Giải trí',
    'Pháp luật', 'Giáo dục', 'Đời sống'
]

LABEL_INFO: Dict[str, Dict] = {
    'Thời sự': {'id': 0, 'icon': '📰', 'color': '#2563EB', 'desc': 'Chính trị, sự kiện quốc gia, điều hành chính phủ, chính sách vĩ mô'},
    'Thế giới': {'id': 1, 'icon': '🌍', 'color': '#0284C7', 'desc': 'Ngoại giao quốc tế, xung đột, kinh tế toàn cầu, hợp tác đa phương'},
    'Kinh doanh': {'id': 2, 'icon': '📈', 'color': '#059669', 'desc': 'Tài chính, chứng khoán, ngân hàng, doanh nghiệp, thương mại'},
    'Khoa học công nghệ': {'id': 3, 'icon': '🔬', 'color': '#7C3AED', 'desc': 'Trí tuệ nhân tạo, thiết bị viễn thông, phần mềm, phát minh khoa học'},
    'Bất động sản': {'id': 4, 'icon': '🏢', 'color': '#D97706', 'desc': 'Quy hoạch đất đai, thị trường nhà ở, chung cư, dự án địa ốc, sổ đỏ'},
    'Sức khỏe': {'id': 5, 'icon': '🏥', 'color': '#DC2626', 'desc': 'Y tế, phòng chống dịch bệnh, vaccine, dinh dưỡng, bệnh viện'},
    'Thể thao': {'id': 6, 'icon': '⚽', 'color': '#16A34A', 'desc': 'Bóng đá, giải đấu quốc tế, chuyển nhượng, vận động viên, thể thao'},
    'Giải trí': {'id': 7, 'icon': '🎬', 'color': '#DB2777', 'desc': 'Điện ảnh, âm nhạc, nghệ sĩ, show truyền hình thực tế, văn hóa'},
    'Pháp luật': {'id': 8, 'icon': '⚖️', 'color': '#475569', 'desc': 'Xét xử tòa án, khởi tố vụ án, an ninh trật tự, luật tố tụng hình sự'},
    'Giáo dục': {'id': 9, 'icon': '📚', 'color': '#EA580C', 'desc': 'Tuyển sinh, kỳ thi tốt nghiệp THPT, trường đại học, học bổng'},
    'Đời sống': {'id': 10, 'icon': '🏡', 'color': '#0D9488', 'desc': 'Gia đình, du lịch, ẩm thực, phong cách sống, kinh nghiệm dân sinh'}
}

# Danh sách 11 bài báo mẫu đại diện cho 11 chủ đề
SAMPLE_ARTICLES: Dict[str, str] = {
    "🔬 Công nghệ - Chip xử lý": "Apple chính thức công bố chip xử lý thế hệ mới với kiến trúc bán dẫn 3nm mang lại hiệu năng tính toán AI vượt trội trên thiết bị di động.",
    "🏢 Bất động sản - Dự án địa ốc": "Giám đốc Công an tỉnh đề nghị luật hóa việc cấm quảng cáo quá mức thực tế sai sự thật về dự án bất động sản trước khi bán cho khách hàng tại phiên thảo luận Luật Kinh doanh bất động sản sửa đổi.",
    "🏥 Sức khỏe - Phòng ngừa dịch": "Bộ Y tế khuyến cáo người dân chủ động tiêm vaccine phòng bệnh mùa đông xuân và chủ động tiêu diệt bọ gậy để phòng sốt xuất huyết.",
    "📈 Kinh doanh - Lãi suất điều hành": "Ngân hàng Nhà nước giảm lãi suất điều hành nhằm hỗ trợ các doanh nghiệp tiếp cận nguồn vốn vay phục hồi sản xuất và kinh doanh.",
    "📰 Thời sự - Kỳ họp Quốc hội": "Chủ tịch Quốc hội chủ trì phiên họp toàn thể thảo luận về dự thảo các luật trọng điểm phục vụ phát triển kinh tế xã hội đất nước.",
    "⚽ Thể thao - Chiến thắng bóng đá": "Đội tuyển U23 Việt Nam đã có màn trình diễn xuất sắc và giành chiến thắng thuyết phục 3-0 trong trận đấu ra quân tại giải vô địch châu Á.",
    "⚖️ Pháp luật - Tòa án tuyên án": "Tòa án nhân dân cấp cao mở phiên tòa xét xử phúc thẩm và tuyên phạt bị cáo 15 năm tù về tội lừa đảo chiếm đoạt tài sản qua mạng.",
    "📚 Giáo dục - Kỳ thi tốt nghiệp": "Bộ Giáo dục và Đào tạo công bố cấu trúc định dạng đề thi tham khảo tốt nghiệp THPT và hướng dẫn tuyển sinh đại học năm nay.",
    "🌍 Thế giới - Đối thoại ngoại giao": "Hội đồng Bảo an Liên Hợp Quốc thông qua nghị quyết kêu gọi các bên xung đột lập tức ngừng bắn và nối lại đàm phán hòa bình.",
    "🎬 Giải trí - Phim điện ảnh": "Bộ phim điện ảnh mới của đạo diễn Việt Nam đã chính thức vượt mốc doanh thu 200 tỷ đồng sau hai tuần công chiếu tại các rạp trên toàn quốc.",
    "🏡 Đời sống - Du lịch nghỉ lễ": "Nhiều gia đình tại các thành phố lớn lựa chọn kỳ nghỉ du lịch sinh thái và cắm trại ngoài trời hòa mình với thiên nhiên dịp cuối tuần."
}

# Stopwords chuẩn sử dụng trong suốt quá trình tiền xử lý & huấn luyện
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

# =============================================================================
# 3. TẢI MÔ HÌNH & BỘ HÀM XỬ LÝ SẠCH (CLEAN CORE ENGINE)
# =============================================================================

@st.cache_resource(show_spinner="⏳ Đang nạp mô hình PhoBERT vào bộ nhớ...")
def load_engine() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy thư mục checkpoint tại {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    return tokenizer, model, device


def preprocess_text(text: Optional[str]) -> str:
    """Hàm tiền xử lý chuỗi văn bản thuần túy (Khớp 100% logic huấn luyện trong Notebook)."""
    if text is None or not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text).lower()
    # 2. Lọc bỏ URLs, HTML, Emails, ký tự đặc biệt, số
    text = re.sub(r'http\S+|www\S+|<[^>]+>|\S+@\S+|[^\w\s]|\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. Tách từ ghép tiếng Việt bằng underthesea
    try:
        from underthesea import word_tokenize
        text = word_tokenize(text, format='text')
    except Exception:
        pass
    # 4. Lọc Stop words
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)


def clear_gpu_memory():
    """Giải phóng bộ nhớ GPU an toàn."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_single(raw_text: str, tokenizer, model, device: torch.device) -> Tuple[int, np.ndarray, str]:
    """Dự đoán nhãn cho 1 bài viết đơn lẻ (Tối ưu bộ nhớ VRAM)."""
    cleaned = preprocess_text(raw_text)
    if not cleaned:
        cleaned = raw_text.strip() if isinstance(raw_text, str) else ""
        
    enc = tokenizer(cleaned, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        try:
            logits = model(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device)).logits.clone()
        except torch.cuda.OutOfMemoryError:
            clear_gpu_memory()
            logits = model(input_ids=enc['input_ids'].to('cpu'), attention_mask=enc['attention_mask'].to('cpu')).logits.clone()
            
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    pred_idx = int(np.argmax(probs))
    clear_gpu_memory()
    return pred_idx, probs, cleaned


def compute_xai_attribution(raw_text: str, tokenizer, model, device: torch.device, xai_batch_size: int = 16) -> Tuple[int, float, List[Tuple[str, float]]]:
    """Tính toán điểm số đóng góp của từng từ theo Leave-One-Out (Xử lý theo Mini-batch tránh tràn VRAM CUDA)."""
    cleaned = preprocess_text(raw_text)
    if not cleaned:
        return 0, 0.0, []
        
    words = cleaned.split()
    if not words:
        return 0, 0.0, []
        
    # Giới hạn tối đa 60 từ đầu cho XAI để đảm bảo tốc độ và an toàn bộ nhớ
    if len(words) > 60:
        words = words[:60]
        
    pred_idx, base_probs, _ = predict_single(cleaned, tokenizer, model, device)
    base_p = float(base_probs[pred_idx])
    
    # Tạo danh sách ablated texts
    ablated_texts = [" ".join([w for j, w in enumerate(words) if j != i]) for i in range(len(words))]
    probs_list: List[float] = []
    
    # Xử lý theo từng Mini-batch nhỏ để tránh CUDA Out of Memory
    with torch.no_grad():
        for i in range(0, len(ablated_texts), xai_batch_size):
            chunk = ablated_texts[i:i + xai_batch_size]
            enc_chunk = tokenizer(chunk, max_length=256, padding=True, truncation=True, return_tensors='pt')
            try:
                logits_chunk = model(
                    input_ids=enc_chunk['input_ids'].to(device), 
                    attention_mask=enc_chunk['attention_mask'].to(device)
                ).logits.clone()
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                clear_gpu_memory()
                logits_chunk = model(
                    input_ids=enc_chunk['input_ids'].to('cpu'), 
                    attention_mask=enc_chunk['attention_mask'].to('cpu')
                ).logits.clone()
                
            chunk_p = torch.softmax(logits_chunk, dim=1).cpu().numpy()[:, pred_idx]
            probs_list.extend(chunk_p.tolist())
            clear_gpu_memory()
            
    word_scores: List[Tuple[str, float]] = []
    for i, word in enumerate(words):
        drop = float(base_p - probs_list[i])
        word_scores.append((word, drop))
        
    clear_gpu_memory()
    return pred_idx, base_p, word_scores


def batch_predict_df(df: pd.DataFrame, text_col: str, tokenizer, model, device: torch.device, batch_size: int = 16, progress_bar = None) -> pd.DataFrame:
    """Xử lý phân loại hàng loạt trên GPU theo Mini-batches an toàn bộ nhớ (batch_size=16)."""
    df_res = df.copy()
    if df_res.empty:
        df_res['Chủ đề dự đoán'] = []
        df_res['Độ tin cậy (%)'] = []
        return df_res
        
    raw_texts = [str(x) if pd.notna(x) else "" for x in df_res[text_col]]
    cleaned_texts = [preprocess_text(t) or t for t in raw_texts]
    
    pred_labels: List[str] = []
    pred_confs: List[float] = []
    
    total = len(cleaned_texts)
    for i in range(0, total, batch_size):
        batch = cleaned_texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=256, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            try:
                logits = model(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device)).logits.clone()
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                clear_gpu_memory()
                logits = model(input_ids=enc['input_ids'].to('cpu'), attention_mask=enc['attention_mask'].to('cpu')).logits.clone()
                
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            indices = np.argmax(probs, axis=1)
            
            for j, idx in enumerate(indices):
                pred_labels.append(LABEL_NAMES[idx])
                pred_confs.append(round(float(probs[j, idx] * 100), 2))
                
        if progress_bar and total > 0:
            progress_bar.progress(min(1.0, (i + len(batch)) / total))
        clear_gpu_memory()
            
    df_res['Chủ đề dự đoán'] = pred_labels
    df_res['Độ tin cậy (%)'] = pred_confs
    return df_res

# =============================================================================
# 4. GIAO DIỆN CHÍNH (STREAMLIT APP LAYOUT)
# =============================================================================

device_name = "N/A"
try:
    tokenizer, model, device = load_engine()
    device_name = device.type.upper()
    is_ready = True
except Exception as e:
    is_ready = False
    err_msg = str(e)

# --- SIDEBAR: CẤU HÌNH & THÔNG TIN HỆ THỐNG ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/news.png", width=100)
    st.markdown("### 🇻🇳 NLP Master Project")
    st.caption("Đề tài: Phân loại văn bản báo chí tiếng Việt")
    st.markdown("---")
    
    st.markdown("#### 💻 Trạng thái phần cứng")
    st.markdown(f"- **Thiết bị:** `{device_name}`")
    st.markdown(f"- **Kiến trúc:** `PhoBERT-base`")
    st.markdown(f"- **Tham số:** `~135 triệu`")
    st.markdown(f"- **Từ vựng BPE:** `64,000 tokens`")
    st.markdown(f"- **Max Length:** `256 tokens`")
    
    st.markdown("---")
    st.markdown("#### 📋 11 Danh mục đề tài")
    for name, info in LABEL_INFO.items():
        st.markdown(f"- {info['icon']} **{name}**")

# --- HERO BANNER ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Hệ Thống Phân Loại Văn Bản Tiếng Việt Thông Minh</div>
    <div class="hero-subtitle">
        Ứng dụng mô hình học sâu ngôn ngữ <b>PhoBERT (VinAI)</b> kết hợp công nghệ giải thích <b>Explainable AI (XAI)</b> để tự động nhận diện và minh bạch hóa lý do phân loại 11 chuyên mục tin tức báo chí.
    </div>
</div>
""", unsafe_allow_html=True)

if not is_ready:
    st.error(f"❌ Lỗi nạp mô hình: {err_msg}")
    st.stop()

# --- CÁC TABS CHỨC NĂNG CHÍNH ---
tab1, tab2, tab3 = st.tabs([
    "🎯 1. Phân Loại Trực Tiếp", 
    "🔍 2. Explainable AI Studio", 
    "📁 3. Phân Loại File Hàng Loạt"
])

# =============================================================================
# TAB 1: PHÂN LOẠI TRỰC TIẾP
# =============================================================================
with tab1:
    if "input_text_live" not in st.session_state:
        st.session_state["input_text_live"] = SAMPLE_ARTICLES["🏢 Bất động sản - Dự án địa ốc"]

    col_l, col_r = st.columns([1.15, 1], gap="large")
    
    with col_l:
        st.markdown("#### ✍️ Nhập văn bản bài viết")
        st.caption("Bấm chọn nhanh 1 bài báo mẫu đại diện bên dưới:")
        
        # 3 hàng nút mẫu bao phủ 11 chủ đề
        row_c1 = st.columns(4)
        if row_c1[0].button("🏢 Bất động sản", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🏢 Bất động sản - Dự án địa ốc"]
            st.rerun()
        if row_c1[1].button("🔬 Công nghệ", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🔬 Công nghệ - Chip xử lý"]
            st.rerun()
        if row_c1[2].button("🏥 Sức khỏe", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🏥 Sức khỏe - Phòng ngừa dịch"]
            st.rerun()
        if row_c1[3].button("📈 Kinh doanh", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["📈 Kinh doanh - Lãi suất điều hành"]
            st.rerun()
            
        row_c2 = st.columns(4)
        if row_c2[0].button("📰 Thời sự", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["📰 Thời sự - Kỳ họp Quốc hội"]
            st.rerun()
        if row_c2[1].button("⚽ Thể thao", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["⚽ Thể thao - Chiến thắng bóng đá"]
            st.rerun()
        if row_c2[2].button("⚖️ Pháp luật", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["⚖️ Pháp luật - Tòa án tuyên án"]
            st.rerun()
        if row_c2[3].button("📚 Giáo dục", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["📚 Giáo dục - Kỳ thi tốt nghiệp"]
            st.rerun()

        row_c3 = st.columns(3)
        if row_c3[0].button("🌍 Thế giới", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🌍 Thế giới - Đối thoại ngoại giao"]
            st.rerun()
        if row_c3[1].button("🎬 Giải trí", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🎬 Giải trí - Phim điện ảnh"]
            st.rerun()
        if row_c3[2].button("🏡 Đời sống", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🏡 Đời sống - Du lịch nghỉ lễ"]
            st.rerun()

        user_content = st.text_area(
            "Nội dung bài viết (Tiêu đề hoặc đoạn văn):",
            key="input_text_live",
            height=160,
            placeholder="Dán nội dung bài báo vào đây để phân loại..."
        )
        
        c_act1, c_act2 = st.columns([1, 1])
        with c_act1:
            btn_classify = st.button("🚀 Phân loại bài viết", type="primary", use_container_width=True)
        with c_act2:
            if st.button("🗑️ Xóa trắng", use_container_width=True):
                st.session_state["input_text_live"] = ""
                st.rerun()

    with col_r:
        st.markdown("#### 🎯 Kết quả nhận diện")
        
        if user_content.strip():
            with st.spinner("Đang phân tích ngữ cảnh..."):
                t_start = time.time()
                pred_idx, probs, cleaned_str = predict_single(user_content, tokenizer, model, device)
                lat_ms = (time.time() - t_start) * 1000
                
                label_res = LABEL_NAMES[pred_idx]
                info_res = LABEL_INFO[label_res]
                conf_val = probs[pred_idx] * 100
                
                # Card kết quả nổi bật
                st.markdown(f"""
                <div class="result-card-main" style="background: {info_res['color']}15; border: 2px solid {info_res['color']};">
                    <div style="font-size: 3.2rem; margin-bottom: 0.3rem;">{info_res['icon']}</div>
                    <div style="font-size: 2rem; font-weight: 800; color: {info_res['color']};">{label_res.upper()}</div>
                    <div style="font-size: 1.15rem; color: #334155; margin-top: 0.4rem;">
                        Độ tin cậy: <b style="color: {info_res['color']}; font-size: 1.4rem;">{conf_val:.1f}%</b>
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.4rem;">
                        Tốc độ: <b>{lat_ms:.1f} ms</b> • Thiết bị: <b>{device.type.upper()}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Biểu đồ thanh ngang phân phối xác suất toàn bộ các lớp (Plotly)
                st.markdown("**Phân phối xác suất các chủ đề:**")
                df_probs = pd.DataFrame({
                    'Chủ đề': [f"{LABEL_INFO[name]['icon']} {name}" for name in LABEL_NAMES],
                    'Xác suất (%)': probs * 100,
                    'Màu': [LABEL_INFO[name]['color'] for name in LABEL_NAMES]
                })
                    
                df_probs = df_probs.sort_values(by='Xác suất (%)', ascending=True)
                
                fig_bar = px.bar(
                    df_probs, 
                    x='Xác suất (%)', 
                    y='Chủ đề', 
                    orientation='h',
                    color='Chủ đề',
                    color_discrete_sequence=df_probs['Màu'].tolist(),
                    text='Xác suất (%)'
                )
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_bar.update_layout(
                    showlegend=False,
                    height=360,
                    margin=dict(l=10, r=30, t=10, b=10),
                    xaxis=dict(range=[0, 115], title=""),
                    yaxis=dict(title="")
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                with st.expander("🔍 Xem chuỗi văn bản sau khi qua Pipeline Tiền xử lý"):
                    st.code(cleaned_str, language="text")
        else:
            st.info("👈 Hãy nhập hoặc dán nội dung bài báo vào ô bên trái để nhận diện chuyên mục.")

# =============================================================================
# TAB 2: EXPLAINABLE AI (XAI STUDIO)
# =============================================================================
with tab2:
    st.markdown("### 🔍 Explainable AI Studio (Trực Quan Hóa Lý Do Quyết Định)")
    st.write("Phương pháp **Leave-One-Out Sensitivity** đo lường chính xác mức độ sụt giảm xác suất khi từng từ bị loại bỏ khỏi câu. Những từ có điểm số đóng góp cao nhất chính là **chứng cứ cốt lõi** mà PhoBERT dựa vào để phân loại.")
    
    def update_xai_sample():
        st.session_state["ta_xai"] = SAMPLE_ARTICLES[st.session_state["sb_xai"]]

    if "ta_xai" not in st.session_state:
        st.session_state["ta_xai"] = SAMPLE_ARTICLES[list(SAMPLE_ARTICLES.keys())[0]]

    col_x1, col_x2 = st.columns([1, 1.3], gap="large")
    
    with col_x1:
        xai_sample = st.selectbox(
            "Chọn bài báo mẫu để kiểm tra XAI:",
            options=list(SAMPLE_ARTICLES.keys()),
            key="sb_xai",
            on_change=update_xai_sample
        )
        xai_text = st.text_area(
            "Văn bản cần giải thích:",
            height=140,
            key="ta_xai"
        )
        btn_run_xai = st.button("🔬 Phân Tích Bản Đồ Nhiệt XAI", type="primary", use_container_width=True)
        
    with col_x2:
        if xai_text.strip():
            with st.spinner("Đang tính toán trọng số XAI cho từng từ vị..."):
                x_pid, x_base_p, x_scores = compute_xai_attribution(xai_text, tokenizer, model, device)
                x_name = LABEL_NAMES[x_pid]
                x_info = LABEL_INFO[x_name]
                
                st.markdown(f"#### 🎯 Nhãn dự đoán: {x_info['icon']} **{x_name}** (`{x_base_p*100:.1f}%`)")
                st.markdown("##### 🌈 Bản đồ nhiệt từ khóa (Text Saliency Heatmap):")
                st.caption("Các từ được **tô màu xanh lá đậm** là các từ có đóng góp lớn nhất tới quyết định của AI.")
                
                max_score = max([s for _, s in x_scores] + [0.001])
                badges_html = []
                for word, score in x_scores:
                    if score > 0.005:
                        intensity = min(1.0, score / max_score)
                        alpha = 0.15 + 0.75 * intensity
                        badges_html.append(f'<span class="xai-badge" style="background-color: rgba(34, 197, 94, {alpha:.2f}); border: 1px solid #22C55E; color: #064E3B;" title="Đóng góp: +{score*100:.2f}%">{word} <small>(+{score*100:.1f}%)</small></span>')
                    elif score < -0.01:
                        badges_html.append(f'<span class="xai-badge" style="background-color: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; color: #7F1D1D;" title="Tác động nghịch: {score*100:.2f}%">{word}</span>')
                    else:
                        badges_html.append(f'<span class="xai-badge" style="background-color: #F1F5F9; border: 1px solid #E2E8F0; color: #475569;" title="Trung tính">{word}</span>')
                        
                st.markdown(f"""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; line-height: 2.2; margin-bottom: 1.5rem;">
                    {' '.join(badges_html)}
                </div>
                """, unsafe_allow_html=True)
                
                # Bảng xếp hạng Top từ khóa
                st.markdown("##### 📊 Top từ khóa có sức ảnh hưởng lớn nhất:")
                top_impact = sorted([x for x in x_scores if x[1] > 0], key=lambda x: x[1], reverse=True)[:8]
                if top_impact:
                    df_xai = pd.DataFrame(top_impact, columns=['Từ khóa / Cụm từ', 'Độ đóng góp (%)'])
                    df_xai['Độ đóng góp (%)'] = df_xai['Độ đóng góp (%)'].apply(lambda s: f"+{s*100:.2f}%")
                    st.dataframe(df_xai, use_container_width=True)
        else:
            st.info("Hãy nhập văn bản để hiển thị bản đồ nhiệt XAI.")

# =============================================================================
# TAB 3: PHÂN LOẠI FILE HÀNG LOẠT (BATCH ANALYTICS)
# =============================================================================
with tab3:
    st.markdown("### 📂 Phân Loại Dữ Liệu Hàng Loạt Từ File CSV / Excel")
    st.write("Tải lên tệp dữ liệu chứa danh sách bài viết. Hệ thống sử dụng GPU Batching (32 mẫu/bước) để phân loại hàng nghìn bài chỉ trong vài giây.")
    
    uploaded = st.file_uploader("Kéo thả hoặc tải lên file dữ liệu (.csv hoặc .xlsx):", type=['csv', 'xlsx'])
    
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
                
            st.success(f"✅ Đã tải file: **{len(df_raw):,} dòng**, {len(df_raw.columns)} cột.")
            st.dataframe(df_raw.head(3), use_container_width=True)
            
            c_sel1, c_sel2 = st.columns([1.5, 1])
            with c_sel1:
                target_col = st.selectbox("Chọn cột chứa văn bản cần phân loại:", options=df_raw.columns)
            with c_sel2:
                st.write("")
                st.write("")
                btn_start_batch = st.button("⚡ Bắt đầu phân loại toàn bộ", type="primary", use_container_width=True)
                
            if btn_start_batch:
                p_bar = st.progress(0)
                t0_batch = time.time()
                df_classified = batch_predict_df(
                    df_raw, target_col, tokenizer, model, device, 
                    batch_size=16, progress_bar=p_bar
                )
                t_spent = time.time() - t0_batch
                
                st.success(f"🎉 Hoàn thành phân loại **{len(df_classified):,} bài** trong **{t_spent:.2f}s** ({len(df_classified)/t_spent:.1f} bài/giây)!")
                
                # Trực quan hóa phân bổ
                col_chart1, col_chart2 = st.columns([1, 1.2])
                with col_chart1:
                    st.markdown("**Cơ cấu các chuyên mục trong tệp:**")
                    counts = df_classified['Chủ đề dự đoán'].value_counts().reset_index()
                    counts.columns = ['Chủ đề', 'Số lượng']
                    fig_pie = px.pie(counts, values='Số lượng', names='Chủ đề', hole=0.45)
                    fig_pie.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=320)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_chart2:
                    st.markdown("**Thống kê số lượng theo chủ đề:**")
                    fig_col = px.bar(counts, x='Chủ đề', y='Số lượng', color='Chủ đề')
                    fig_col.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=320)
                    st.plotly_chart(fig_col, use_container_width=True)
                    
                st.markdown("**Bảng kết quả sau gán nhãn:**")
                st.dataframe(df_classified.head(15), use_container_width=True)
                
                csv_bytes = df_classified.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button("📥 Tải xuống kết quả phân loại (CSV)", csv_bytes, "ket_qua_phan_loai.csv", "text/csv")
        except Exception as e:
            st.error(f"Lỗi xử lý file: {e}")
