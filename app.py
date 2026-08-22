"""
=============================================================================
HỆ THỐNG PHÂN LOẠI VĂN BẢN TIẾNG VIỆT ĐA CHỦ ĐỀ & GIẢI THÍCH MÔ HÌNH (PHOBERT + XAI)
Đề tài: Xử lý ngôn ngữ tự nhiên (NLP) — Bậc Thạc sĩ
Mô hình nòng cốt: PhoBERT (VinAI Research) + Explainable AI (Leave-One-Out Attribution)
Thiết kế giao diện: Modern SaaS UI / Clean Minimalist Dashboard (Tích hợp XAI trực tiếp)
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
import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =============================================================================
# 1. CẤU HÌNH TRANG VÀ THIẾT KẾ GIAO DIỆN HIỆN ĐẠI (PREMIUM MINIMALIST UI)
# =============================================================================
st.set_page_config(
    page_title="VietText AI - Phân Loại Văn Bản Tiếng Việt",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background tổng thể */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Hero Header gọn gàng & sang trọng */
    .header-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #1E3A8A 100%);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.15);
    }
    .header-badge {
        display: inline-block;
        background: rgba(96, 165, 250, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(147, 197, 253, 0.3);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .header-title {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 60%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-desc {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }
    
    /* Khối thẻ giao diện (Clean Card) */
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.3rem;
        box-shadow: 0 2px 10px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
    }
    
    /* Thanh chuyển Tab hiện đại dạng Pill */
    div[data-baseweb="tab-list"] {
        background: #E2E8F0;
        padding: 5px;
        border-radius: 12px;
        gap: 6px;
        margin-bottom: 1.3rem;
    }
    button[data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
        font-size: 0.92rem !important;
    }
    button[aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }
    
    /* Nút bấm tinh gọn */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.45rem 0.9rem;
        transition: all 0.2s ease;
        border: 1px solid #E2E8F0;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    
    /* Nút Primary */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    
    /* Thẻ kết quả dự đoán (Result Badge Card) */
    .prediction-card {
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        background: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    
    /* Badge từ khóa XAI */
    .xai-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        margin: 3px 3px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.86rem;
        transition: all 0.15s ease;
        cursor: default;
    }
    .xai-pill:hover {
        transform: translateY(-2px);
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
    'Thời sự': {'id': 0, 'icon': '📰', 'color': '#2563EB', 'desc': 'Chính trị, điều hành chính phủ, chính sách vĩ mô'},
    'Thế giới': {'id': 1, 'icon': '🌍', 'color': '#0284C7', 'desc': 'Ngoại giao quốc tế, xung đột, kinh tế toàn cầu'},
    'Kinh doanh': {'id': 2, 'icon': '📈', 'color': '#059669', 'desc': 'Tài chính, chứng khoán, ngân hàng, doanh nghiệp'},
    'Khoa học công nghệ': {'id': 3, 'icon': '🔬', 'color': '#7C3AED', 'desc': 'Trí tuệ nhân tạo, phần mềm, thiết bị di động'},
    'Bất động sản': {'id': 4, 'icon': '🏢', 'color': '#D97706', 'desc': 'Thị trường nhà đất, chung cư, quy hoạch địa ốc'},
    'Sức khỏe': {'id': 5, 'icon': '🏥', 'color': '#DC2626', 'desc': 'Y tế, phòng chống dịch bệnh, vaccine, dinh dưỡng'},
    'Thể thao': {'id': 6, 'icon': '⚽', 'color': '#16A34A', 'desc': 'Bóng đá, giải đấu quốc tế, vận động viên'},
    'Giải trí': {'id': 7, 'icon': '🎬', 'color': '#DB2777', 'desc': 'Điện ảnh, âm nhạc, nghệ sĩ, truyền hình thực tế'},
    'Pháp luật': {'id': 8, 'icon': '⚖️', 'color': '#475569', 'desc': 'Xét xử tòa án, khởi tố vụ án, an ninh trật tự'},
    'Giáo dục': {'id': 9, 'icon': '📚', 'color': '#EA580C', 'desc': 'Tuyển sinh, kỳ thi THPT, trường đại học'},
    'Đời sống': {'id': 10, 'icon': '🏡', 'color': '#0D9488', 'desc': 'Gia đình, du lịch sinh thái, ẩm thực, phong cách sống'}
}

SAMPLE_ARTICLES: Dict[str, str] = {
    "🔬 Công nghệ": "Apple chính thức công bố chip xử lý thế hệ mới với kiến trúc bán dẫn 3nm mang lại hiệu năng tính toán AI vượt trội trên thiết bị di động.",
    "🏢 Bất động sản": "Giám đốc Công an tỉnh đề nghị luật hóa việc cấm quảng cáo quá mức thực tế sai sự thật về dự án bất động sản trước khi bán cho khách hàng.",
    "🏥 Sức khỏe": "Bộ Y tế khuyến cáo người dân chủ động tiêm vaccine phòng bệnh mùa đông xuân và chủ động tiêu diệt bọ gậy để phòng sốt xuất huyết.",
    "📈 Kinh doanh": "Ngân hàng Nhà nước giảm lãi suất điều hành nhằm hỗ trợ các doanh nghiệp tiếp cận nguồn vốn vay phục hồi sản xuất và kinh doanh.",
    "📰 Thời sự": "Chủ tịch Quốc hội chủ trì phiên họp toàn thể thảo luận về dự thảo các luật trọng điểm phục vụ phát triển kinh tế xã hội đất nước.",
    "⚽ Thể thao": "Đội tuyển U23 Việt Nam đã có màn trình diễn xuất sắc và giành chiến thắng thuyết phục 3-0 trong trận đấu ra quân tại giải vô địch châu Á.",
    "⚖️ Pháp luật": "Tòa án nhân dân cấp cao mở phiên tòa xét xử phúc thẩm và tuyên phạt bị cáo 15 năm tù về tội lừa đảo chiếm đoạt tài sản qua mạng.",
    "📚 Giáo dục": "Bộ Giáo dục và Đào tạo công bố cấu trúc định dạng đề thi tham khảo tốt nghiệp THPT và hướng dẫn tuyển sinh đại học năm nay.",
    "🌍 Thế giới": "Hội đồng Bảo an Liên Hợp Quốc thông qua nghị quyết kêu gọi các bên xung đột lập tức ngừng bắn và nối lại đàm phán hòa bình.",
    "🎬 Giải trí": "Bộ phim điện ảnh mới của đạo diễn Việt Nam đã chính thức vượt mốc doanh thu 200 tỷ đồng sau hai tuần công chiếu tại các rạp.",
    "🏡 Đời sống": "Nhiều gia đình tại các thành phố lớn lựa chọn kỳ nghỉ du lịch sinh thái và cắm trại ngoài trời hòa mình với thiên nhiên dịp cuối tuần."
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

# =============================================================================
# 3. TẢI MÔ HÌNH & BỘ HÀM XỬ LÝ (CORE ENGINE)
# =============================================================================

@st.cache_resource(show_spinner="⏳ Đang khởi tạo mô hình PhoBERT...")
def load_engine() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    return tokenizer, model, device


def preprocess_text(text: Optional[str]) -> str:
    """Pipeline tiền xử lý 6 bước đồng bộ."""
    if text is None or not isinstance(text, str) or len(text.strip()) == 0:
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


def crawl_news_article(url: str) -> Tuple[str, str, str]:
    """Cào tiêu đề và nội dung bài báo từ URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    resp = requests.get(url.strip(), headers=headers, timeout=10)
    resp.encoding = resp.apparent_encoding or 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    title = ""
    h1 = soup.find('h1')
    if h1:
        title = h1.get_text(strip=True)
    elif soup.find('title'):
        title = soup.find('title').get_text(strip=True)
        
    description = ""
    desc_meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
    if desc_meta and desc_meta.get('content'):
        description = desc_meta['content'].strip()
        
    paragraphs = []
    article_body = soup.find('article') or soup.find('div', class_=re.compile(r'content|detail|body|article|main', re.I)) or soup
    for p in article_body.find_all('p'):
        p_text = p.get_text(strip=True)
        if len(p_text) > 25 and not re.search(r'ảnh:|nguồn:|theo|bản quyền|hotline|liên hệ|quảng cáo', p_text, re.I):
            paragraphs.append(p_text)
            
    body_text = " ".join(paragraphs[:8])
    full_content = f"{title}. {description}. {body_text}".strip()
    return title, description, full_content


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_single(raw_text: str, tokenizer, model, device: torch.device) -> Tuple[int, np.ndarray, str]:
    """Dự đoán đơn lẻ tối ưu bộ nhớ."""
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
    """Tính điểm XAI chuẩn hóa phân bổ tương đối (Relative Attribution)."""
    cleaned = preprocess_text(raw_text)
    if not cleaned:
        return 0, 0.0, []
        
    words = cleaned.split()
    if not words:
        return 0, 0.0, []
        
    if len(words) > 50:
        words = words[:50]
        
    pred_idx, base_probs, _ = predict_single(cleaned, tokenizer, model, device)
    base_p = float(base_probs[pred_idx])
    
    ablated_texts = [" ".join([w for j, w in enumerate(words) if j != i]) for i in range(len(words))]
    probs_list: List[float] = []
    
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


def render_probability_chart(probs: np.ndarray) -> go.Figure:
    """Vẽ biểu đồ phân phối xác suất tối giản, hiện đại."""
    df_probs = pd.DataFrame({
        'Chủ đề': [f"{LABEL_INFO[name]['icon']} {name}" for name in LABEL_NAMES],
        'Xác suất (%)': probs * 100,
        'Màu': [LABEL_INFO[name]['color'] for name in LABEL_NAMES]
    }).sort_values(by='Xác suất (%)', ascending=True)
    
    fig = px.bar(
        df_probs, 
        x='Xác suất (%)', 
        y='Chủ đề', 
        orientation='h',
        color='Chủ đề',
        color_discrete_sequence=df_probs['Màu'].tolist(),
        text='Xác suất (%)'
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%', 
        textposition='outside',
        marker=dict(line=dict(width=0))
    )
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#F1F5F9", title=""),
        yaxis=dict(title=""),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def render_xai_visualization(raw_text: str, tokenizer, model, device: torch.device):
    """Component hiển thị bản đồ nhiệt XAI & Top từ khóa đóng góp tích hợp trực tiếp."""
    st.markdown("---")
    st.markdown("##### 🌈 Bản đồ nhiệt Explainable AI (Lý do mô hình quyết định)")
    st.caption("Các từ được **tô màu xanh lục nổi bật** là chứng cứ then chốt giúp mô hình đưa ra dự đoán này:")
    
    with st.spinner("Đang tính toán trọng số XAI cho từng từ vị..."):
        x_pid, x_base_p, x_scores = compute_xai_attribution(raw_text, tokenizer, model, device)
        
        # Chuẩn hóa cường độ màu tương đối
        pos_scores = [s for _, s in x_scores if s > 0]
        max_pos = max(pos_scores) if pos_scores else 1.0
        min_pos = min(pos_scores) if pos_scores else 0.0
        range_pos = max(max_pos - min_pos, 1e-7)
        
        badges_html = []
        for word, score in x_scores:
            if score > 0:
                norm = (score - min_pos) / range_pos
                alpha = 0.20 + 0.70 * norm
                badges_html.append(
                    f'<span class="xai-pill" style="background: rgba(16, 185, 129, {alpha:.2f}); border: 1px solid #10B981; color: #064E3B;" title="Đóng góp: +{score*100:.2f}%">'
                    f'{word} <small style="font-size:0.72rem; margin-left:3px; opacity:0.85;">+{score*100:.2f}%</small></span>'
                )
            elif score < -0.005:
                badges_html.append(
                    f'<span class="xai-pill" style="background: #FFF1F2; border: 1px solid #FECDD3; color: #9F1239;" title="Tác động nghịch: {score*100:.2f}%">'
                    f'{word}</span>'
                )
            else:
                badges_html.append(
                    f'<span class="xai-pill" style="background: #F8FAFC; border: 1px solid #E2E8F0; color: #64748B;" title="Trung tính">'
                    f'{word}</span>'
                )
                
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.9rem; line-height: 2.1; margin-bottom: 0.9rem;">
            {' '.join(badges_html)}
        </div>
        """, unsafe_allow_html=True)
        
        # Bảng Top từ khóa có sức ảnh hưởng
        top_impact = sorted([x for x in x_scores if x[1] > 0], key=lambda x: x[1], reverse=True)[:6]
        if top_impact:
            st.markdown("**🏆 Top từ khóa mang tính quyết định nhất:**")
            cols_top = st.columns(min(len(top_impact), 3))
            for idx_t, (kw, sc) in enumerate(top_impact):
                c_idx = idx_t % len(cols_top)
                with cols_top[c_idx]:
                    st.markdown(f"""
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 6px 10px; margin-bottom: 6px;">
                        <div style="font-size: 0.72rem; color: #166534; font-weight: 600;">TOP #{idx_t+1}</div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: #14532D;">{kw}</div>
                        <div style="font-size: 0.78rem; color: #15803D;">Độ đóng góp: <b>+{sc*100:.2f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)


def batch_predict_df(df: pd.DataFrame, text_col: str, tokenizer, model, device: torch.device, batch_size: int = 16, progress_bar = None) -> pd.DataFrame:
    """Xử lý phân loại hàng loạt trên GPU."""
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
# 4. GIAO DIỆN ỨNG DỤNG (STREAMLIT APP LAYOUT)
# =============================================================================

device_name = "N/A"
try:
    tokenizer, model, device = load_engine()
    device_name = device.type.upper()
    is_ready = True
except Exception as e:
    is_ready = False
    err_msg = str(e)

# --- SIDEBAR GỌN GÀNG ---
with st.sidebar:
    st.markdown("### 🇻🇳 NLP Master Project")
    st.caption("Phân loại văn bản báo chí tiếng Việt")
    
    st.markdown(f"""
    <div style="background: #F1F5F9; border-radius: 10px; padding: 10px 12px; margin: 10px 0 15px 0;">
        <div style="font-size: 0.8rem; color: #64748B;">MÔ HÌNH NÒNG CỐT</div>
        <div style="font-weight: 700; color: #1E293B; font-size: 0.95rem;">PhoBERT-base</div>
        <div style="font-size: 0.78rem; color: #475569; margin-top: 4px;">
            Thiết bị: <b style="color: #2563EB;">{device_name}</b> • Vô số: <b>135M params</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🏷️ 11 Chủ đề phân loại")
    for name, info in LABEL_INFO.items():
        st.markdown(f"<span style='font-size:0.86rem;'>{info['icon']} <b>{name}</b></span>", unsafe_allow_html=True)

# --- HEADER CARD ---
st.markdown("""
<div class="header-card">
    <div>
        <div class="header-badge">NLP MASTER PROJECT • PHOBERT AI + XAI</div>
        <div class="header-title">Hệ Thống Phân Loại Văn Bản Tiếng Việt</div>
        <div class="header-desc">Nhận diện tự động 11 chuyên mục tin tức báo chí và trực quan hóa lý do quyết định bằng Explainable AI.</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not is_ready:
    st.error(f"❌ Lỗi nạp mô hình: {err_msg}")
    st.stop()

# --- TABS GIAO DIỆN (3 TABS TINH GỌN, TÍCH HỢP XAI TRỰC TIẾP) ---
tab1, tab2, tab3 = st.tabs([
    "🎯 1. Phân Loại Trực Tiếp (Văn Bản)", 
    "🌐 2. Phân Loại Từ Link Báo (URL)", 
    "📁 3. Phân Loại File Hàng Loạt"
])

# =============================================================================
# TAB 1: PHÂN LOẠI TRỰC TIẾP (TÍCH HỢP XAI)
# =============================================================================
with tab1:
    if "input_text_live" not in st.session_state:
        st.session_state["input_text_live"] = SAMPLE_ARTICLES["🏢 Bất động sản"]

    col_l, col_r = st.columns([1.05, 1], gap="medium")
    
    with col_l:
        st.markdown("**Chọn nhanh bài báo mẫu:**")
        cols_chip1 = st.columns(4)
        if cols_chip1[0].button("🔬 Công nghệ", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🔬 Công nghệ"]
            st.rerun()
        if cols_chip1[1].button("🏢 BĐS", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🏢 Bất động sản"]
            st.rerun()
        if cols_chip1[2].button("📈 Kinh doanh", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["📈 Kinh doanh"]
            st.rerun()
        if cols_chip1[3].button("🏥 Sức khỏe", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["🏥 Sức khỏe"]
            st.rerun()
            
        cols_chip2 = st.columns(4)
        if cols_chip2[0].button("📰 Thời sự", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["📰 Thời sự"]
            st.rerun()
        if cols_chip2[1].button("⚽ Thể thao", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["⚽ Thể thao"]
            st.rerun()
        if cols_chip2[2].button("⚖️ Pháp luật", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["⚖️ Pháp luật"]
            st.rerun()
        if cols_chip2[3].button("📚 Giáo dục", use_container_width=True):
            st.session_state["input_text_live"] = SAMPLE_ARTICLES["📚 Giáo dục"]
            st.rerun()

        user_content = st.text_area(
            "Nội dung văn bản cần phân loại:",
            key="input_text_live",
            height=140,
            placeholder="Nhập hoặc dán nội dung bài báo vào đây..."
        )
        
        c_b1, c_b2 = st.columns([1, 1])
        with c_b1:
            st.button("🚀 Phân loại ngay", type="primary", use_container_width=True)
        with c_b2:
            if st.button("🗑️ Xóa trắng", use_container_width=True):
                st.session_state["input_text_live"] = ""
                st.rerun()

    with col_r:
        if user_content.strip():
            with st.spinner("Đang nhận diện..."):
                t_start = time.time()
                pred_idx, probs, cleaned_str = predict_single(user_content, tokenizer, model, device)
                lat_ms = (time.time() - t_start) * 1000
                
                label_res = LABEL_NAMES[pred_idx]
                info_res = LABEL_INFO[label_res]
                conf_val = probs[pred_idx] * 100
                
                # Card kết quả
                st.markdown(f"""
                <div class="prediction-card" style="border-left: 5px solid {info_res['color']};">
                    <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">{info_res['icon']}</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: {info_res['color']};">{label_res}</div>
                    <div style="font-size: 0.95rem; color: #475569; margin-top: 0.3rem;">
                        Độ tin cậy: <b style="color: {info_res['color']}; font-size: 1.15rem;">{conf_val:.1f}%</b> • Độ trễ: <b>{lat_ms:.1f} ms</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**📊 Phân phối xác suất 11 chủ đề:**")
                st.plotly_chart(render_probability_chart(probs), use_container_width=True)
                
                with st.expander("🔍 Chuỗi văn bản sau tiền xử lý"):
                    st.code(cleaned_str, language="text")
        else:
            st.info("👈 Dán văn bản vào ô bên trái để phân loại.")

    # --- PHẦN XAI STUDIO TÍCH HỢP TRỰC TIẾP DƯỚI TAB 1 ---
    if user_content.strip():
        render_xai_visualization(user_content, tokenizer, model, device)

# =============================================================================
# TAB 2: PHÂN LOẠI TỪ URL BÁO CHÍ (TÍCH HỢP XAI)
# =============================================================================
with tab2:
    st.markdown("#### 🌐 Cào và phân loại tự động từ đường link bài báo")
    st.caption("Hỗ trợ hầu hết các trang báo lớn: VnExpress, Dân Trí, Tuổi Trẻ, VietnamNet, Nhân Dân, Thanh Niên...")
    
    col_u1, col_u2 = st.columns([1.05, 1], gap="medium")
    
    with col_u1:
        input_url = st.text_input(
            "Nhập link bài báo (URL):",
            placeholder="https://vnexpress.net/..."
        )
        btn_crawl = st.button("🚀 Cào & Phân loại bài viết", type="primary", use_container_width=True)
        
    c_full_for_xai = ""
    with col_u2:
        if btn_crawl and input_url.strip():
            with st.spinner("⏳ Đang cào dữ liệu và phân tích..."):
                try:
                    c_title, c_desc, c_full = crawl_news_article(input_url)
                    if not c_full or len(c_full) < 15:
                        st.warning("⚠️ Không thể trích xuất nội dung từ link này. Vui lòng kiểm tra lại URL.")
                    else:
                        st.session_state["last_crawled_content"] = c_full
                        st.success("✅ Cào dữ liệu thành công!")
                        with st.expander("📰 Tiêu đề & nội dung đã cào", expanded=True):
                            st.markdown(f"**{c_title}**")
                            if c_desc:
                                st.caption(c_desc)
                                
                        p_idx, p_probs, p_cleaned = predict_single(c_full, tokenizer, model, device)
                        p_name = LABEL_NAMES[p_idx]
                        p_info = LABEL_INFO[p_name]
                        p_conf = p_probs[p_idx] * 100
                        
                        st.markdown(f"""
                        <div class="prediction-card" style="border-left: 5px solid {p_info['color']}; margin-top: 10px;">
                            <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">{p_info['icon']}</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: {p_info['color']};">{p_name}</div>
                            <div style="font-size: 0.95rem; color: #475569; margin-top: 0.3rem;">
                                Độ tin cậy: <b style="color: {p_info['color']}; font-size: 1.15rem;">{p_conf:.1f}%</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**📊 Phân phối xác suất 11 chủ đề:**")
                        st.plotly_chart(render_probability_chart(p_probs), use_container_width=True)
                except Exception as ex:
                    st.error(f"❌ Lỗi khi truy cập link: {ex}")
        elif not input_url.strip():
            st.info("👈 Dán đường link bài báo vào ô bên trái để nhận diện chuyên mục.")

    # --- PHẦN XAI STUDIO TÍCH HỢP TRỰC TIẾP DƯỚI TAB 2 ---
    if "last_crawled_content" in st.session_state and st.session_state["last_crawled_content"]:
        render_xai_visualization(st.session_state["last_crawled_content"], tokenizer, model, device)

# =============================================================================
# TAB 3: PHÂN LOẠI FILE HÀNG LOẠT
# =============================================================================
with tab3:
    st.markdown("#### 📂 Phân Loại Hàng Loạt Từ File CSV / Excel")
    st.caption("Tải tệp dữ liệu lên để phân loại đồng thời hàng nghìn bài viết bằng GPU Batching.")
    
    uploaded = st.file_uploader("Kéo thả hoặc tải lên file dữ liệu (.csv hoặc .xlsx):", type=['csv', 'xlsx'])
    
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.success(f"✅ Đã tải file: **{len(df_raw):,} dòng**, {len(df_raw.columns)} cột.")
            st.dataframe(df_raw.head(3), use_container_width=True)
            
            c_sel1, c_sel2 = st.columns([1.5, 1])
            with c_sel1:
                target_col = st.selectbox("Chọn cột chứa văn bản cần phân loại:", options=df_raw.columns)
            with c_sel2:
                st.write("")
                st.write("")
                btn_start_batch = st.button("⚡ Bắt đầu phân loại", type="primary", use_container_width=True)
                
            if btn_start_batch:
                p_bar = st.progress(0)
                t0_batch = time.time()
                df_classified = batch_predict_df(
                    df_raw, target_col, tokenizer, model, device, 
                    batch_size=16, progress_bar=p_bar
                )
                t_spent = time.time() - t0_batch
                
                st.success(f"🎉 Hoàn thành phân loại **{len(df_classified):,} bài** trong **{t_spent:.2f}s** ({len(df_classified)/t_spent:.1f} bài/giây)!")
                
                col_chart1, col_chart2 = st.columns([1, 1.2])
                with col_chart1:
                    st.markdown("**Cơ cấu các chuyên mục:**")
                    counts = df_classified['Chủ đề dự đoán'].value_counts().reset_index()
                    counts.columns = ['Chủ đề', 'Số lượng']
                    fig_pie = px.pie(counts, values='Số lượng', names='Chủ đề', hole=0.45)
                    fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_chart2:
                    st.markdown("**Thống kê số lượng:**")
                    fig_col = px.bar(counts, x='Chủ đề', y='Số lượng', color='Chủ đề')
                    fig_col.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=280)
                    st.plotly_chart(fig_col, use_container_width=True)
                    
                st.dataframe(df_classified.head(10), use_container_width=True)
                csv_bytes = df_classified.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button("📥 Tải xuống kết quả (CSV)", csv_bytes, "ket_qua_phan_loai.csv", "text/csv")
        except Exception as e:
            st.error(f"Lỗi xử lý file: {e}")
