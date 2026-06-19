# File: src/preprocessing/text_processor.py
import re
import unicodedata
import pandas as pd
from underthesea import word_tokenize

# === BƯỚC 1: Load stop words tiếng Việt ===
def load_stopwords(path="data/vietnamese_stopwords-vi.txt"):

    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

STOPWORDS = load_stopwords()s

# === BƯỚC 2: Chuẩn hóa Unicode ===
def normalize_unicode(text):
    """
    Chuyển về dạng NFC chuẩn.
    Ví dụ: 'tiê\u0301ng' -> 'tiếng' (cùng nhìn nhưng khác bytes)
    """
    return unicodedata.normalize("NFC", text)

# === BƯỚC 3: Chuyển chữ thường ===
def to_lowercase(text):
    return text.lower()

# === BƯỚC 4: Bỏ ký tự đặc biệt, giữ chữ cái tiếng Việt ===
def remove_special_chars(text):
    """
    Giữ lại: chữ cái a-z, ký tự có dấu tiếng Việt, khoảng trắng.
    Bỏ: số, ký tự đặc biệt, HTML tags, URLs.
    """
    # Bỏ URL
    text = re.sub(r"http\S+|www\S+", "", text)
    # Bỏ HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Bỏ email
    text = re.sub(r"\S+@\S+", "", text)
    # Giữ chữ cái và khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)          # bỏ dấu câu
    text = re.sub(r"\d+", " ", text)               # bỏ số
    text = re.sub(r"\s+", " ", text).strip()       # chuẩn hóa khoảng trắng
    return text

# === BƯỚC 5: Tách từ tiếng Việt (Underthesea) ===
def tokenize_vietnamese(text):
    """
    Underthesea tách từ ghép: 'học sinh' -> 'học_sinh'
    Điều này giúp mô hình hiểu đúng từ ghép tiếng Việt.
    
    Ví dụ:
      Input:  "Bộ trưởng Giáo dục họp hội đồng"
      Output: "bộ_trưởng giáo_dục họp hội_đồng"
    """
    tokens = word_tokenize(text, format="text")
    return tokens

# === BƯỚC 6: Bỏ stop words ===
def remove_stopwords(text):
    """
    Bỏ các từ không mang ý nghĩa phân biệt chủ đề.
    Ví dụ: 'và', 'của', 'là', 'các', 'để'...
    """
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(filtered)

# === Pipeline hoàn chỉnh ===
def preprocess(text):
    """
    Áp dụng toàn bộ pipeline theo thứ tự:
    Unicode → Lowercase → Tách từ → Bỏ ký tự đặc biệt → Bỏ stop words
    """
    text = normalize_unicode(text)
    text = to_lowercase(text)
    text = remove_special_chars(text)
    text = tokenize_vietnamese(text)       # Tách từ TRƯỚC khi bỏ stopwords
    text = remove_stopwords(text)
    return text

# === Xử lý toàn bộ dataset ===
def preprocess_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    print(f"Đọc {len(df)} bản ghi")
    
    # Áp dụng preprocessing
    df["text_clean"] = df["content"].fillna("").apply(preprocess)
    
    # Bỏ dòng rỗng sau xử lý
    df = df[df["text_clean"].str.len() > 20]
    
    # Encode nhãn thành số
    label_map = {
        "chinh_tri": 0, "kinh_te": 1,
        "the_thao": 2, "y_te": 3, "giao_duc": 4
    }
    df["label_id"] = df["label"].map(label_map)
    
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Lưu {len(df)} bản ghi sạch vào {output_csv}")
    print("\nPhân phối nhãn:")
    print(df["label"].value_counts())
    return df

# Test thử
if __name__ == "__main__":
    text_mau = "Bộ trưởng Bộ Giáo dục và Đào tạo vừa ký quyết định điều chỉnh!"
    print("Input: ", text_mau)
    print("Output:", preprocess(text_mau))
    # Output: "bộ_trưởng bộ_giáo_dục đào_tạo ký quyết_định điều_chỉnh"