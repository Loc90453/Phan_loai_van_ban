# TÀI LIỆU PHƯƠNG PHÁP GIẢI QUYẾT BÀI TOÁN
## Đề tài: Phân loại văn bản tiếng Việt tự động đa chủ đề (Vietnamese Text Classification)
**Môn học / Chuyên ngành:** Xử lý ngôn ngữ tự nhiên (NLP) — Bậc Thạc sĩ  
**Mô hình nòng cốt:** PhoBERT (State-of-the-Art Pre-trained Language Model for Vietnamese)

---

## 1. TỔNG QUAN BÀI TOÁN VÀ MỤC TIÊU

### 1.1. Phát biểu bài toán
Phân loại văn bản (Text Classification) là bài toán gán nhãn tự động một đoạn văn bản $X$ vào một hoặc nhiều danh mục định trước trong tập nhãn $C = \{c_1, c_2, ..., c_K\}$.

Trong đề tài này, bài toán được định nghĩa là **Phân loại văn bản đơn nhãn đa lớp (Multi-class Single-label Classification)** trên tập văn bản báo chí / hành chính tiếng Việt gồm **$K = 11$ chủ đề**:
1. `thoi_su` (Thời sự)
2. `the_gioi` (Thế giới)
3. `kinh_doanh` (Kinh doanh)
4. `khoa_hoc_cong_nghe` (Khoa học & Công nghệ)
5. `bat_dong_san` (Bất động sản)
6. `suc_khoe` (Sức khỏe)
7. `the_thao` (Thể thao)
8. `giai_tri` (Giải trí)
9. `phap_luat` (Pháp luật)
10. `giao_duc` (Giáo dục)
11. `doi_song` (Đời sống)

### 1.2. Thách thức đặc thù của tiếng Việt
* **Ranh giới từ phức tạp:** Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không tương ứng với dấu cách (ví dụ: *"học sinh"*, *"khoa học công nghệ"*, *"bất động sản"* là các từ ghép đơn vị nghĩa).
* **Hiện tượng đa nghĩa & đồng âm:** Từ ngữ có ngữ nghĩa thay đổi mạnh tùy vào ngữ cảnh (context).
* **Nhiễu dữ liệu:** Văn bản báo chí thu thập trên web chứa nhiều thẻ HTML, link, quảng cáo, email, ký tự đặc biệt và mã hóa Unicode không đồng nhất (NFC vs NFD).

---

## 2. QUY TRÌNH TỔNG THỂ (END-TO-END PIPELINE)

Hệ thống được thiết kế theo quy trình 6 giai đoạn tiêu chuẩn trong Machine Learning / Deep Learning NLP:

```
[Thu thập dữ liệu] 
       │ (Crawl báo chí: VnExpress, Báo Nhân Dân, Vietnamnet & HF Dataset)
       ▼
[Tiền xử lý & Chuẩn hóa]
       │ (Unicode NFC, Lowercase, Regex làm sạch, Tách từ Underthesea, Bỏ Stopwords)
       ▼
[Phân chia tập dữ liệu]
       │ (Stratified Split: 70% Train - 15% Validation - 15% Test)
       ▼
[Huấn luyện & Tinh chỉnh mô hình (Fine-tuning)]
       │ (PhoBERT + Cross-Entropy Loss + AdamW + Linear Warmup + Mixed Precision FP16)
       ▼
[Đánh giá toàn diện (Evaluation)]
       │ (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Learning Curves)
       ▼
[Triển khai & Ứng dụng thực tế]
       │ (Inference Script, REST API / Streamlit Demo Web App)
```

---

## 3. CHI TIẾT CÁC BƯỚC THỰC HIỆN

### 3.1. Thu thập và Xây dựng Tập dữ liệu
1. **Nguồn dữ liệu:**
   * Dữ liệu crawl thực tế từ các báo điện tử lớn tại Việt Nam: *VnExpress*, *Nhân Dân*, *Vietnamnet* (lưu tại `data/raw/`).
   * Bộ dữ liệu chuẩn quy mô lớn: `NamSyntax/vietnamese-news-classification` (~1.3 triệu bài báo được phân loại sẵn 11 chuyên mục).
2. **Lọc trùng lặp & Dữ liệu rác:**
   * Loại bỏ các bài trùng lặp nội dung (`drop_duplicates(subset=['text'])`).
   * Loại bỏ các văn bản quá ngắn (độ dài $\le 20$ ký tự hoặc thiếu nhãn).

---

### 3.2. Pipeline Tiền xử lý văn bản (Data Preprocessing)
Pipeline tiền xử lý gồm 6 bước tuần tự:

1. **Chuẩn hóa Unicode:** Áp dụng chuẩn **NFC (Normalization Form C)** để đồng nhất các cách gõ dấu tiếng Việt khác nhau (tổ hợp vs dựng sẵn).
2. **Chuyển chữ thường (Lowercasing):** Giảm kích thước không gian từ vựng và triệt tiêu sai khác giữa chữ hoa/chữ thường.
3. **Lọc nhiễu bằng Regex:**
   * Loại bỏ URLs: `http\S+|www\S+`
   * Loại bỏ mã HTML/thẻ định dạng: `<[^>]+>`
   * Loại bỏ địa chỉ email: `\S+@\S+`
4. **Loại bỏ ký tự đặc biệt & chữ số:** Loại bỏ emoji, ký tự lạ `[^\w\s]`, các con số `\d+` không mang giá trị ngữ nghĩa phân loại chủ đề, và rút gọn nhiều dấu cách liền nhau.
5. **Phân đoạn từ tiếng Việt (Word Segmentation):**
   * Sử dụng công cụ **`underthesea.word_tokenize(format='text')`** để nhóm các từ ghép tiếng Việt lại bằng dấu gạch dưới (ví dụ: `khoa học công nghệ` $\rightarrow$ `khoa_học công_nghệ`).
6. **Loại bỏ từ dừng (Stop words Removal):**
   * Sử dụng bộ từ dừng tiếng Việt [stopwords-vi.txt](data/stopwords_vi.txt) để loại bỏ các hư từ, từ đệm (ví dụ: *và, của, là, các, để, với, trong, đã, những...*).

---

### 3.3. Phân chia Dữ liệu (Stratified Dataset Splitting)
Dữ liệu được chia theo chiến lược **Phân tầng (Stratified Sampling)** theo tỷ lệ:
* **Train Set (70%):** Dùng để huấn luyện trọng số mô hình.
* **Validation Set (15%):** Dùng để tinh chỉnh siêu tham số, theo dõi tránh Overfitting và lưu checkpoint có `val_acc` cao nhất.
* **Test Set (15%):** Độc lập hoàn toàn, chỉ dùng để đánh giá độ chính xác cuối cùng của mô hình.

> **Đảm bảo tính khách quan:** Tập Test được giữ nguyên vẹn (`random_state=42`), không tham gia vào quá trình chọn lựa mô hình, chống rò rỉ dữ liệu (*Data Leakage*).

---

### 3.4. Mô hình Hóa: Phương pháp Tiếp cận PhoBERT

#### A. Kiến trúc PhoBERT
* **PhoBERT** (Pre-trained RoBERTa for Vietnamese) kế thừa kiến trúc Transformer Encoder đa tầng với cơ chế **Multi-Head Self-Attention**.
* Được tiền huấn luyện trên hơn 20GB văn bản tiếng Việt chất lượng cao bằng 2 tác vụ:
  * Masked Language Modeling (MLM).
  * Segment-level representation learning.
* Bộ mã hóa sử dụng **BPE (Byte-Pair Encoding)** tối ưu riêng cho cấu trúc âm tiết và từ vị tiếng Việt.

#### B. Cơ chế Phân loại (Sequence Classification Head)
* Đầu ra vector biểu diễn `[CLS]` của câu (kích thước $d = 768$) được đưa qua lớp Dropout và một Linear Layer:
  $$\mathbf{z} = \mathbf{W}_{cls} \cdot \mathbf{h}_{[CLS]} + \mathbf{b}$$
  với $\mathbf{W}_{cls} \in \mathbb{R}^{K \times 768}$, $K = 11$.
* Hàm kích hoạt **Softmax** chuyển đổi logits thành phân phối xác suất trên 11 lớp:
  $$P(y = c_i | X) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

#### C. Chiến lược Huấn luyện (Training Strategy)
1. **Hàm mất mát (Loss Function):** Cross-Entropy Loss đa lớp:
   $$\mathcal{L}_{CE} = -\sum_{i=1}^{K} y_i \log(P(y = c_i | X))$$
2. **Bộ tối ưu (Optimizer):** `AdamW` với Weight Decay giúp kiểm soát regularization tốt hơn `Adam` truyền thống.
3. **Learning Rate Scheduler:** `CosineAnnealingLR` / `Linear Warmup` giúp hội tụ mượt mà ở các epoch đầu và tinh chỉnh sâu ở các epoch cuối.
4. **Tối ưu tốc độ & Bộ nhớ:**
   * **Mixed Precision Training (PyTorch AMP - FP16):** Tăng tốc độ tính toán gấp 2–3 lần trên GPU CUDA và giảm bộ nhớ VRAM.
   * **Gradient Clipping ($max\_norm = 1.0$):** Ngăn chặn hiện tượng bùng nổ gradient (*Exploding Gradients*).
   * **Stateful Checkpointing:** Tự động lưu trạng thái `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict` cho phép resume huấn luyện khi bị ngắt quãng.

---

### 3.5. Phương pháp Đánh giá Mô hình (Evaluation Metrics)

Để đánh giá toàn diện trên cả 11 lớp, đồ án áp dụng các chỉ số:

1. **Accuracy (Độ chính xác tổng thể):**
   $$\text{Accuracy} = \frac{\text{Số mẫu dự đoán đúng}}{\text{Tổng số mẫu}}$$
2. **Precision (Độ chuẩn xác), Recall (Độ thu hồi) & F1-Score theo từng lớp:**
   $$\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}, \quad \text{Recall}_i = \frac{TP_i}{TP_i + FN_i}, \quad \text{F1}_i = 2 \cdot \frac{\text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$
3. **Weighted Average F1-score:** Tính trung bình có trọng số theo tỷ lệ phân phối thực tế của từng lớp.
4. **Confusion Matrix (Ma trận nhầm lẫn):** Phân tích trực quan các lớp dễ bị nhầm lẫn (ví dụ: bài báo giao thoa giữa *Thời sự* và *Pháp luật*, hoặc *Khoa học công nghệ* và *Giáo dục*).

---

## 4. BẢNG TỔNG HỢP CÁC SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)

| Siêu tham số | Giá trị thiết lập | Giải thích kỹ thuật |
| :--- | :--- | :--- |
| **Pre-trained Backbone** | `phobert-base` / `phobert_best` | RoBERTa architecture tối ưu cho tiếng Việt |
| **Max Sequence Length** | 256 tokens | Đủ bao phủ phần lớn câu tiêu đề + tóm tắt + mở đầu bài báo |
| **Batch Size** | 32 (hoặc 16 tùy VRAM) | Đảm bảo gradient ổn định và tận dụng song song GPU |
| **Optimizer** | `AdamW` (lr = 2e-5) | Tốc độ học chuẩn cho Transfer Learning Fine-tuning |
| **Loss Function** | Categorical Cross Entropy | Chuẩn hóa cho bài toán phân loại đa lớp rời rạc |
| **Gradient Clipping** | 1.0 | Ổn định gradient trong mạng Deep Transformers |
| **Precision Mode** | FP16 (Automatic Mixed Precision) | Tiết kiệm VRAM, tăng tốc độ xử lý |
| **Tỷ lệ Train / Val / Test** | 70% / 15% / 15% | Phân chia chuẩn phân tầng (Stratified) |

---

---

## 5. PHƯƠNG PHÁP GIẢI THÍCH MÔ HÌNH (EXPLAINABLE AI - XAI)

Để trả lời câu hỏi: *"Tại sao mô hình PhoBERT lại dự đoán bài viết này thuộc chủ đề X mà không phải chủ đề Y?"*, hệ thống tích hợp các kỹ thuật Explainable AI (XAI) chuyên biệt cho kiến trúc Transformer trong xử lý văn bản tiếng Việt:

### 5.1. Phương pháp Phân tích Độ nhạy cục bộ (Leave-One-Out Sensitivity / Occlusion)
* **Nguyên lý:** Với một câu $X = (w_1, w_2, ..., w_n)$, ta lần lượt loại bỏ từng từ / cụm từ ghép $w_i$ để tạo ra biến thể câu $X_{\setminus i}$.
* **Điểm đóng góp (Word Attribution / Importance Score):**
  $$\text{Importance}(w_i) = P(y = c_{pred} \mid X) - P(y = c_{pred} \mid X_{\setminus i})$$
* **Ý nghĩa:**
  * $\text{Importance}(w_i) > 0$: Từ $w_i$ là **bằng chứng quan trọng** ủng hộ mô hình ra quyết định gán nhãn $c_{pred}$. Điểm số càng lớn, từ đó càng mang tính quyết định (ví dụ: *"lãi suất"*, *"chứng khoán"* $\rightarrow$ *Kinh doanh*; *"tiêm vaccine"*, *"sốt xuất huyết"* $\rightarrow$ *Sức khỏe*).
  * $\text{Importance}(w_i) \approx 0$: Từ trung tính, không ảnh hưởng đến quyết định.
  * $\text{Importance}(w_i) < 0$: Từ gây nhiễu hoặc hướng mô hình sang chủ đề khác.

### 5.2. Phương pháp Gradient $\times$ Input & Integrated Gradients (IG)
* Tính đạo hàm riêng của logit lớp dự đoán đối với vector biểu diễn từ $E(w_i)$ tại tầng Embedding của PhoBERT:
  $$\text{Attribution}_{IG}(w_i) = (E(w_i) - E(x_0)) \times \int_0^1 \frac{\partial F(x_0 + \alpha(E(w_i) - E(x_0)))}{\partial E(w_i)} d\alpha$$
* Giúp khắc phục triệt để hiện tượng bão hòa gradient (*Gradient Saturation*) trong mạng Deep Transformers.

### 5.3. Trực quan hóa Bản đồ nhiệt (Text Saliency Heatmap)
* Biểu diễn trực tiếp trọng số đóng góp dưới dạng màu sắc phủ lên từng từ của bài báo trong giao diện Streamlit.
* Giúp chuyên gia / hội đồng đánh giá kiểm chứng được tính hợp lý logic (*Factual correctness & Model transparency*) của mô hình, đảm bảo mô hình không học vẹt các đặc trưng giả (*Spurious correlations*).

---

## 6. KẾT LUẬN & HƯỚNG MỞ RỘNG

### 6.1. Ưu điểm nổi bật của giải pháp
* **Tận dụng tối đa Transfer Learning:** PhoBERT mang lại khả năng biểu diễn ngữ cảnh vượt trội so với các mô hình Word2Vec, FastText hay TF-IDF truyền thống.
* **Tiền xử lý phù hợp với ngôn ngữ tiếng Việt:** Sử dụng chuẩn tách từ `underthesea` và chuẩn hóa Unicode giúp giảm thiểu tối đa hiện tượng Out-Of-Vocabulary (OOV).
* **Minh bạch và có khả năng giải thích (XAI):** Tích hợp công cụ XAI Heatmap giúp người dùng hiểu rõ căn cứ đưa ra dự đoán của mạng nơ-ron.
* **Pipeline khép kín, tối ưu:** Tích hợp cơ chế Resume Checkpoint, đánh giá đa chiều với biểu đồ trực quan, và có sẵn giao diện Web Demo Streamlit tương tác thời gian thực.

### 6.2. Hướng phát triển tiếp theo
1. Thử nghiệm các mô hình lớn hơn như **PhoBERT-large**, **ViDeBERTa** hoặc **mBART/ViT5**.
2. Áp dụng kỹ thuật **Data Augmentation** tiếng Việt (Back-translation, synonym replacement) để cân bằng các lớp có ít mẫu.
3. Mở rộng tính năng giải thích Attention Head Visualization để phân tích sâu hơn các tầng Transformer.
