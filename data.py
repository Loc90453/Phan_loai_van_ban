# File: src/data_collection/crawler_vnexpress.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# Mapping chuyên mục VnExpress sang nhãn
CATEGORY_MAP = {
    "https://vnexpress.net/chinh-tri": "chinh_tri",
    "https://vnexpress.net/kinh-doanh": "kinh_te",
    "https://vnexpress.net/the-thao": "the_thao",
    "https://vnexpress.net/suc-khoe": "y_te",
    "https://vnexpress.net/giao-duc": "giao_duc",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_article_links(category_url, max_pages=5):
    """Lấy danh sách link bài viết từ một chuyên mục."""
    links = []
    for page in range(1, max_pages + 1):
        url = f"{category_url}-p{page}" if page > 1 else category_url
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            # Tìm tất cả link bài báo
            articles = soup.select("h3.title-news a, h2.title-news a")
            for a in articles:
                href = a.get("href", "")
                if href.startswith("https://vnexpress.net/"):
                    links.append(href)
            time.sleep(random.uniform(1.5, 3.0))  # Lịch sự với server
        except Exception as e:
            print(f"Lỗi trang {page}: {e}")
    return list(set(links))  # Bỏ trùng

def scrape_article(url, label):
    """Lấy nội dung một bài viết."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        title = soup.select_one("h1.title-detail")
        title = title.get_text(strip=True) if title else ""
        
        # Lấy đoạn mô tả
        description = soup.select_one("p.description")
        description = description.get_text(strip=True) if description else ""
        
        # Lấy nội dung chính
        paragraphs = soup.select("article.fck_detail p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)
        
        # Ghép title + mô tả + nội dung
        full_text = f"{title}. {description}. {content}"
        
        return {
            "url": url,
            "title": title,
            "content": full_text,
            "label": label,
            "source": "vnexpress"
        }
    except Exception as e:
        print(f"Lỗi bài {url}: {e}")
        return None

def crawl_all(output_path="data/raw/vnexpress_raw.csv"):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_data = []
    
    for category_url, label in CATEGORY_MAP.items():
        print(f"\n[+] Đang crawl: {label}")
        links = get_article_links(category_url, max_pages=10)
        print(f"    Tìm thấy {len(links)} bài")
        
        for i, link in enumerate(links[:500]):  # Tối đa 500 bài/nhãn
            article = scrape_article(link, label)
            if article:
                all_data.append(article)
            if i % 50 == 0:
                print(f"    Đã xử lý {i}/{len(links)}")
            time.sleep(random.uniform(0.5, 1.5))
    
    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=["url"], inplace=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu {len(df)} bài vào {output_path}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    crawl_all()