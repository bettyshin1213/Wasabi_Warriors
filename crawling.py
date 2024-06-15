import requests
from bs4 import BeautifulSoup
import os

# 검색어 설정
search_term = "salmon sushi one piece"

# 이미지 저장 폴더 생성
if not os.path.exists(search_term):
    os.makedirs(search_term)

# 구글 이미지 검색 URL
base_url = "https://www.google.com/search"
params = {
    "q": search_term,
    "tbm": "isch",
    "start": 1
}

# 요청 헤더 설정
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

image_count = 0
while image_count < 100:
    # 웹페이지 요청
    response = requests.get(base_url, params=params, headers=headers)

    # BeautifulSoup을 사용하여 HTML 파싱
    soup = BeautifulSoup(response.text, "html.parser")

    # 이미지 링크 추출
    img_tags = soup.find_all("img")
    for i, img_tag in enumerate(img_tags):
        img_url = img_tag.get("src")
        if "http" in img_url:
            # 이미지 다운로드
            try:
                response = requests.get(img_url)
                with open(os.path.join(search_term, f"{image_count}.jpg"), "wb") as f:
                    f.write(response.content)
                print(f"이미지 {image_count} 다운로드 완료")
                image_count += 1
                if image_count >= 100:
                    break
            except:
                print(f"이미지 {image_count} 다운로드 실패")
                continue

    # 다음 페이지로 이동
    params["start"] += 20