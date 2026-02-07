Image Preprocessing Project
📖 프로젝트 개요

본 프로젝트는 Hugging Face의 Food101 데이터셋을 활용하여
AI 학습을 위한 이미지 전처리 과정을 구현한 것이다.

랜덤으로 선택된 이미지를 대상으로 다음 과정을 수행하였다:

크기 통일 (Resize)

이상치 제거

흑백 변환 (Grayscale)

정규화 (Normalization)

노이즈 제거 (Blur)

전처리된 이미지 5장 저장

📂 사용 데이터셋

Dataset: ethz/food101

Split: train

Hugging Face datasets 라이브러리를 사용하여 로드

dataset = load_dataset("ethz/food101", split="train")
dataset = dataset.shuffle(seed=random.randint(0, 10000))


데이터는 완전 랜덤 셔플 후 순차적으로 검사하여
전처리 조건을 만족하는 이미지 5장을 저장하였다.

⚙ 전처리 과정
1️⃣ 이미지 크기 조정 (Resize)
image = cv2.resize(image, (224, 224))


모든 이미지를 224×224 크기로 통일

딥러닝 모델 입력 크기 표준화

연산 효율 향상

2️⃣ 이상치 제거 (Outlier Filtering)
✔ (1) 너무 어두운 이미지 제거
np.mean(gray) < threshold


이미지 평균 밝기 계산

기준값(기본 50)보다 낮으면 제거

학습에 부적절한 이미지 필터링

✔ (2) 객체 크기가 너무 작은 이미지 제거
white_pixels / total_pixels < min_area_ratio


이진화 후 객체 비율 계산

전체 픽셀 대비 객체 비율이 5% 미만이면 제거

정보가 부족한 이미지 제거

3️⃣ 흑백 변환 (Grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


색상 정보를 제거하여 계산량 감소

형태 및 구조 중심 학습 가능

4️⃣ 정규화 (Normalization)
normalized = gray / 255.0


픽셀 값을 0~255 → 0~1 범위로 변환

학습 안정성 향상

모델 수렴 속도 개선

5️⃣ 노이즈 제거 (Gaussian Blur)
blurred = cv2.GaussianBlur(normalized, (5, 5), 0)


작은 잡음 제거

불필요한 세부 요소 완화

특징 추출 안정화

6️⃣ 이미지 저장

전처리를 통과한 이미지 중
랜덤하게 5장을 저장하였다.

cv2.imwrite("preprocessed_samples/random_sample_X.jpg", processed)


저장 경로:

preprocessed_samples/

🎯 전처리 목적

데이터 품질 향상

불필요한 이상치 제거

모델 학습 안정성 확보

연산 효율 개선

전처리는 AI 모델 성능에 직접적인 영향을 미치는
핵심 단계이다.

📦 실행 방법
1️⃣ 필요 패키지 설치
pip install opencv-python numpy datasets

2️⃣ 실행
python image_preprocessing.py


실행 후 preprocessed_samples 폴더에
전처리된 이미지 5장이 저장된다.


