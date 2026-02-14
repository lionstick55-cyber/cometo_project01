import os
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation

# -----------------------------
# 1. 이미지 폴더 설정
# -----------------------------
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "images"

# -----------------------------
# 2. 이미지 이름 입력받기
# -----------------------------
image_name = input("images 폴더 안에 있는 이미지 파일 이름을 입력하세요: ")
image_path = os.path.join(IMAGE_FOLDER, image_name)

if not os.path.exists(image_path):
    print("이미지 파일이 존재하지 않습니다.")
    print("현재 찾는 경로:", os.path.abspath(image_path))
    exit()

# -----------------------------
# 3. 이미지 로드
# -----------------------------
image = Image.open(image_path).convert("RGB")

# -----------------------------
# 4. 모델 로드 (최초 1회 다운로드됨)
# -----------------------------
model_name = "Intel/dpt-large"
image_processor = DPTImageProcessor.from_pretrained(model_name)
model = DPTForDepthEstimation.from_pretrained(model_name)

# -----------------------------
# 5. Depth 추론
# -----------------------------
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# -----------------------------
# 6. Depth 정규화
# -----------------------------
depth = predicted_depth.squeeze().cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min())
depth = (depth * 255).astype("uint8")

# -----------------------------
# 7. 컬러맵 적용
# -----------------------------
depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

# -----------------------------
# 8. 저장
# -----------------------------
output_name = f"depth_{image_name}"
output_path = os.path.join(OUTPUT_FOLDER, output_name)

cv2.imwrite(output_path, depth_colormap)

print("Depth 이미지 저장 완료!")
print("저장 위치:", os.path.abspath(output_path))




