import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, DPTForDepthEstimation

def main():

    base_dir = os.path.dirname(os.path.dirname(__file__))
    images_dir = os.path.join(base_dir, "images")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------
    # 1️⃣ 파일 이름 직접 입력
    # ----------------------------
    filename = input("변환할 이미지 파일 이름을 입력하세요 (예: cat.jpg): ")

    image_path = os.path.join(images_dir, filename)

    if not os.path.exists(image_path):
        print("❌ 이미지 파일이 존재하지 않습니다.")
        print("찾는 경로:", image_path)
        return

    print("선택된 이미지:", image_path)

    image = Image.open(image_path).convert("RGB")

    # ----------------------------
    # 2️⃣ 모델 로드
    # ----------------------------
    processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # ----------------------------
    # 3️⃣ Depth 변환
    # ----------------------------
    depth = predicted_depth.squeeze().cpu().numpy()

    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)

    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    # ----------------------------
    # 4️⃣ 저장
    # ----------------------------
    name_only = os.path.splitext(filename)[0]
    save_path = os.path.join(results_dir, f"{name_only}_depth.png")

    cv2.imwrite(save_path, depth_color)

    print("✅ Depth 생성 완료")
    print("저장 위치:", save_path)


if __name__ == "__main__":
    main()



