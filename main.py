import cv2
import numpy as np
import sys
import os

from src.depth_map import create_depth_map
from src.pointcloud import depth_to_pointcloud


def main():

    # -------------------------
    # 1️⃣ 이미지 경로 입력 받기
    # -------------------------
    if len(sys.argv) < 2:
        print("사용법: python main.py 이미지파일명")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("이미지 파일이 존재하지 않습니다.")
        return

    # -------------------------
    # 2️⃣ 이미지 로드
    # -------------------------
    image = cv2.imread(image_path)

    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    print("이미지 로드 완료")

    # -------------------------
    # 3️⃣ Depth Map 생성
    # -------------------------
    depth_map = create_depth_map(image)

    # -------------------------
    # 4️⃣ Depth Normalize (0~255)
    # -------------------------
    depth_normalized = cv2.normalize(
        depth_map, None, 0, 255, cv2.NORM_MINMAX
    )

    depth_uint8 = depth_normalized.astype(np.uint8)

    # -------------------------
    # 5️⃣ 컬러맵 적용
    # -------------------------
    depth_color = cv2.applyColorMap(
        depth_uint8, cv2.COLORMAP_JET
    )

    # -------------------------
    # 6️⃣ 결과 폴더 생성
    # -------------------------
    os.makedirs("results", exist_ok=True)

    # -------------------------
    # 7️⃣ 이미지 저장
    # -------------------------
    cv2.imwrite("results/depth_output.png", depth_uint8)
    cv2.imwrite("results/depth_color.png", depth_color)

    print("Depth 이미지 저장 완료")

    # -------------------------
    # 8️⃣ 3D Point Cloud 생성
    # -------------------------
    pointcloud = depth_to_pointcloud(depth_map)

    print("Point Cloud shape:", pointcloud.shape)
    print("샘플 좌표 5개:\n", pointcloud[:5])

    # -------------------------
    # 9️⃣ 화면에 표시
    # -------------------------
    cv2.imshow("Original", image)
    cv2.imshow("Depth Color", depth_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


