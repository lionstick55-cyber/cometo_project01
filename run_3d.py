import cv2
import numpy as np
from src.depth_map import create_depth_map
from src.pointcloud import depth_to_pointcloud

# 이미지 불러오기
image = cv2.imread("apple.jpg")

if image is None:
    raise ValueError("이미지를 불러올 수 없습니다.")

# Depth Map 생성
depth = create_depth_map(image)

# 시각화를 위해 0~255로 변환
depth_visual = (depth * 255).astype(np.uint8)

# 결과 저장
cv2.imwrite("results/depth_result.png", depth_visual)

# 3D 포인트 클라우드 생성
points = depth_to_pointcloud(depth)

print("Point Cloud shape:", points.shape)
print("샘플 좌표 5개:\n", points[:5])
