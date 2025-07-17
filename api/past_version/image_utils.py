import cv2
import base64
import numpy as np
from typing import Dict, Any


class ImageProcessor:
    def decode_base64_image(self, image_data: str) -> np.ndarray:
        try:
            if ',' in image_data:
                img_bytes = base64.b64decode(image_data.split(',')[1])
            else:
                img_bytes = base64.b64decode(image_data)

            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            raise ValueError(f"图像解码失败: {str(e)}")

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        try:
            _, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return 'data:image/jpeg;base64,' + base64.b64encode(jpeg.tobytes()).decode()
        except Exception as e:
            raise ValueError(f"图像编码失败: {str(e)}")

    def draw_annotations(self, frame: np.ndarray, pose_data: Dict[str, Any],
                         score_data: Dict[str, Any] = None) -> np.ndarray:
        annotated_frame = frame.copy()

        # 绘制姿态关键点
        if pose_data.get('persons'):
            for person in pose_data['persons']:
                if 'keypoints' in person:
                    for kp in person['keypoints']:
                        if kp.get('visible', True) and kp.get('confidence', 1.0) > 0.5:
                            cv2.circle(annotated_frame,
                                       (int(kp.get('x', 0)), int(kp.get('y', 0))),
                                       4, (0, 255, 0), -1)

        # 绘制分数信息
        if score_data:
            cv2.putText(annotated_frame, f"Score: {score_data.get('total_score', 0):.1f}%",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 绘制性能信息
        fps = 1000.0 / max(pose_data.get('processing_time_ms', 1), 1)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_frame