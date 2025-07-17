import cv2
import numpy as np
import time
from typing import List, Optional, Dict, Any
from collections import deque
from game_models import Person, Keypoint, SystemStatus


class PoseService:
    def __init__(self):
        self.status = SystemStatus(service_name="pose_service", status="ready")
        self.reference_landmarks = None
        self.user_landmarks = None
        self.reference_sequence = deque(maxlen=30)
        self.user_sequence = deque(maxlen=30)
        self.total_detections = 0
        self.total_processing_time = 0.0

    def is_ready(self) -> bool:
        return self.status.status == "ready"

    def get_status(self) -> SystemStatus:
        return self.status

    async def detect_pose(self, frame) -> Dict[str, Any]:
        start_time = time.time()
        persons = []
        h, w = frame.shape[:2]

        # 模拟姿态检测
        if np.random.random() > 0.3:
            person = Person()
            for i in range(33):
                x = w * (0.3 + 0.4 * np.random.random())
                y = h * (0.2 + 0.6 * np.random.random())
                kp = Keypoint(x=x, y=y, z=0, confidence=0.8 + 0.2 * np.random.random())
                person.keypoints.append(kp)
            persons.append(person)

        processing_time = (time.time() - start_time) * 1000
        self.total_detections += 1
        self.total_processing_time += processing_time

        if persons:
            self.user_landmarks = persons[0].keypoints
            self.user_sequence.append(persons[0].keypoints)

        return {
            'success': True,
            'persons': [person.dict() for person in persons],
            'landmarks': self.user_landmarks,
            'processing_time_ms': processing_time
        }

    async def set_reference_pose(self, frame) -> Dict[str, Any]:
        result = await self.detect_pose(frame)
        if result['success'] and result['landmarks']:
            self.reference_landmarks = result['landmarks']
            self.reference_sequence.append(self.reference_landmarks)
            return {'success': True, 'landmarks_count': len(self.reference_landmarks)}
        return {'success': False, 'landmarks_count': 0}

    async def get_reference_landmarks(self) -> Optional[List[Keypoint]]:
        return self.reference_landmarks

    def calculate_pose_similarity(self, landmarks1, landmarks2) -> float:
        if not landmarks1 or not landmarks2 or len(landmarks1) != len(landmarks2):
            return 0.0

        try:
            total_distance = 0.0
            valid_points = 0

            for kp1, kp2 in zip(landmarks1, landmarks2):
                if hasattr(kp1, 'visible') and hasattr(kp2, 'visible'):
                    if kp1.visible and kp2.visible and kp1.confidence > 0.5 and kp2.confidence > 0.5:
                        distance = np.sqrt((kp1.x - kp2.x) ** 2 + (kp1.y - kp2.y) ** 2)
                        total_distance += distance
                        valid_points += 1

            if valid_points == 0:
                return 0.0

            avg_distance = total_distance / valid_points
            similarity = max(0, 1 - avg_distance * 2)
            return similarity
        except:
            return 0.0

    async def get_performance_stats(self) -> Dict[str, Any]:
        return {
            'total_detections': self.total_detections,
            'average_processing_time': self.total_processing_time / max(1, self.total_detections),
            'reference_set': self.reference_landmarks is not None
        }