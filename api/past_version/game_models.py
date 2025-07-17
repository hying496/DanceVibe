from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum
import time


class GameStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


class DetectorType(str, Enum):
    MEDIAPIPE = "mediapipe"
    YOLOV8 = "yolov8"
    HYBRID = "hybrid"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"
    EXPERT = "expert"


class Keypoint(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    confidence: float = 1.0
    visible: bool = True


class Person(BaseModel):
    keypoints: List[Keypoint] = []
    bbox: List[float] = [0, 0, 100, 100]
    id: int = 0
    hands: List[Dict] = []
    confidence: float = 1.0


class ScoreData(BaseModel):
    pose_score: float = 0.0
    rhythm_score: float = 0.0
    hand_score: float = 0.0
    gesture_score: float = 0.0
    total_score: float = 0.0
    timestamp: float = 0.0

    def to_dict(self):
        return {
            'pose_score': round(self.pose_score, 1),
            'rhythm_score': round(self.rhythm_score, 1),
            'hand_score': round(self.hand_score, 1),
            'gesture_score': round(self.gesture_score, 1),
            'total_score': round(self.total_score, 1),
            'timestamp': self.timestamp
        }


class MoveData(BaseModel):
    name: str
    emoji: str = "🤸‍♂️"
    duration: float = 5.0
    description: str = ""
    difficulty: DifficultyLevel = DifficultyLevel.NORMAL
    required_poses: List[str] = []
    target_score: float = 80.0


class SystemStatus(BaseModel):
    service_name: str
    status: str = "unknown"
    last_updated: float = 0.0
    performance_stats: Dict[str, Any] = {}
    error_message: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self.last_updated = time.time()

    def update_status(self, status: str, error_message: str = ""):
        self.status = status
        self.error_message = error_message
        self.last_updated = time.time()


# 默认舞蹈动作
DEFAULT_DANCE_MOVES = [
    MoveData(name="准备姿势", emoji="🤸‍♂️", duration=3.0),
    MoveData(name="举起双手", emoji="🙋‍♂️", duration=4.0),
    MoveData(name="左右摆动", emoji="🤸‍♂️", duration=5.0),
    MoveData(name="转身动作", emoji="💫", duration=4.0),
    MoveData(name="跳跃节拍", emoji="🦘", duration=5.0),
    MoveData(name="结束姿势", emoji="🙌", duration=3.0)
]

# ===== api/game_service.py =====
import time
import uuid
from typing import Dict, List, Optional, Any
from collections import deque
from game_models import GameStatus, MoveData, SystemStatus, DEFAULT_DANCE_MOVES, ScoreData


class GameService:
    def __init__(self):
        self.status = SystemStatus(service_name="game_service", status="ready")
        self.game_active = False
        self.game_paused = False
        self.current_move_index = 0
        self.move_start_time = None
        self.sessions_history = deque(maxlen=100)
        self.current_scores = ScoreData()
        self.cumulative_score = 0.0
        self.score_history = deque(maxlen=100)

    def is_ready(self) -> bool:
        return self.status.status == "ready"

    def get_status(self) -> SystemStatus:
        return self.status

    async def start_game(self, dance_type: str = "freestyle", difficulty: str = "normal", duration: int = 180) -> Dict[
        str, Any]:
        try:
            self.game_active = True
            self.game_paused = False
            self.current_move_index = 0
            self.move_start_time = time.time()
            self.score_history.clear()
            self.cumulative_score = 0.0

            session_id = str(uuid.uuid4())
            return {
                'game_id': f"game_{int(time.time())}",
                'session_id': session_id,
                'dance_moves': [move.dict() for move in DEFAULT_DANCE_MOVES],
                'difficulty': difficulty,
                'duration': duration,
                'start_time': time.time()
            }
        except Exception as e:
            raise e

    async def pause_game(self) -> Dict[str, Any]:
        if not self.game_active:
            raise ValueError("游戏未在运行状态")
        self.game_paused = True
        return {'paused_at': time.time()}

    async def resume_game(self) -> Dict[str, Any]:
        if not self.game_paused:
            raise ValueError("游戏未在暂停状态")
        self.game_paused = False
        self.move_start_time = time.time()
        return {'resumed_at': time.time()}

    async def stop_game(self) -> Dict[str, Any]:
        final_score = self.cumulative_score
        self.game_active = False
        self.game_paused = False
        self.current_move_index = 0
        self.move_start_time = None

        return {
            'final_score': final_score,
            'summary': {
                'total_moves': len(DEFAULT_DANCE_MOVES),
                'completed_moves': self.current_move_index,
                'average_score': final_score,
                'score_history_count': len(self.score_history)
            }
        }

    async def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return config

    async def get_dance_moves(self) -> List[Dict[str, Any]]:
        return [move.dict() for move in DEFAULT_DANCE_MOVES]

    async def get_current_game(self) -> Dict[str, Any]:
        if not self.game_active:
            return {'active': False, 'session': None, 'current_move': None, 'remaining_time': 0}

        current_move = None
        remaining_time = 0
        if self.current_move_index < len(DEFAULT_DANCE_MOVES):
            current_move = DEFAULT_DANCE_MOVES[self.current_move_index]
            if self.move_start_time:
                elapsed = time.time() - self.move_start_time
                remaining_time = max(0, current_move.duration - elapsed)

        return {
            'active': self.game_active,
            'paused': self.game_paused,
            'current_move': current_move.dict() if current_move else None,
            'current_move_index': self.current_move_index,
            'remaining_time': remaining_time,
            'total_moves': len(DEFAULT_DANCE_MOVES)
        }

    async def get_game_status(self) -> Dict[str, Any]:
        return await self.get_current_game()

    async def is_game_active(self) -> bool:
        return self.game_active

    async def get_performance_stats(self) -> Dict[str, Any]:
        return {
            'total_games': len(self.sessions_history),
            'current_game_active': self.game_active,
            'best_score': max((s.total_score for s in self.score_history), default=0),
            'average_score': self.cumulative_score
        }


# ===== api/pose_service.py =====
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