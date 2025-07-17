import time
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
from game_models import ScoreData, SystemStatus


class ScoringService:
    def __init__(self):
        self.status = SystemStatus(service_name="scoring_service", status="ready")
        self.current_scores = ScoreData()
        self.score_history = deque(maxlen=100)
        self.cumulative_score = 0.0

    def is_ready(self) -> bool:
        return self.status.status == "ready"

    def get_status(self) -> SystemStatus:
        return self.status

    async def calculate_detailed_scores(self, user_landmarks, reference_landmarks=None, timestamp=None, **kwargs) -> \
    Dict[str, Any]:
        # 模拟详细评分计算
        pose_score = np.random.uniform(60, 95)
        rhythm_score = np.random.uniform(50, 90)
        hand_score = np.random.uniform(40, 90)
        total_score = pose_score * 0.5 + rhythm_score * 0.3 + hand_score * 0.2

        scores = ScoreData(
            pose_score=pose_score,
            rhythm_score=rhythm_score,
            hand_score=hand_score,
            total_score=total_score,
            timestamp=timestamp or time.time()
        )

        self.current_scores = scores
        self.score_history.append(scores)

        # 更新累积分数
        if self.score_history:
            total = sum(s.total_score for s in self.score_history)
            self.cumulative_score = total / len(self.score_history)

        return scores.to_dict()

    async def get_current_scores(self) -> Dict[str, Any]:
        return self.current_scores.to_dict()

    async def get_average_score(self) -> float:
        return self.cumulative_score

    async def get_score_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [score.to_dict() for score in list(self.score_history)[-limit:]]

    async def get_performance_stats(self) -> Dict[str, Any]:
        return {
            'total_scores': len(self.score_history),
            'average_score': self.cumulative_score,
            'current_score': self.current_scores.total_score
        }