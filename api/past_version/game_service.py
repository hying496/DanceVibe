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