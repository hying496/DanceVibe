from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
from scoring_service import ScoringService

router = APIRouter()
scoring_service = ScoringService()


class ScoreCalculationRequest(BaseModel):
    user_landmarks: list
    reference_landmarks: Optional[list] = None
    timestamp: Optional[float] = None
    include_rhythm: bool = True
    include_pose: bool = True
    include_gesture: bool = True


@router.post("/calculate")
async def calculate_score(request: ScoreCalculationRequest):
    """计算相似度分数"""
    try:
        result = await scoring_service.calculate_detailed_scores(
            user_landmarks=request.user_landmarks,
            reference_landmarks=request.reference_landmarks,
            timestamp=request.timestamp,
            include_rhythm=request.include_rhythm,
            include_pose=request.include_pose,
            include_gesture=request.include_gesture
        )

        return {
            'success': True,
            'scores': result,
            'timestamp': request.timestamp or time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"分数计算失败: {str(e)}")


@router.get("/current")
async def get_current_scores():
    """获取当前分数"""
    try:
        scores = await scoring_service.get_current_scores()
        return {
            'success': True,
            'current_scores': scores,
            'average_score': await scoring_service.get_average_score()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取分数失败: {str(e)}")


@router.get("/history")
async def get_score_history(limit: int = 100):
    """获取分数历史"""
    try:
        history = await scoring_service.get_score_history(limit=limit)
        return {
            'success': True,
            'score_history': history,
            'total_entries': len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取历史分数失败: {str(e)}")


@router.get("/average")
async def get_average_score():
    """获取平均分数"""
    try:
        avg_score = await scoring_service.get_average_score()
        return {'success': True, 'average_score': avg_score}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取平均分数失败: {str(e)}")