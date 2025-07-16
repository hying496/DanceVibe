from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from game_service import GameService

router = APIRouter()
game_service = GameService()

class GameStartRequest(BaseModel):
    dance_type: Optional[str] = "freestyle"
    difficulty: Optional[str] = "normal"
    duration: Optional[int] = 180

class GameConfigRequest(BaseModel):
    auto_score: Optional[bool] = True
    show_skeleton: Optional[bool] = True
    feedback_level: Optional[str] = "normal"

@router.post("/start")
async def start_game(request: GameStartRequest):
    """开始游戏"""
    try:
        result = await game_service.start_game(
            dance_type=request.dance_type,
            difficulty=request.difficulty,
            duration=request.duration
        )
        return {
            'success': True,
            'game_id': result['game_id'],
            'session_id': result['session_id'],
            'dance_moves': result['dance_moves'],
            'message': '游戏开始成功!'
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"游戏启动失败: {str(e)}")

@router.post("/pause")
async def pause_game():
    """暂停游戏"""
    try:
        result = await game_service.pause_game()
        return {'success': True, 'message': '游戏已暂停'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"暂停失败: {str(e)}")

@router.post("/resume")
async def resume_game():
    """恢复游戏"""
    try:
        result = await game_service.resume_game()
        return {'success': True, 'message': '游戏已恢复'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"恢复失败: {str(e)}")

@router.post("/stop")
async def stop_game():
    """停止游戏"""
    try:
        result = await game_service.stop_game()
        return {
            'success': True,
            'final_score': result.get('final_score', 0),
            'game_summary': result.get('summary', {}),
            'message': '游戏已结束'
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"停止失败: {str(e)}")

@router.post("/config")
async def update_game_config(config: GameConfigRequest):
    """更新游戏配置"""
    try:
        result = await game_service.update_config(config.dict())
        return {'success': True, 'config': result, 'message': '配置更新成功'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"配置更新失败: {str(e)}")

@router.get("/moves")
async def get_dance_moves():
    """获取舞蹈动作列表"""
    try:
        moves = await game_service.get_dance_moves()
        return {'success': True, 'moves': moves}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取动作失败: {str(e)}")

@router.get("/current")
async def get_current_game():
    """获取当前游戏状态"""
    try:
        game_state = await game_service.get_current_game()
        return {'success': True, 'game_state': game_state}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取游戏状态失败: {str(e)}")