from fastapi import APIRouter, HTTPException
import time
import psutil
from game_service import GameService
from pose_service import PoseService
from audio_service import AudioService
from scoring_service import ScoringService
from video_service import VideoService

router = APIRouter()

# 初始化所有服务
game_service = GameService()
pose_service = PoseService()
audio_service = AudioService()
scoring_service = ScoringService()
video_service = VideoService()


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        'status': 'healthy',
        'version': '3.0.0',
        'services': {
            'pose_detection': pose_service.is_ready(),
            'audio_analysis': audio_service.is_ready(),
            'scoring_system': scoring_service.is_ready(),
            'game_engine': game_service.is_ready(),
            'video_processing': video_service.is_ready()
        },
        'timestamp': time.time()
    }


@router.get("/game")
async def get_game_status():
    """获取游戏状态"""
    try:
        status = await game_service.get_game_status()
        return {
            'success': True,
            'game_status': status,
            'timestamp': time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取游戏状态失败: {str(e)}")


@router.get("/performance")
async def get_performance_stats():
    """获取性能统计"""
    try:
        stats = {
            'pose_detection': await pose_service.get_performance_stats(),
            'scoring_system': await scoring_service.get_performance_stats(),
            'game_engine': await game_service.get_performance_stats(),
            'audio_analysis': await audio_service.get_performance_stats()
        }
        return {
            'success': True,
            'performance_stats': stats,
            'timestamp': time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取性能统计失败: {str(e)}")


@router.get("/system")
async def get_system_status():
    """获取系统状态"""
    try:
        # 获取系统信息
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        status = {
            'services': {
                'pose_service': pose_service.get_status().dict(),
                'audio_service': audio_service.get_status().dict(),
                'scoring_service': scoring_service.get_status().dict(),
                'game_service': game_service.get_status().dict(),
                'video_service': video_service.get_status().dict()
            },
            'system_info': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_total': memory.total,
                'memory_available': memory.available
            }
        }

        return {
            'success': True,
            'system_status': status,
            'timestamp': time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取系统状态失败: {str(e)}")