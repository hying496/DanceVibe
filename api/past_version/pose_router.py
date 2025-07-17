from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
from pose_service import PoseService
from image_utils import ImageProcessor

router = APIRouter()
pose_service = PoseService()
image_processor = ImageProcessor()


class PoseDetectionRequest(BaseModel):
    image_data: str
    timestamp: Optional[float] = None
    save_as_reference: bool = False


@router.post("/detect")
async def detect_pose(request: PoseDetectionRequest):
    """姿态检测接口"""
    try:
        frame = image_processor.decode_base64_image(request.image_data)
        result = await pose_service.detect_pose(frame)

        if request.save_as_reference and result.get('success'):
            await pose_service.set_reference_pose(frame)

        return {
            'success': True,
            'pose_data': result,
            'timestamp': request.timestamp or time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"姿态检测失败: {str(e)}")


@router.get("/landmarks/reference")
async def get_reference_landmarks():
    """获取参考姿态关键点"""
    try:
        landmarks = await pose_service.get_reference_landmarks()
        return {
            'success': True,
            'landmarks': landmarks,
            'count': len(landmarks) if landmarks else 0
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取参考数据失败: {str(e)}")


@router.post("/landmarks/reference")
async def set_reference_landmarks(request: PoseDetectionRequest):
    """设置参考姿态关键点"""
    try:
        frame = image_processor.decode_base64_image(request.image_data)
        result = await pose_service.set_reference_pose(frame)
        return {
            'success': True,
            'landmarks_saved': result['landmarks_count'],
            'message': '参考姿态已保存'
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"保存参考姿态失败: {str(e)}")


@router.get("/stats")
async def get_pose_stats():
    """获取姿态检测统计信息"""
    try:
        stats = await pose_service.get_performance_stats()
        return {'success': True, 'stats': stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取统计信息失败: {str(e)}")