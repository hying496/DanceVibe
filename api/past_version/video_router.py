from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
from video_service import VideoService
from pose_service import PoseService
from scoring_service import ScoringService
from image_utils import ImageProcessor

router = APIRouter()
video_service = VideoService()
pose_service = PoseService()
scoring_service = ScoringService()
image_processor = ImageProcessor()


class VideoUploadRequest(BaseModel):
    video_data: str
    video_type: str = "reference"
    extract_audio: bool = True


class FrameProcessRequest(BaseModel):
    image_data: str
    frame_type: str = "webcam"
    timestamp: Optional[float] = None
    detect_pose: bool = True
    calculate_score: bool = True


@router.post("/upload")
async def upload_video(request: VideoUploadRequest):
    """上传视频文件"""
    try:
        result = await video_service.process_video(
            video_data=request.video_data,
            video_type=request.video_type,
            extract_audio=request.extract_audio
        )
        return {
            'success': True,
            'video_id': result['video_id'],
            'duration': result['duration'],
            'frame_count': result['frame_count'],
            'audio_extracted': result.get('audio_extracted', False),
            'message': '视频上传成功'
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"视频上传失败: {str(e)}")


@router.post("/process-frame")
async def process_frame(request: FrameProcessRequest):
    """处理单帧图像"""
    try:
        frame = image_processor.decode_base64_image(request.image_data)

        result = {
            'success': True,
            'timestamp': request.timestamp or time.time(),
            'frame_type': request.frame_type
        }

        if request.detect_pose:
            pose_result = await pose_service.detect_pose(frame)
            result['pose_detection'] = pose_result

            if request.calculate_score and request.frame_type == "webcam":
                score_result = await scoring_service.calculate_detailed_scores(
                    user_landmarks=pose_result.get('landmarks'),
                    timestamp=request.timestamp
                )
                result['scoring'] = score_result

        annotated_frame = image_processor.draw_annotations(
            frame,
            result.get('pose_detection', {}),
            result.get('scoring', {})
        )

        result['processed_image'] = image_processor.encode_image_to_base64(annotated_frame)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"帧处理失败: {str(e)}")


@router.get("/info/{video_id}")
async def get_video_info(video_id: str):
    """获取视频信息"""
    try:
        info = await video_service.get_video_info(video_id)
        return {'success': True, 'video_info': info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取视频信息失败: {str(e)}")


@router.get("/list")
async def list_videos():
    """获取视频列表"""
    try:
        videos = await video_service.list_videos()
        return {'success': True, 'videos': videos}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取视频列表失败: {str(e)}")