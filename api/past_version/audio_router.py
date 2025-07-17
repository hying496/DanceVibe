from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from audio_service import AudioService

router = APIRouter()
audio_service = AudioService()


class AudioAnalysisRequest(BaseModel):
    audio_data: Optional[str] = None
    video_id: Optional[str] = None
    extract_beats: bool = True
    extract_tempo: bool = True


@router.post("/analyze")
async def analyze_audio(request: AudioAnalysisRequest):
    """分析音频文件"""
    try:
        if request.audio_data:
            result = await audio_service.analyze_audio_data(request.audio_data)
        elif request.video_id:
            result = await audio_service.analyze_video_audio(request.video_id)
        else:
            raise HTTPException(status_code=400, detail="需要提供音频数据或视频ID")

        return {
            'success': True,
            'analysis': result,
            'tempo': result.get('tempo'),
            'beat_times': result.get('beat_times', []),
            'beat_count': len(result.get('beat_times', [])),
            'message': '音频分析完成'
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音频分析失败: {str(e)}")


@router.get("/beats")
async def get_beat_times():
    """获取当前音乐的节拍点"""
    try:
        beat_times = await audio_service.get_beat_times()
        return {
            'success': True,
            'beat_times': beat_times,
            'beat_count': len(beat_times),
            'next_beat': audio_service.get_next_beat_time()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取节拍失败: {str(e)}")


@router.post("/extract-from-video")
async def extract_audio_from_video(video_path: str):
    """从视频中提取音频"""
    try:
        result = await audio_service.extract_from_video(video_path)
        return {
            'success': True,
            'audio_id': result['audio_id'],
            'audio_path': result['audio_path'],
            'duration': result['duration'],
            'message': '音频提取成功'
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音频提取失败: {str(e)}")