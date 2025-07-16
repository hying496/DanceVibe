import time
import uuid
from typing import Dict, Any
from game_models import SystemStatus


class VideoService:
    def __init__(self):
        self.status = SystemStatus(service_name="video_service", status="ready")
        self.uploaded_videos = {}

    def is_ready(self) -> bool:
        return self.status.status == "ready"

    def get_status(self) -> SystemStatus:
        return self.status

    async def process_video(self, video_data: str, video_type: str = "reference", extract_audio: bool = True) -> Dict[
        str, Any]:
        video_id = str(uuid.uuid4())
        result = {
            'video_id': video_id,
            'duration': 30.0,
            'frame_count': 900,
            'audio_extracted': extract_audio,
            'video_type': video_type,
            'upload_time': time.time()
        }
        self.uploaded_videos[video_id] = result
        return result

    async def get_video_info(self, video_id: str) -> Dict[str, Any]:
        if video_id in self.uploaded_videos:
            return self.uploaded_videos[video_id]
        raise ValueError(f"视频ID {video_id} 不存在")

    async def list_videos(self) -> Dict[str, Any]:
        return {'videos': list(self.uploaded_videos.values())}

    async def delete_video(self, video_id: str) -> bool:
        if video_id in self.uploaded_videos:
            del self.uploaded_videos[video_id]
            return True
        return False