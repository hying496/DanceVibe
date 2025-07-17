from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import json
import time
import base64
import cv2
import numpy as np

from pose_service import PoseService
from audio_service import AudioService
from scoring_service import ScoringService
from game_service import GameService
from image_utils import ImageProcessor
from video_utils import mp4_2_mp3, get_beats

websocket_router = APIRouter()

# 初始化服务
pose_service = PoseService()
audio_service = AudioService()
scoring_service = ScoringService()
game_service = GameService()
image_processor = ImageProcessor()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ 客户端已连接，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ 客户端已断开，当前连接数: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"发送消息失败: {e}")


manager = ConnectionManager()


@websocket_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(data, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket错误: {e}")
        manager.disconnect(websocket)


async def handle_websocket_message(data: dict, websocket: WebSocket):
    event_type = data.get('event', data.get('type'))

    try:
        if event_type == 'start_game':
            await handle_start_game(data, websocket)
        elif event_type == 'pause_game':
            await handle_pause_game(data, websocket)
        elif event_type == 'resume_game':
            await handle_resume_game(data, websocket)
        elif event_type == 'stop_game':
            await handle_stop_game(data, websocket)
        elif event_type == 'upload_reference_video':
            await handle_upload_reference_video(data, websocket)
        elif event_type == 'frame':
            await handle_frame_processing(data, websocket)
        elif event_type == 'get_status':
            await handle_get_status(data, websocket)
        else:
            await manager.send_personal_message({
                'event': 'error',
                'message': f'未知的事件类型: {event_type}'
            }, websocket)

    except Exception as e:
        await manager.send_personal_message({
            'event': 'error',
            'message': f'处理消息失败: {str(e)}'
        }, websocket)


async def handle_start_game(data: dict, websocket: WebSocket):
    try:
        print("🎮 开始游戏")
        result = await game_service.start_game(
            dance_type=data.get('dance', {}).get('name', 'freestyle'),
            difficulty=data.get('difficulty', 'normal'),
            duration=data.get('duration', 180)
        )

        await manager.send_personal_message({
            'event': 'game_started',
            'success': True,
            'game_id': result['game_id'],
            'session_id': result['session_id'],
            'dance': data.get('dance', {}),
            'first_move': result['dance_moves'][0] if result['dance_moves'] else {},
            'message': '游戏开始！'
        }, websocket)

    except Exception as e:
        await manager.send_personal_message({
            'event': 'error',
            'message': f'游戏启动失败: {str(e)}'
        }, websocket)


async def handle_pause_game(data: dict, websocket: WebSocket):
    try:
        await game_service.pause_game()
        await manager.send_personal_message({'event': 'game_paused', 'success': True}, websocket)
    except Exception as e:
        await manager.send_personal_message({'event': 'error', 'message': f'暂停失败: {str(e)}'}, websocket)


async def handle_resume_game(data: dict, websocket: WebSocket):
    try:
        await game_service.resume_game()
        await manager.send_personal_message({'event': 'game_resumed', 'success': True}, websocket)
    except Exception as e:
        await manager.send_personal_message({'event': 'error', 'message': f'恢复失败: {str(e)}'}, websocket)


async def handle_stop_game(data: dict, websocket: WebSocket):
    try:
        result = await game_service.stop_game()
        await manager.send_personal_message({
            'event': 'game_stopped',
            'success': True,
            'final_score': result.get('final_score', 0),
            'summary': result.get('summary', {})
        }, websocket)
    except Exception as e:
        await manager.send_personal_message({'event': 'error', 'message': f'停止失败: {str(e)}'}, websocket)


async def handle_upload_reference_video(data: dict, websocket: WebSocket):
    try:
        print("📹 处理参考视频上传")
        video_data = data.get('video')

        if not video_data:
            await manager.send_personal_message({'event': 'error', 'message': '视频数据为空'}, websocket)
            return

        # 解码和保存视频
        if ',' in video_data:
            video_bytes = base64.b64decode(video_data.split(',')[1])
        else:
            video_bytes = base64.b64decode(video_data)

        reference_video_path = 'tmp_reference.mp4'
        with open(reference_video_path, 'wb') as f:
            f.write(video_bytes)

        # 提取音频和节拍
        try:
            audio_path = mp4_2_mp3(reference_video_path)
            tempo, beat_frames, beat_times = get_beats(audio_path)

            # 更新音频服务的节拍数据
            audio_service.beat_times = beat_times
            audio_service.tempo = tempo

            print(f"✅ 节拍提取完成，共 {len(beat_times)} 个节拍点")

            await manager.send_personal_message({
                'event': 'reference_ready',
                'success': True,
                'beat_count': len(beat_times),
                'tempo': tempo,
                'video_id': reference_video_path
            }, websocket)

        except Exception as e:
            print(f"⚠️ 音频处理失败: {e}")
            await manager.send_personal_message({
                'event': 'reference_ready',
                'success': True,
                'beat_count': 0,
                'video_id': reference_video_path,
                'warning': '音频处理失败，仅支持姿态检测'
            }, websocket)

    except Exception as e:
        print(f"❌ 视频处理失败: {e}")
        await manager.send_personal_message({
            'event': 'error',
            'message': f'视频上传失败: {str(e)}'
        }, websocket)


async def handle_frame_processing(data: dict, websocket: WebSocket):
    try:
        frame_type = data.get('frame_type', data.get('type', 'webcam'))
        image_data = data.get('image', '')
        current_time = data.get('current_time', None)

        if not image_data:
            return

        # 解码图像
        frame = image_processor.decode_base64_image(image_data)
        if frame is None:
            return

        # 姿态检测
        pose_result = await pose_service.detect_pose(frame)

        similarity_data = None
        if pose_result.get('success') and pose_result.get('landmarks'):

            if frame_type == 'reference':
                # 保存参考姿态
                await pose_service.set_reference_pose(frame)

            elif frame_type == 'webcam':
                # 计算相似度分数
                reference_landmarks = await pose_service.get_reference_landmarks()
                if reference_landmarks:
                    similarity_data = await scoring_service.calculate_detailed_scores(
                        user_landmarks=pose_result['landmarks'],
                        reference_landmarks=reference_landmarks,
                        timestamp=current_time
                    )

        # 绘制注释
        annotated_frame = image_processor.draw_annotations(
            frame,
            pose_result,
            similarity_data
        )

        # 编码返回图像
        b64img = image_processor.encode_image_to_base64(annotated_frame)

        # 发送帧结果
        result_data = {
            'event': 'frame_result',
            'type': frame_type,
            'image': b64img,
            'persons_detected': len(pose_result.get('persons', [])),
            'processing_time_ms': pose_result.get('processing_time_ms', 0),
            'hands_detected': 0,
            'gestures_recognized': 0
        }

        if similarity_data:
            result_data.update({
                'similarity': similarity_data,
                'average_score': await scoring_service.get_average_score()
            })

        await manager.send_personal_message(result_data, websocket)

        # 发送实时分数更新
        if frame_type == 'webcam' and similarity_data:
            await manager.send_personal_message({
                'event': 'score_update',
                'current_scores': similarity_data,
                'average_score': await scoring_service.get_average_score(),
                'game_active': await game_service.is_game_active()
            }, websocket)

    except Exception as e:
        print(f"帧处理失败: {e}")
        await manager.send_personal_message({
            'event': 'error',
            'message': f'帧处理失败: {str(e)}'
        }, websocket)


async def handle_get_status(data: dict, websocket: WebSocket):
    try:
        game_status = await game_service.get_game_status()
        current_scores = await scoring_service.get_current_scores()

        status = {
            'game_active': game_status.get('active', False),
            'game_paused': game_status.get('paused', False),
            'current_move_index': game_status.get('current_move_index', 0),
            'total_moves': game_status.get('total_moves', 0),
            'current_scores': current_scores,
            'has_reference': await pose_service.get_reference_landmarks() is not None,
            'has_user_data': pose_service.user_landmarks is not None,
            'beat_times_count': len(await audio_service.get_beat_times())
        }

        if game_status.get('active') and game_status.get('current_move'):
            status.update({
                'current_move': game_status['current_move'],
                'remaining_time': game_status.get('remaining_time', 0)
            })

        await manager.send_personal_message({
            'event': 'status_update',
            'status': status
        }, websocket)

    except Exception as e:
        await manager.send_personal_message({
            'event': 'error',
            'message': f'状态查询失败: {str(e)}'
        }, websocket)