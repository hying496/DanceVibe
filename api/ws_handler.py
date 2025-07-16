from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import base64
import numpy as np
import time
import cv2
import tempfile
import os
from typing import Dict, List, Optional
import sys

# 添加项目根路径到sys.path
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .model import Keypoint, Landmarks
from .utils import decode_base64_image, encode_image_to_base64, pad_landmarks, draw_landmarks

# 修复导入路径 - 直接从根目录导入
try:
    from score.similarity import calculate_pose_similarity
    from score.score_pose import score_pose
    from score.music_beat import mp4_2_mp3, get_beats
    from score.motion_match import match_motion_to_beats
    from score.average_similarity import CumulativeScore
except ImportError as e:
    print(f"⚠️ Score模块导入失败: {e}")


    # 提供备用函数
    def calculate_pose_similarity(lm1, lm2):
        return 0.8  # 默认相似度


    def score_pose(pose_score, delta_t):
        return pose_score * 0.9


    def mp4_2_mp3(video_path):
        return ""


    def get_beats(audio_path):
        return 120, [], []


    def match_motion_to_beats(motion, beats):
        return []


    class CumulativeScore:
        def __init__(self):
            self.scores = []
            self.average = 0.0

        def update(self, score):
            self.scores.append(score)
            self.average = sum(self.scores) / len(self.scores)

        def reset(self):
            self.scores = []
            self.average = 0.0

# 导入姿态检测管理器
try:
    from detector.pose_detector import DetectorType, PoseDetectionManager
except ImportError as e:
    print(f"⚠️ Detector模块导入失败: {e}")


    # 提供备用类
    class DetectorType:
        MEDIAPIPE = "mediapipe"


    class PoseDetectionManager:
        def __init__(self, detector_type):
            self.detector_type = detector_type
            print(f"✅ 使用备用检测器: {detector_type}")

        def detect_poses(self, frame):
            # 返回空的检测结果
            return [], {"processing_time_ms": 10}

ws_router = APIRouter()


# 全局存储
class GameSession:
    def __init__(self):
        self.reference_landmarks: Optional[List[Keypoint]] = None
        self.reference_video_path: Optional[str] = None
        self.beat_times: List[float] = []
        self.game_started: bool = False
        self.game_paused: bool = False
        self.selected_dance: Dict = {'id': 1, 'name': 'Easy'}
        self.level: str = 'Easy'  # 新增难度字段
        self.cumulative_score = CumulativeScore()
        self.start_time: Optional[float] = None
        self.pose_manager = PoseDetectionManager(DetectorType.MEDIAPIPE)
        self.frame_count: int = 0
        self.score_history: List[float] = []


# 连接管理
active_sessions: Dict[str, GameSession] = {}


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = id(websocket)
    active_sessions[session_id] = GameSession()
    session = active_sessions[session_id]

    print(f"✅ WebSocket连接成功 (Session: {session_id})")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                await handle_message(msg, websocket, session)
            except Exception as e:
                print(f"消息处理失败: {e}")
                await websocket.send_json({
                    'event': 'error',
                    'message': f'消息处理失败: {str(e)}'
                })

    except WebSocketDisconnect:
        print(f"❌ WebSocket断开 (Session: {session_id})")
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        print(f"WebSocket错误: {e}")
        if session_id in active_sessions:
            del active_sessions[session_id]


async def handle_message(msg: Dict, websocket: WebSocket, session: GameSession):
    """处理WebSocket消息"""
    event = msg.get('event')
    print(f"📨 收到WebSocket消息: {event}")

    if event == 'frame':
        await handle_frame(msg, websocket, session)
    elif event == 'upload_reference_video':
        await handle_upload_reference_video(msg, websocket, session)
    elif event == 'start_game':
        await handle_start_game(msg, websocket, session)
    elif event == 'pause_game':
        await handle_pause_game(websocket, session)
    elif event == 'resume_game':
        await handle_resume_game(websocket, session)
    elif event == 'stop_game':
        await handle_stop_game(websocket, session)
    else:
        print(f"❓ 未知事件: {event}")


async def handle_frame(msg: Dict, websocket: WebSocket, session: GameSession):
    """处理视频帧"""
    frame_type = msg.get('frame_type', 'webcam')
    image_data = msg.get('image', '')
    current_time = msg.get('current_time', 0.0)

    print(f"🎬 处理{frame_type}帧，图片数据长度: {len(image_data)}")

    if not image_data:
        print("❌ 图片数据为空")
        return

    try:
        # 解码图片
        frame = decode_base64_image(image_data)
        if frame is None:
            print("❌ 图片解码失败")
            return

        print(f"✅ 图片解码成功，尺寸: {frame.shape}")

        # 姿态检测
        start_time = time.time()
        persons, det_info = session.pose_manager.detect_poses(frame)
        processing_time = (time.time() - start_time) * 1000

        print(f"🔍 姿态检测完成，检测到 {len(persons) if persons else 0} 人，耗时: {processing_time:.2f}ms")

        # 提取关键点
        landmarks = None
        if persons and len(persons) > 0:
            # 取第一个人的关键点
            person = persons[0]
            kps = []
            for kp in person.keypoints:
                kps.append(Keypoint(
                    x=float(kp.x),
                    y=float(kp.y),
                    z=getattr(kp, 'z', 0.0),
                    confidence=getattr(kp, 'confidence', 1.0),
                    visible=getattr(kp, 'visible', True)
                ))
            landmarks = pad_landmarks(kps, 33)
            print(f"✅ 提取关键点完成，共{len(landmarks)}个点")

        # 绘制姿态 - 对于参考视频使用红色，用户视频使用绿色
        if landmarks:
            color = (0, 0, 255) if frame_type == 'reference' else (0, 255, 0)  # 红色/绿色
            vis_frame = draw_landmarks(frame.copy(), landmarks, color=color)
            print(f"✅ 绘制姿态完成，颜色: {'红色' if frame_type == 'reference' else '绿色'}")
        else:
            vis_frame = frame.copy()
            print("⚠️ 无关键点，使用原图")

        # 编码图片
        vis_img_b64 = encode_image_to_base64(vis_frame)

        # 发送帧结果
        await websocket.send_json({
            'event': 'frame_result',
            'type': frame_type,
            'image': vis_img_b64,
            'persons_detected': len(persons) if persons else 0,
            'processing_time_ms': processing_time,
        })

        print(f"📤 发送{frame_type}帧结果完成")

        # 处理参考帧
        if frame_type == 'reference':
            session.reference_landmarks = landmarks
            print(f"📹 参考视频关键点已保存")

        # 处理用户帧并计算分数
        elif frame_type == 'webcam' and session.reference_landmarks and landmarks:
            print(f"🎯 开始计算用户帧分数")
            await calculate_and_send_score(landmarks, current_time, websocket, session)

    except Exception as e:
        print(f"❌ 帧处理错误: {e}")
        await websocket.send_json({
            'event': 'error',
            'message': f'帧处理失败: {str(e)}'
        })


async def handle_upload_reference_video(msg: Dict, websocket: WebSocket, session: GameSession):
    """处理参考视频上传"""
    video_data = msg.get('video', '')
    if not video_data:
        return

    try:
        # 解码视频数据
        if ',' in video_data:
            video_bytes = base64.b64decode(video_data.split(',')[1])
        else:
            video_bytes = base64.b64decode(video_data)

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            session.reference_video_path = tmp_file.name

        # 提取音频和节拍
        try:
            audio_path = mp4_2_mp3(session.reference_video_path)
            tempo, beats, beat_times = get_beats(audio_path)
            session.beat_times = beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times)

            # 清理临时音频文件
            if os.path.exists(audio_path):
                os.remove(audio_path)

            print(f"✅ 节拍提取成功，共{len(session.beat_times)}个节拍点")

            await websocket.send_json({
                'event': 'reference_ready',
                'beat_count': len(session.beat_times),
                'tempo': tempo
            })

        except Exception as e:
            print(f"节拍提取失败: {e}")
            session.beat_times = []
            await websocket.send_json({
                'event': 'error',
                'message': f'节拍提取失败: {str(e)}'
            })

    except Exception as e:
        print(f"视频上传处理失败: {e}")
        await websocket.send_json({
            'event': 'error',
            'message': f'视频上传失败: {str(e)}'
        })


async def handle_start_game(msg: Dict, websocket: WebSocket, session: GameSession):
    """开始游戏"""
    session.selected_dance = msg.get('dance', session.selected_dance)
    session.level = msg.get('level', session.selected_dance.get('name', 'Easy'))  # 解析level
    session.game_started = True
    session.game_paused = False
    session.start_time = time.time()
    session.frame_count = 0
    session.score_history = []
    session.cumulative_score.reset()
    await websocket.send_json({
        'event': 'game_started',
        'dance': session.selected_dance,
        'level': session.level
    })
    print(f"🎮 游戏开始: {session.selected_dance['name']} 难度: {session.level}")


async def handle_pause_game(websocket: WebSocket, session: GameSession):
    """暂停游戏"""
    session.game_paused = True
    await websocket.send_json({'event': 'game_paused'})
    print("⏸️ 游戏暂停")


async def handle_resume_game(websocket: WebSocket, session: GameSession):
    """恢复游戏"""
    session.game_paused = False
    await websocket.send_json({'event': 'game_resumed'})
    print("▶️ 游戏恢复")


async def handle_stop_game(websocket: WebSocket, session: GameSession):
    """停止游戏"""
    final_score = session.cumulative_score.average * 100

    session.game_started = False
    session.game_paused = False

    await websocket.send_json({
        'event': 'game_stopped',
        'final_score': round(final_score, 2)
    })

    print(f"🛑 游戏结束，最终得分: {final_score:.2f}")


async def calculate_and_send_score(user_landmarks: List[Keypoint], current_time: float,
                                   websocket: WebSocket, session: GameSession):
    """计算并发送分数"""
    if not session.game_started or session.game_paused:
        return
    try:
        pose_score = calculate_pose_similarity(user_landmarks, session.reference_landmarks) or 0.0
        rhythm_score = 0.0
        delta_t = 1.0
        if session.beat_times and session.start_time:
            relative_time = current_time - (time.time() - session.start_time)
            if session.beat_times:
                delta_t = min([abs(relative_time - bt) for bt in session.beat_times])
                rhythm_score = max(0, 1 - delta_t / 0.4)
        hand_score = pose_score * 0.8
        # 难度权重表
        LEVEL_WEIGHTS = {
            'Easy':    (0.8, 0.15, 0.05),
            'Medium':  (0.6, 0.3, 0.1),
            'Hard':    (0.5, 0.4, 0.1),
            'Expert':  (0.4, 0.5, 0.1)
        }
        w_pose, w_rhythm, w_hand = LEVEL_WEIGHTS.get(session.level, (0.8, 0.15, 0.05))
        total_score = w_pose * pose_score + w_rhythm * rhythm_score + w_hand * hand_score
        session.cumulative_score.update(total_score)
        session.score_history.append(total_score)
        session.frame_count += 1
        await websocket.send_json({
            'event': 'score_update',
            'current_scores': {
                'pose_score': round(pose_score * 100, 2),
                'rhythm_score': round(rhythm_score * 100, 2),
                'hand_score': round(hand_score * 100, 2),
                'total_score': round(total_score * 100, 2)
            },
            'average_score': round(session.cumulative_score.average * 100, 2),
            'frame_count': session.frame_count
        })
        if session.start_time and (time.time() - session.start_time) > 60:
            await handle_stop_game(websocket, session)
    except Exception as e:
        print(f"分数计算错误: {e}")
        await websocket.send_json({
            'event': 'error',
            'message': f'分数计算失败: {str(e)}'
        })