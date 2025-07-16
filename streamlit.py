import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import threading
from queue import Queue
import tempfile
import os

# 页面配置
st.set_page_config(
    page_title="🎭 Dance Master Challenge",
    page_icon="🕺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
    }

    .stApp {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }

    .score-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #00ff41;
        text-shadow: 0 0 20px #00ff41;
    }

    .feedback-text {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        color: #ff6b35;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .stButton > button {
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# 初始化MediaPipe
@st.cache_resource
def initialize_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    return mp_pose, pose, mp_drawing


# 姿态检测函数
def detect_pose(frame, pose, mp_pose, mp_drawing):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    output_frame = frame.copy()
    keypoints = None

    if results.pose_landmarks:
        # 绘制姿态
        mp_drawing.draw_landmarks(
            output_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # 提取关键点
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        keypoints = np.array(keypoints)

        # 添加检测状态
        cv2.putText(output_frame, "POSE DETECTED! 🎯", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(output_frame, "No pose detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return output_frame, keypoints


# 计算姿态相似度
def calculate_similarity(pose1, pose2):
    if pose1 is None or pose2 is None:
        return 0

    if len(pose1) != len(pose2):
        return 0

    # 重要关节点索引
    important_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]

    similarities = []
    for joint in important_joints:
        if joint < len(pose1) and joint < len(pose2):
            if pose1[joint][3] > 0.5 and pose2[joint][3] > 0.5:  # 检查可见性
                distance = np.linalg.norm(pose1[joint][:2] - pose2[joint][:2])
                similarity = max(0, 1 - distance * 2)  # 调整敏感度
                similarities.append(similarity)

    return np.mean(similarities) if similarities else 0


# 获取反馈信息
def get_feedback(similarity):
    if similarity >= 0.9:
        return "🌟 PERFECT! AMAZING!", "#00ff41"
    elif similarity >= 0.8:
        return "⭐ EXCELLENT! GREAT JOB!", "#00ff41"
    elif similarity >= 0.7:
        return "🔥 SUPER! KEEP IT UP!", "#ffff00"
    elif similarity >= 0.6:
        return "👍 GOOD! NICE MOVES!", "#ff9500"
    elif similarity >= 0.4:
        return "💪 OK! TRY HARDER!", "#ff6b35"
    else:
        return "😅 MISS! FOLLOW THE MOVES!", "#ff3333"


# 初始化会话状态
def initialize_session_state():
    if 'mp_pose' not in st.session_state:
        st.session_state.mp_pose, st.session_state.pose, st.session_state.mp_drawing = initialize_mediapipe()

    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    if 'reference_poses' not in st.session_state:
        st.session_state.reference_poses = []

    if 'user_poses' not in st.session_state:
        st.session_state.user_poses = []

    if 'similarity_history' not in st.session_state:
        st.session_state.similarity_history = []

    if 'current_score' not in st.session_state:
        st.session_state.current_score = 0

    if 'skill_scores' not in st.session_state:
        st.session_state.skill_scores = {
            'timing': 0,
            'movement': 0,
            'energy': 0,
            'accuracy': 0
        }


# 主应用
def main():
    initialize_session_state()

    # 标题
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #00ff41; font-size: 3rem; text-shadow: 0 0 20px #00ff41;">
            🎭 DANCE MASTER CHALLENGE 🕺
        </h1>
        <p style="color: #ffffff; font-size: 1.2rem;">
            AI-Powered Real-time Dance Evaluation System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏控制
    with st.sidebar:
        st.markdown("## 🎮 Control Panel")

        # 参考视频上传
        st.markdown("### 📹 Reference Video")
        uploaded_video = st.file_uploader(
            "Upload your reference dance video",
            type=['mp4', 'avi', 'mov', 'mkv']
        )

        # 摄像头控制
        st.markdown("### 📷 Webcam Control")
        webcam_start = st.button("🚀 Start Webcam", type="primary")
        webcam_stop = st.button("⏹️ Stop Webcam")

        if webcam_start:
            st.session_state.webcam_running = True
        if webcam_stop:
            st.session_state.webcam_running = False

        # 显示模式
        st.markdown("### 🎨 Display Mode")
        show_skeleton_only = st.checkbox("Show skeleton only", False)

        # 评分设置
        st.markdown("### ⚙️ Scoring Settings")
        sensitivity = st.slider("Sensitivity", 0.1, 2.0, 1.0, 0.1)

        # 统计信息
        st.markdown("### 📊 Statistics")
        st.metric("Current Score", f"{st.session_state.current_score:.1f}")
        st.metric("Best Score",
                  f"{max(st.session_state.similarity_history) * 100 if st.session_state.similarity_history else 0:.1f}")
        st.metric("Sessions", len(st.session_state.similarity_history))

    # 主内容区域
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Reference Dance")
        ref_placeholder = st.empty()

        if uploaded_video is not None:
            # 保存上传的视频
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            # 处理参考视频
            cap = cv2.VideoCapture(video_path)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame, keypoints = detect_pose(
                        frame, st.session_state.pose,
                        st.session_state.mp_pose, st.session_state.mp_drawing
                    )

                    if keypoints is not None:
                        st.session_state.reference_poses.append(keypoints)

                    # 显示处理后的帧
                    ref_placeholder.image(
                        cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                        caption="Reference dance with pose detection",
                        use_column_width=True
                    )

            cap.release()
            os.unlink(video_path)  # 清理临时文件
        else:
            ref_placeholder.markdown("""
            <div style="text-align: center; padding: 3rem; border: 2px dashed #666; border-radius: 10px;">
                <h3 style="color: #888;">📺 Upload a reference video</h3>
                <p style="color: #666;">Select a dance video to compare with your performance</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🤳 Your Performance")
        user_placeholder = st.empty()

        if st.session_state.webcam_running:
            # 使用camera_input (Streamlit的内置摄像头功能)
            camera_input = st.camera_input("Take a picture for pose detection")

            if camera_input is not None:
                # 处理摄像头输入
                image = Image.open(camera_input)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                processed_frame, keypoints = detect_pose(
                    frame, st.session_state.pose,
                    st.session_state.mp_pose, st.session_state.mp_drawing
                )

                if keypoints is not None:
                    st.session_state.user_poses.append(keypoints)

                user_placeholder.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    caption="Your performance with pose detection",
                    use_column_width=True
                )
        else:
            user_placeholder.markdown("""
            <div style="text-align: center; padding: 3rem; border: 2px dashed #666; border-radius: 10px;">
                <h3 style="color: #888;">📷 Start webcam to begin</h3>
                <p style="color: #666;">Click 'Start Webcam' in the sidebar to start dancing!</p>
            </div>
            """, unsafe_allow_html=True)

    # 实时评分区域
    if st.session_state.reference_poses and st.session_state.user_poses:
        # 计算相似度
        latest_ref = st.session_state.reference_poses[-1]
        latest_user = st.session_state.user_poses[-1]
        similarity = calculate_similarity(latest_ref, latest_user)

        score = similarity * 100
        st.session_state.current_score = score
        st.session_state.similarity_history.append(similarity)

        # 更新技能分数（模拟）
        st.session_state.skill_scores['timing'] = min(100, score + np.random.randint(-10, 15))
        st.session_state.skill_scores['movement'] = min(100, score + np.random.randint(-15, 10))
        st.session_state.skill_scores['energy'] = min(100, score + np.random.randint(-5, 20))
        st.session_state.skill_scores['accuracy'] = score

        # 显示评分
        st.markdown("---")
        st.markdown("## 🏆 Live Scoring Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            feedback_text, feedback_color = get_feedback(similarity)
            st.markdown(f"""
            <div class="metric-card">
                <div class="score-display">{score:.0f}</div>
                <div style="text-align: center; color: white;">Current Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="feedback-text" style="color: {feedback_color};">{feedback_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_score = np.mean(st.session_state.similarity_history) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: white; text-align: center;">{avg_score:.1f}</h3>
                <div style="text-align: center; color: #ccc;">Average Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            best_score = max(st.session_state.similarity_history) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: white; text-align: center;">{best_score:.1f}</h3>
                <div style="text-align: center; color: #ccc;">Best Score</div>
            </div>
            """, unsafe_allow_html=True)

        # 技能分解图表
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Skill Breakdown")
            skills_df = pd.DataFrame(
                list(st.session_state.skill_scores.items()),
                columns=['Skill', 'Score']
            )

            fig = px.bar(
                skills_df, x='Skill', y='Score',
                title="Performance by Category",
                color='Score',
                color_continuous_scale='viridis',
                range_y=[0, 100]
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📈 Performance History")
            if len(st.session_state.similarity_history) > 1:
                history_df = pd.DataFrame({
                    'Time': range(len(st.session_state.similarity_history)),
                    'Score': [s * 100 for s in st.session_state.similarity_history]
                })

                fig = px.line(
                    history_df, x='Time', y='Score',
                    title="Score Over Time",
                    range_y=[0, 100]
                )
                fig.update_traces(line_color='#00ff41', line_width=3)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #888;">
        <p>🎭 Dance Master Challenge - AI-Powered Dance Evaluation System</p>
        <p>Built with Streamlit, MediaPipe, and ❤️</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()