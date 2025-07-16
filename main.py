# Import necessary libraries
import sys
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import mediapipe as mp
import importlib.util
sys.path.append(os.path.join(os.path.dirname(__file__), 'detector'))
from detector.pose_detector import DetectorType, PoseDetectionManager
from detector.smoother import smooth_keypoints
from detector.fixer import fix_keypoints
from detector.tracker import Tracker
# 导入相似度计算
import numpy as np
from score.similarity import calculate_pose_similarity, center_landmarks, normalize_landmarks
from score.music_beat import mp4_2_mp3, get_beats
from score.motion_match import match_motion_to_beats
from score.score_pose import score_pose
from score.average_similarity import CumulativeScore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress MediaPipe warnings
import logging

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'，需本地有该字体
matplotlib.rcParams['axes.unicode_minus'] = False

logging.getLogger('mediapipe').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 标准骨架连接（MediaPipe 33点）
MEDIAPIPE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(27,29),
    (29,31),(26,28),(28,30),(30,32)
]
# YOLOv8/COCO 17点骨架连接
YOLOV8_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# 优化后的可视化函数
def draw_persons(frame, persons, detector_type):
    color_list = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
    # 选择骨架连接
    if detector_type.value == 'mediapipe':
        connections = MEDIAPIPE_CONNECTIONS
    else:
        connections = YOLOV8_CONNECTIONS
    for idx, p in enumerate(persons):
        color = color_list[idx % len(color_list)]
        # 画骨架
        for pt1, pt2 in connections:
            if pt1 < len(p.keypoints) and pt2 < len(p.keypoints):
                kp1, kp2 = p.keypoints[pt1], p.keypoints[pt2]
                if kp1.visible and kp2.visible:
                    cv2.line(frame, (int(kp1.x), int(kp1.y)), (int(kp2.x), int(kp2.y)), color, 2)
        # 画关键点
        for kp in p.keypoints:
            if kp.visible:
                cv2.circle(frame, (int(kp.x), int(kp.y)), 4, color, -1)
        # 画手部
        for hand in getattr(p, 'hands', []):
            for kp in hand.keypoints:
                cv2.circle(frame, (int(kp.x), int(kp.y)), 2, (255,0,255), -1)
        # 显示ID
        if hasattr(p, 'id'):
            bbox = p.bbox
            cv2.putText(frame, f'ID:{p.id}', (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def calc_similarity(lm1, lm2):
    return calculate_pose_similarity(lm1, lm2)

def center_landmarks_safe(landmarks):
    return center_landmarks(landmarks)
def normalize_landmarks_safe(landmarks_np):
    return normalize_landmarks(landmarks_np)

class MediaPipePoseApp:
    def __init__(self, root,
                 detector_type=DetectorType.MEDIAPIPE,
                 smoother_method=None,
                 fixer_method=None,
                 tracker_enable=False):
        self.root = root
        self.root.title("MediaPipe Dance GUI")
        # 自动最大化窗口
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux/Mac
            except:
                pass
        self.root.configure(bg='lightgray')

        # 参数
        self.detector_type = detector_type
        self.smoother_method = smoother_method  # 'ema'/'kalman'/None
        self.fixer_method = fixer_method        # 'linear'/'symmetric'/None
        self.tracker_enable = tracker_enable

        # 分别为左/右流维护独立对象
        self.file_pose_manager = PoseDetectionManager(detector_type)
        self.file_tracker = Tracker() if tracker_enable else None
        self.cam_pose_manager = PoseDetectionManager(detector_type)
        self.cam_tracker = Tracker() if tracker_enable else None

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        self.similarity_calculator = SimilarityCalculator()
        self.last_file_landmarks = None  # 记录参考视频主舞者landmarks
        self.last_cam_landmarks = None   # 记录webcam主舞者landmarks
        self.last_similarity = None

        self.beat_times = []
        self.cumulative_score = CumulativeScore()
        self.score_history = []
        self.fig, self.ax = plt.subplots(figsize=(4,1.2), dpi=100)
        self.score_canvas = None

        self.setup_gui()

    def setup_gui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='lightgray')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for reference video
        self.left_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right frame for webcam
        self.right_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Left side - Reference Video
        tk.Label(self.left_frame, text="Reference Video", font=("Arial", 16, "bold"), bg='white').pack(pady=5)

        video_frame = tk.Frame(self.left_frame, bg='white')
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Canvas自适应
        self.canvas_file = tk.Canvas(video_frame, bg="black")
        self.canvas_file.pack(fill=tk.BOTH, expand=True, pady=5)
        self.canvas_file.create_text(350, 262, text="No video loaded", fill="white", font=("Arial", 12))

        self.controls_file = tk.Frame(self.left_frame, bg='white')
        self.controls_file.pack(side=tk.BOTTOM, pady=10)
        tk.Button(self.controls_file, text="📁 Open Video", command=self.load_video,
                  bg='lightblue', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="▶️ Start Video", command=self.start_video,
                  bg='lightgreen', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="⏹️ Stop Video", command=self.stop_video,
                  bg='lightcoral', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="🔄 Show/Hide", command=self.toggle_video_display,
                  bg='lightyellow', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)

        # Right side - Webcam
        tk.Label(self.right_frame, text="Your Webcam", font=("Arial", 16, "bold"), bg='white').pack(pady=5)
        cam_frame = tk.Frame(self.right_frame, bg='white')
        cam_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.canvas_cam = tk.Canvas(cam_frame, bg="black")
        self.canvas_cam.pack(fill=tk.BOTH, expand=True, pady=5)
        self.canvas_cam.create_text(350, 262, text="Webcam not started", fill="white", font=("Arial", 12))

        self.controls_cam = tk.Frame(self.right_frame, bg='white')
        self.controls_cam.pack(side=tk.BOTTOM, pady=10)
        tk.Button(self.controls_cam, text="📷 Start Webcam", command=self.start_cam,
                  bg='lightgreen', font=("Arial", 10), width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="⏹️ Stop Webcam", command=self.stop_cam,
                  bg='lightcoral', font=("Arial", 10), width=15).pack(side=tk.LEFT, padx=5)

        # Bottom info frame
        self.info_frame = tk.Frame(self.root, bg='lightgray')
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.info_label = tk.Label(self.info_frame,
                                   text="✅ Ready to start! Click 'Start Webcam' to test pose detection.",
                                   font=("Arial", 12), bg='lightgray', fg='darkgreen')
        self.info_label.pack()
        self.status_frame = tk.Frame(self.info_frame, bg='lightgray')
        self.status_frame.pack(pady=5)
        self.video_status = tk.Label(self.status_frame, text="📹 Video: Stopped",
                                     font=("Arial", 10), bg='lightgray')
        self.video_status.pack(side=tk.LEFT, padx=10)
        self.cam_status = tk.Label(self.status_frame, text="📷 Webcam: Stopped",
                                   font=("Arial", 10), bg='lightgray')
        self.cam_status.pack(side=tk.LEFT, padx=10)
        # 分数区自适应
        self.score_frame = tk.Frame(self.root, bg='white', relief=tk.RAISED, bd=2)
        self.score_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        tk.Label(self.score_frame, text="分数统计", font=("Arial", 14, "bold"), bg='white').pack(pady=5)
        self.pose_score_var = tk.StringVar(value="姿态分: 0.00")
        self.rhythm_score_var = tk.StringVar(value="节奏分: 0.00")
        self.total_score_var = tk.StringVar(value="总分: 0.00")
        self.avg_score_var = tk.StringVar(value="累计平均分: 0.00")
        self.similarity_var = tk.StringVar(value="相似度: 0.00")
        tk.Label(self.score_frame, textvariable=self.similarity_var, font=("Arial", 12), bg='white').pack()
        tk.Label(self.score_frame, textvariable=self.pose_score_var, font=("Arial", 12), bg='white').pack()
        tk.Label(self.score_frame, textvariable=self.rhythm_score_var, font=("Arial", 12), bg='white').pack()
        tk.Label(self.score_frame, textvariable=self.total_score_var, font=("Arial", 12), bg='white').pack()
        tk.Label(self.score_frame, textvariable=self.avg_score_var, font=("Arial", 12), bg='white').pack()
        self.fig, self.ax = plt.subplots(figsize=(4,1.2), dpi=100)
        self.score_canvas = FigureCanvasTkAgg(self.fig, master=self.score_frame)
        self.score_canvas.get_tk_widget().pack(pady=5, fill=tk.BOTH, expand=True)

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Dance Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            filename = os.path.basename(path)
            self.info_label.config(text=f"✅ Video loaded: {filename}")
            # 在Canvas上显示文本
            self.canvas_file.delete("all")
            self.canvas_file.create_text(350, 262, text=f"Video loaded:\n{filename}",
                                         fill="white", font=("Arial", 12))
            # 新增：自动提取节拍锚点
            try:
                audio_path = mp4_2_mp3(path)
                tempo, beats, beat_times = get_beats(audio_path)
                self.beat_times = beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times)
                self.info_label.config(text=f"🎵 节拍锚点提取完成，共{len(self.beat_times)}个")
            except Exception as e:
                self.beat_times = []
                self.info_label.config(text=f"⚠️ 节拍提取失败: {e}")

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file first!")
            return
        if not self.running_file:
            self.running_file = True
            self.video_status.config(text="📹 Video: Playing", fg='green')
            self.info_label.config(text="🎬 Video playing! You can now compare with webcam.")
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False
        self.video_status.config(text="📹 Video: Stopped", fg='red')
        self.info_label.config(text="⏹️ Video stopped.")
        # 在Canvas上显示停止文本
        self.canvas_file.delete("all")
        self.canvas_file.create_text(350, 262, text="Video stopped", fill="white", font=("Arial", 12))
        if self.cap_file:
            self.cap_file.release()

    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame
        mode = "Original video" if self.show_video_frame else "Pose skeleton only"
        self.info_label.config(text=f"🔄 Display mode: {mode}")

    def start_cam(self):
        if not self.running_cam:
            self.running_cam = True
            self.cam_status.config(text="📷 Webcam: Running", fg='green')
            self.info_label.config(text="📷 Webcam started! Move around to test pose detection.")
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False
        self.cam_status.config(text="📷 Webcam: Stopped", fg='red')
        self.info_label.config(text="📷 Webcam stopped.")
        # 在Canvas上显示停止文本
        self.canvas_cam.delete("all")
        self.canvas_cam.create_text(350, 262, text="Webcam stopped", fill="white", font=("Arial", 12))
        if self.cap_cam:
            self.cap_cam.release()

    def process_video_file(self):
        try:
            self.cap_file = cv2.VideoCapture(self.video_path)
            if not self.cap_file.isOpened():
                messagebox.showerror("Error", "Cannot open video file!")
                return

            fps = self.cap_file.get(cv2.CAP_PROP_FPS) or 30
            delay = max(1, int(1000 / fps))

            while self.cap_file.isOpened() and self.running_file:
                ret, frame = self.cap_file.read()
                if not ret:
                    # Video ended, restart from beginning
                    self.cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                processed_frame = self.process_pose(frame, which='file')
                self.update_canvas(self.canvas_file, processed_frame)

                cv2.waitKey(delay)

        except Exception as e:
            messagebox.showerror("Video Error", f"Error processing video: {str(e)}")
        finally:
            if self.cap_file:
                self.cap_file.release()

    def process_webcam(self):
        try:
            self.cap_cam = cv2.VideoCapture(0)
            if not self.cap_cam.isOpened():
                messagebox.showerror("Error", "Cannot open webcam!")
                return

            while self.cap_cam.isOpened() and self.running_cam:
                ret, frame = self.cap_cam.read()
                if not ret:
                    continue

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                processed_frame = self.process_pose(frame, which='cam')
                self.update_canvas(self.canvas_cam, processed_frame)

        except Exception as e:
            messagebox.showerror("Webcam Error", f"Error processing webcam: {str(e)}")
        finally:
            if self.cap_cam:
                self.cap_cam.release()

    def process_pose(self, frame, which='file'):
        """集成自定义检测器/滤波/补全/追踪，支持多人、手势、优化前后对比"""
        import copy
        height, width = frame.shape[:2]
        # 选择对应对象
        if which == 'file':
            pose_manager = self.file_pose_manager
            tracker = self.file_tracker
        else:
            pose_manager = self.cam_pose_manager
            tracker = self.cam_tracker
        # 检测
        persons, det_info = pose_manager.detect_poses(frame)
        # 记录原始关键点
        orig_persons = copy.deepcopy(persons)
        # 遮挡补全
        if self.fixer_method and persons:
            keypoints_seq = [[np.array([kp.x, kp.y]) for kp in p.keypoints] for p in persons]
            conf_seq = [[np.array([kp.confidence for kp in p.keypoints])] for p in persons]
            if keypoints_seq and all(len(kps) >= 1 for kps in keypoints_seq):
                fixed_kps = fix_keypoints([keypoints_seq], [conf_seq], method=self.fixer_method)[0]
                for i, p in enumerate(persons):
                    for j, kp in enumerate(p.keypoints):
                        kp.x, kp.y = fixed_kps[i][j][0], fixed_kps[i][j][1]
        # 平滑滤波
        if self.smoother_method and persons:
            keypoints_seq = [[np.array([kp.x, kp.y]) for kp in p.keypoints] for p in persons]
            if keypoints_seq and all(len(kps) >= 1 for kps in keypoints_seq):
                smoothed_kps = smooth_keypoints([keypoints_seq], method=self.smoother_method)[0]
                for i, p in enumerate(persons):
                    for j, kp in enumerate(p.keypoints):
                        kp.x, kp.y = smoothed_kps[i][j][0], smoothed_kps[i][j][1]
        # 多人追踪
        if self.tracker_enable and persons and tracker is not None:
            dets = [{'keypoints': np.array([[kp.x, kp.y] for kp in p.keypoints]), 'bbox': np.array(p.bbox)} for p in persons]
            tracked = tracker.update(dets, 0)
            for i, p in enumerate(persons):
                p.id = tracked[i]['id'] if i < len(tracked) else p.id
        # 可视化
        if self.show_video_frame:
            output_frame = frame.copy()
        else:
            output_frame = np.ones_like(frame) * 255
        output_frame = draw_persons(output_frame, persons, self.detector_type)
        fps = 1000.0 / max(det_info.get('processing_time_ms', 1), 1)
        cv2.putText(output_frame, f"Detector: {self.detector_type.value} | Smoother: {self.smoother_method or 'None'} | Fixer: {self.fixer_method or 'None'} | Tracker: {self.tracker_enable}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        cv2.putText(output_frame, f"FPS: {fps:.1f} | Persons: {len(persons)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        # 记录主舞者landmarks
        # 健壮性判断，确保有主舞者且关键点数量足够
        if persons and hasattr(persons[0], 'keypoints') and persons[0].keypoints:
            kps = persons[0].keypoints
            if len(kps) < 33:
                class DummyKP:
                    x, y, z = 0.0, 0.0, 0.0
                kps = list(kps) + [DummyKP() for _ in range(33 - len(kps))]
            class LM:
                pass
            lm = LM()
            class L:
                pass
            lm.landmark = [L() for _ in range(33)]
            for i in range(33):
                kp = kps[i]
                lm.landmark[i].x = getattr(kp, 'x', 0.0)
                lm.landmark[i].y = getattr(kp, 'y', 0.0)
                lm.landmark[i].z = getattr(kp, 'z', 0.0)
            if which == 'file':
                self.last_file_landmarks = lm
            else:
                self.last_cam_landmarks = lm
        else:
            if which == 'file':
                self.last_file_landmarks = None
            else:
                self.last_cam_landmarks = None
        # 计算相似度
        similarity = None
        if self.last_file_landmarks and self.last_cam_landmarks:
            similarity = calc_similarity(self.last_file_landmarks, self.last_cam_landmarks)
            self.last_similarity = similarity
        if self.last_similarity is not None:
            cv2.putText(output_frame, f"Similarity: {self.last_similarity:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        pose_score = rhythm_score = total_score = avg_score = 0.0
        delta_t = 1.0  # 先给默认值，防止未定义
        if self.last_file_landmarks and self.last_cam_landmarks:
            pose_score = calc_similarity(self.last_file_landmarks, self.last_cam_landmarks)
            rhythm_score = 0.0
            if self.beat_times and which == 'cam':
                frame_idx = len(self.score_history)
                current_time = frame_idx / 30.0
                try:
                    delta_t = min([abs(current_time - t) for t in self.beat_times])
                except:
                    delta_t = 1.0
                rhythm_score = max(0, 1 - delta_t / 0.4)
            total_score = score_pose(pose_score, delta_t if self.beat_times else 1.0)
            self.cumulative_score.update(total_score)
            avg_score = self.cumulative_score.average
            self.score_history.append(total_score)
        self.similarity_var.set(f"相似度: {self.last_similarity:.2f}" if self.last_similarity is not None else "相似度: 0.00")
        self.pose_score_var.set(f"姿态分: {pose_score:.2f}")
        self.rhythm_score_var.set(f"节奏分: {rhythm_score:.2f}")
        self.total_score_var.set(f"总分: {total_score:.2f}")
        self.avg_score_var.set(f"累计平均分: {avg_score:.2f}")
        self.update_score_plot()
        return output_frame

    def update_canvas(self, canvas, frame):
        try:
            canvas.update_idletasks()
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((w, h))
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.delete("all")
            canvas.create_image(w//2, h//2, image=imgtk)
            canvas.imgtk = imgtk
        except Exception as e:
            print(f"Error updating canvas: {e}")

    def update_score_plot(self):
        self.ax.clear()
        self.ax.plot(self.score_history, color='orange', label='总分')
        self.ax.set_ylim(0, 1.05)
        self.ax.set_ylabel('分数')
        self.ax.set_xlabel('帧')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.fig.tight_layout()
        self.score_canvas.draw()

    def cleanup(self):
        """Cleanup resources"""
        self.running_file = False
        self.running_cam = False
        if self.cap_file:
            self.cap_file.release()
        if self.cap_cam:
            self.cap_cam.release()
        if hasattr(self, 'pose'):
            self.pose.close()


# 用于姿态相似度计算的工具类
class SimilarityCalculator:
    def __init__(self):
        pass

    def calculate(self, lm1, lm2):
        return calculate_pose_similarity(lm1, lm2)

    def center_landmarks(self, landmarks):
        return center_landmarks(landmarks)

    def normalize_landmarks(self, landmarks_np):
        return normalize_landmarks(landmarks_np)


def main():
    # 可通过命令行或配置文件切换参数，这里用默认参数
    root = tk.Tk()
    app = MediaPipePoseApp(root,
                          detector_type=DetectorType.MEDIAPIPE,  # 可选: MEDIAPIPE/YOLOV8/HYBRID
                          smoother_method=None,                   # 可选: 'ema'/'kalman'/None
                          fixer_method=None,                      # 可选: 'linear'/'symmetric'/None
                          tracker_enable=True)                   # True/False
    def on_closing():
        app.cleanup()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    print("🚀 Starting MediaPipe Dance GUI... (with custom pipeline)")
    root.mainloop()


if __name__ == "__main__":
    main()