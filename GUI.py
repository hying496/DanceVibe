import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import time


class EnhancedScoringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Dance Scoring System")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1a1a1a')  # 深色主题

        # 模拟评分数据
        self.current_score = 0
        self.max_score = 100
        self.similarity_history = []
        self.feedback_messages = []

        self.setup_enhanced_gui()

    def setup_enhanced_gui(self):
        # 主容器
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 顶部标题栏
        title_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(title_frame, text="🎭 DANCE MASTER CHALLENGE 🎭",
                               font=("Arial", 24, "bold"), fg='#00ff00', bg='#2d2d2d')
        title_label.pack(pady=15)

        # 中间内容区域
        content_frame = tk.Frame(main_frame, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧视频区域
        left_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 右侧评分区域
        right_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_panel.pack_propagate(False)

        self.setup_video_panels(left_panel)
        self.setup_scoring_panel(right_panel)

        # 底部控制栏
        self.setup_control_bar(main_frame)

    def setup_video_panels(self, parent):
        # 视频标题
        video_title = tk.Label(parent, text="📹 DANCE COMPARISON",
                               font=("Arial", 16, "bold"), fg='#ffffff', bg='#2d2d2d')
        video_title.pack(pady=10)

        # 视频容器
        video_container = tk.Frame(parent, bg='#2d2d2d')
        video_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 参考视频
        ref_frame = tk.Frame(video_container, bg='#1a1a1a', relief=tk.SUNKEN, bd=2)
        ref_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(ref_frame, text="🎯 Reference Dance", font=("Arial", 12, "bold"),
                 fg='#00ffff', bg='#1a1a1a').pack(pady=5)

        self.ref_canvas = tk.Canvas(ref_frame, width=550, height=400, bg="black")
        self.ref_canvas.pack(pady=5)
        self.ref_canvas.create_text(275, 200, text="📺 Load Reference Video",
                                    fill="gray", font=("Arial", 14))

        # 用户视频
        user_frame = tk.Frame(video_container, bg='#1a1a1a', relief=tk.SUNKEN, bd=2)
        user_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        tk.Label(user_frame, text="🤳 Your Performance", font=("Arial", 12, "bold"),
                 fg='#ff00ff', bg='#1a1a1a').pack(pady=5)

        self.user_canvas = tk.Canvas(user_frame, width=550, height=400, bg="black")
        self.user_canvas.pack(pady=5)
        self.user_canvas.create_text(275, 200, text="📷 Start Webcam",
                                     fill="gray", font=("Arial", 14))

    def setup_scoring_panel(self, parent):
        # 评分标题
        score_title = tk.Label(parent, text="🏆 LIVE SCORING",
                               font=("Arial", 18, "bold"), fg='#ffff00', bg='#2d2d2d')
        score_title.pack(pady=15)

        # 当前分数显示
        score_frame = tk.Frame(parent, bg='#1a1a1a', relief=tk.RAISED, bd=3)
        score_frame.pack(fill=tk.X, padx=15, pady=10)

        tk.Label(score_frame, text="CURRENT SCORE", font=("Arial", 12),
                 fg='#ffffff', bg='#1a1a1a').pack(pady=5)

        self.score_label = tk.Label(score_frame, text="0", font=("Arial", 48, "bold"),
                                    fg='#00ff00', bg='#1a1a1a')
        self.score_label.pack(pady=10)

        # 进度条
        progress_frame = tk.Frame(parent, bg='#2d2d2d')
        progress_frame.pack(fill=tk.X, padx=15, pady=10)

        tk.Label(progress_frame, text="🎯 Accuracy", font=("Arial", 12),
                 fg='#ffffff', bg='#2d2d2d').pack()

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=300)
        self.progress_bar.pack(pady=5)

        # 实时反馈
        feedback_frame = tk.Frame(parent, bg='#1a1a1a', relief=tk.RAISED, bd=3)
        feedback_frame.pack(fill=tk.X, padx=15, pady=10)

        tk.Label(feedback_frame, text="💬 LIVE FEEDBACK", font=("Arial", 12, "bold"),
                 fg='#ffffff', bg='#1a1a1a').pack(pady=5)

        self.feedback_label = tk.Label(feedback_frame, text="Ready to dance!",
                                       font=("Arial", 14, "bold"), fg='#00ffff', bg='#1a1a1a')
        self.feedback_label.pack(pady=10)

        # 技能分解
        skills_frame = tk.Frame(parent, bg='#2d2d2d')
        skills_frame.pack(fill=tk.X, padx=15, pady=10)

        tk.Label(skills_frame, text="📊 SKILL BREAKDOWN", font=("Arial", 12, "bold"),
                 fg='#ffffff', bg='#2d2d2d').pack(pady=5)

        # 各项技能评分
        self.setup_skill_meters(skills_frame)

        # 历史记录
        history_frame = tk.Frame(parent, bg='#1a1a1a', relief=tk.RAISED, bd=3)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        tk.Label(history_frame, text="📈 PERFORMANCE HISTORY", font=("Arial", 10, "bold"),
                 fg='#ffffff', bg='#1a1a1a').pack(pady=5)

        self.history_canvas = tk.Canvas(history_frame, width=350, height=150, bg='#000000')
        self.history_canvas.pack(pady=5)

    def setup_skill_meters(self, parent):
        skills = [
            ("🕺 Timing", "timing"),
            ("🤸 Movement", "movement"),
            ("⚡ Energy", "energy"),
            ("🎯 Accuracy", "accuracy")
        ]

        self.skill_vars = {}
        self.skill_labels = {}

        for skill_name, skill_key in skills:
            skill_row = tk.Frame(parent, bg='#2d2d2d')
            skill_row.pack(fill=tk.X, pady=2)

            tk.Label(skill_row, text=skill_name, font=("Arial", 9),
                     fg='#ffffff', bg='#2d2d2d', width=12, anchor='w').pack(side=tk.LEFT)

            var = tk.DoubleVar()
            self.skill_vars[skill_key] = var

            meter = ttk.Progressbar(skill_row, variable=var, maximum=100, length=200)
            meter.pack(side=tk.LEFT, padx=5)

            label = tk.Label(skill_row, text="0%", font=("Arial", 9),
                             fg='#00ff00', bg='#2d2d2d', width=4)
            label.pack(side=tk.RIGHT)
            self.skill_labels[skill_key] = label

    def setup_control_bar(self, parent):
        control_frame = tk.Frame(parent, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        # 左侧控制
        left_controls = tk.Frame(control_frame, bg='#2d2d2d')
        left_controls.pack(side=tk.LEFT, padx=20, pady=10)

        tk.Button(left_controls, text="📁 Load Video", font=("Arial", 12),
                  bg='#0066cc', fg='white', width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(left_controls, text="📷 Start Cam", font=("Arial", 12),
                  bg='#00cc66', fg='white', width=12).pack(side=tk.LEFT, padx=5)

        # 中央显示模式
        center_controls = tk.Frame(control_frame, bg='#2d2d2d')
        center_controls.pack(side=tk.LEFT, expand=True)

        tk.Label(center_controls, text="🎭 DANCE CHALLENGE MODE",
                 font=("Arial", 14, "bold"), fg='#ffff00', bg='#2d2d2d').pack()

        # 右侧控制
        right_controls = tk.Frame(control_frame, bg='#2d2d2d')
        right_controls.pack(side=tk.RIGHT, padx=20, pady=10)

        tk.Button(right_controls, text="⏸️ Pause", font=("Arial", 12),
                  bg='#cc6600', fg='white', width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(right_controls, text="🔄 Reset", font=("Arial", 12),
                  bg='#cc0066', fg='white', width=12).pack(side=tk.LEFT, padx=5)

        # 开始演示动画
        self.start_demo_animation()

    def start_demo_animation(self):
        """启动演示动画"""

        def animate():
            while True:
                # 模拟评分变化
                new_score = np.random.randint(60, 100)
                self.update_score(new_score)

                # 模拟技能评分
                skills = {
                    'timing': np.random.randint(70, 100),
                    'movement': np.random.randint(60, 95),
                    'energy': np.random.randint(80, 100),
                    'accuracy': np.random.randint(65, 90)
                }
                self.update_skills(skills)

                # 模拟反馈
                feedbacks = ["PERFECT! 🌟", "GREAT! ⭐", "GOOD! 👍", "AMAZING! 🔥", "EXCELLENT! ⚡"]
                self.update_feedback(np.random.choice(feedbacks))

                time.sleep(1)

        threading.Thread(target=animate, daemon=True).start()

    def update_score(self, score):
        """更新分数显示"""
        self.current_score = score
        self.score_label.config(text=str(score))
        self.progress_var.set(score)

        # 根据分数改变颜色
        if score >= 90:
            self.score_label.config(fg='#00ff00')  # 绿色
        elif score >= 70:
            self.score_label.config(fg='#ffff00')  # 黄色
        else:
            self.score_label.config(fg='#ff6600')  # 橙色

    def update_skills(self, skills):
        """更新技能评分"""
        for skill, value in skills.items():
            if skill in self.skill_vars:
                self.skill_vars[skill].set(value)
                self.skill_labels[skill].config(text=f"{value}%")

    def update_feedback(self, message):
        """更新反馈信息"""
        self.feedback_label.config(text=message)

        # 添加到历史记录
        self.similarity_history.append(self.current_score)
        if len(self.similarity_history) > 50:
            self.similarity_history.pop(0)

        self.draw_history_chart()

    def draw_history_chart(self):
        """绘制历史记录图表"""
        self.history_canvas.delete("all")

        if len(self.similarity_history) < 2:
            return

        width = 350
        height = 150

        # 绘制网格
        for i in range(0, width, 50):
            self.history_canvas.create_line(i, 0, i, height, fill='#333333', width=1)
        for i in range(0, height, 30):
            self.history_canvas.create_line(0, i, width, i, fill='#333333', width=1)

        # 绘制数据线
        points = []
        for i, score in enumerate(self.similarity_history):
            x = (i / len(self.similarity_history)) * width
            y = height - (score / 100) * height
            points.extend([x, y])

        if len(points) >= 4:
            self.history_canvas.create_line(points, fill='#00ff00', width=2, smooth=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedScoringGUI(root)
    root.mainloop()