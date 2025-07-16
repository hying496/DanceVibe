import numpy as np
from typing import List, Optional

class Smoother:
    """
    平滑滤波基类
    """
    def reset(self):
        pass

    def smooth(self, keypoints: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError

class EMASmoother(Smoother):
    """
    指数加权平均滤波器
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.last = None

    def reset(self):
        self.last = None

    def smooth(self, keypoints: List[np.ndarray]) -> List[np.ndarray]:
        smoothed = [] 
        for kp in keypoints:
            if self.last is None:
                self.last = kp.copy()
                smoothed.append(self.last.copy())
            else:
                self.last = self.alpha * kp + (1 - self.alpha) * self.last
                smoothed.append(self.last.copy())
        return smoothed

class KalmanSmoother(Smoother):
    """
    简单卡尔曼滤波器（对每个关键点独立处理）
    """
    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.x = None  # 状态
        self.P = None  # 协方差

    def reset(self):
        self.x = None
        self.P = None

    def smooth(self, keypoints: List[np.ndarray]) -> List[np.ndarray]:
        smoothed = []
        for z in keypoints:
            if self.x is None:
                self.x = z.copy()
                self.P = np.eye(len(z))
                smoothed.append(self.x.copy())
            else:
                # 预测
                self.x = self.x
                self.P = self.P + self.process_noise * np.eye(len(z))
                # 更新
                K = self.P @ np.linalg.inv(self.P + self.measurement_noise * np.eye(len(z)))
                self.x = self.x + K @ (z - self.x)
                self.P = (np.eye(len(z)) - K) @ self.P
                smoothed.append(self.x.copy())
        return smoothed

def smooth_keypoints(
    keypoints_seq: List[List[np.ndarray]],
    method: str = 'ema',
    **kwargs
) -> List[List[np.ndarray]]:
    """
    对多帧多人的关键点序列进行平滑处理
    :param keypoints_seq: 形如[帧][人][关键点坐标]的序列
    :param method: 'ema' 或 'kalman'
    :param kwargs: 传递给滤波器的参数
    :return: 平滑后的关键点序列
    """
    if method == 'ema':
        smoother_cls = EMASmoother
    elif method == 'kalman':
        smoother_cls = KalmanSmoother
    else:
        raise ValueError(f'未知平滑方法: {method}')

    # 假设每个人的数量固定，分别对每个人的关键点序列做滤波
    if not keypoints_seq:
        return []
    num_persons = len(keypoints_seq[0])
    num_frames = len(keypoints_seq)
    results = [[] for _ in range(num_persons)]
    for person_idx in range(num_persons):
        # 收集该人的所有帧的关键点
        person_kps = [keypoints_seq[frame_idx][person_idx] for frame_idx in range(num_frames)]
        smoother = smoother_cls(**kwargs)
        smoothed = smoother.smooth(person_kps)
        for frame_idx, kp in enumerate(smoothed):
            results[person_idx].append(kp)
    # 转置回[帧][人][关键点]
    results_by_frame = [[results[person_idx][frame_idx] for person_idx in range(num_persons)] for frame_idx in range(num_frames)]
    return results_by_frame
