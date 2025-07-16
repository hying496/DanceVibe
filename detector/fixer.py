import numpy as np
from typing import List, Optional

def fix_keypoints(
    keypoints_seq: List[List[np.ndarray]],
    conf_seq: List[List[np.ndarray]],
    method: str = 'linear',
    conf_threshold: float = 0.3
) -> List[List[np.ndarray]]:
    """
    对多帧多人的关键点序列进行遮挡点补全
    :param keypoints_seq: [帧][人][关键点坐标]
    :param conf_seq: [帧][人][关键点置信度]
    :param method: 'linear' 或 'symmetric'
    :param conf_threshold: 置信度阈值
    :return: 补全后的关键点序列
    """
    if method == 'linear':
        return linear_interpolate_missing(keypoints_seq, conf_seq, conf_threshold)
    elif method == 'symmetric':
        return symmetric_fix_missing(keypoints_seq, conf_seq, conf_threshold)
    else:
        raise ValueError(f'未知补全方法: {method}')

def linear_interpolate_missing(
    keypoints_seq: List[List[np.ndarray]],
    conf_seq: List[List[np.ndarray]],
    conf_threshold: float
) -> List[List[np.ndarray]]:
    """
    对每个关键点序列做线性插值补全
    """
    num_frames = len(keypoints_seq)
    num_persons = len(keypoints_seq[0])
    num_keypoints = len(keypoints_seq[0][0])
    fixed_seq = [[kp.copy() for kp in person] for person in keypoints_seq[0:]]
    for person_idx in range(num_persons):
        for kp_idx in range(num_keypoints):
            # 收集该人该关键点的所有帧的置信度
            confs = [conf_seq[frame_idx][person_idx][kp_idx] for frame_idx in range(num_frames)]
            coords = [keypoints_seq[frame_idx][person_idx][kp_idx] for frame_idx in range(num_frames)]
            # 找到有效帧
            valid = [i for i, c in enumerate(confs) if c >= conf_threshold]
            for i in range(num_frames):
                if confs[i] < conf_threshold:
                    # 前后找最近的有效帧做插值
                    prev = next((j for j in range(i-1, -1, -1) if j in valid), None)
                    next_ = next((j for j in range(i+1, num_frames) if j in valid), None)
                    if prev is not None and next_ is not None:
                        ratio = (i - prev) / (next_ - prev)
                        fixed_seq[i][person_idx][kp_idx] = coords[prev] * (1 - ratio) + coords[next_] * ratio
                    elif prev is not None:
                        fixed_seq[i][person_idx][kp_idx] = coords[prev]
                    elif next_ is not None:
                        fixed_seq[i][person_idx][kp_idx] = coords[next_]
                    # 否则保持原值
    return fixed_seq

# MediaPipe 33点对称关系（左:右）
SYMMETRIC_PAIRS = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]

def symmetric_fix_missing(
    keypoints_seq: List[List[np.ndarray]],
    conf_seq: List[List[np.ndarray]],
    conf_threshold: float
) -> List[List[np.ndarray]]:
    """
    用对称点补全丢失关键点
    """
    num_frames = len(keypoints_seq)
    num_persons = len(keypoints_seq[0])
    num_keypoints = len(keypoints_seq[0][0])
    fixed_seq = [[kp.copy() for kp in person] for person in keypoints_seq[0:]]
    for frame_idx in range(num_frames):
        for person_idx in range(num_persons):
            for left, right in SYMMETRIC_PAIRS:
                # 左右点分别判断
                left_conf = conf_seq[frame_idx][person_idx][left]
                right_conf = conf_seq[frame_idx][person_idx][right]
                if left_conf < conf_threshold and right_conf >= conf_threshold:
                    # 用右点对称补左点
                    fixed_seq[frame_idx][person_idx][left] = fixed_seq[frame_idx][person_idx][right].copy()
                    fixed_seq[frame_idx][person_idx][left][0] *= -1  # x轴对称，需根据实际坐标系调整
                elif right_conf < conf_threshold and left_conf >= conf_threshold:
                    # 用左点对称补右点
                    fixed_seq[frame_idx][person_idx][right] = fixed_seq[frame_idx][person_idx][left].copy()
                    fixed_seq[frame_idx][person_idx][right][0] *= -1
    return fixed_seq
