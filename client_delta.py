"""Aloha Client Example

This script demonstrates how to:
- Get real-time observations from Aloha robot (via ROS)
- Send observations to a policy server (OpenPI policy server) via WebSocket
- Receive actions and execute them on the robot

python openpi-agilex/examples/aloha_real/client.py \
  --host 106.63.100.85 \
  --port 8000 \
  --prompt "pick up the cube and put it on the plate "
After stacking the two bowls together, place them on the first shelf.

# 三个任务：


python openpi-agilex/examples/aloha_real/client_delta.py   --host 127.0.0.1    --port 8000   --prompt "pick the cube and place into the plate."   --ros-master-uri http://agilex:11311

python openpi-agilex/examples/aloha_real/client_delta.py   --host 127.0.0.1   --port 8000   --prompt "After stacking the two bowls together, place them on the first shelf."   --ros-master-uri http://agilex:11311

python openpi-agilex/examples/aloha_real/client_delta.py   --host 127.0.0.1   --port 8000   --prompt "Pick up the broom, pick up the dustpan, sweep the trash from the table into the dustpan, pour the trash from the dustpan into the tray, and place the broom and dustpan on the table."   --ros-master-uri http://agilex:11311


export OPENPI_PROXY_URI='https://nat2-notebook-inspire.sii.edu.cn/ws-6040202d-b785-4b37-98b0-c68d65dd52ce/project-35c38163-ef4e-4165-8c54-cd49fd0f730b/user-e286c6a2-2b9b-4867-851c-5f3911f7acd8/vscode/fbcf0f94-d90a-4e22-b722-eafaa5e4f42b/4e5244ee-f02f-401b-82f6-e153604d5a25/proxy/{{port}}/'
python client_delta.py --host "$OPENPI_PROXY_URI" --port 8000


"""



from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import collections
import cv2
import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from openpi_client import websocket_client_policy as _websocket_client_policy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

import constants

class RosLogger:
    """Route Python-style logging calls to rospy logging for ROS visibility."""

    def debug(self, msg, *args, **kwargs):
        if args:
            rospy.logdebug(msg % args)
        else:
            rospy.logdebug(msg)

    def info(self, msg, *args, **kwargs):
        if args:
            rospy.loginfo(msg % args)
        else:
            rospy.loginfo(msg)

    def warning(self, msg, *args, **kwargs):
        if args:
            rospy.logwarn(msg % args)
        else:
            rospy.logwarn(msg)

    def error(self, msg, *args, **kwargs):
        if args:
            rospy.logerr(msg % args)
        else:
            rospy.logerr(msg)


logger = RosLogger()


class RollingStats:
    """Simple rolling window stats for latency/FPS checks."""

    def __init__(self, window: int = 100):
        self.window = window
        self.values: list[float] = []

    def add(self, v: float) -> None:
        self.values.append(v)
        if len(self.values) > self.window:
            self.values.pop(0)

    def avg(self) -> float:
        return float(sum(self.values) / len(self.values)) if self.values else 0.0

    def p50(self) -> float:
        if not self.values:
            return 0.0
        arr = sorted(self.values)
        mid = len(arr) // 2
        return float((arr[mid] + arr[~mid]) / 2) if len(arr) % 2 == 0 else float(arr[mid])

    def count(self) -> int:
        return len(self.values)


def actions_interpolation(pre_action: np.ndarray, actions: np.ndarray, num_inserts: int = 5) -> np.ndarray:
    """
    线性插值：在 [pre_action] + actions 这 1+N 个关键帧之间，
    每一对相邻关键帧插入 num_inserts 个等间距中间动作，并保留段终点。
    
    **注意**：所有维度（包括夹爪）都进行线性插值，以保证动作平滑。
    如果夹爪值已经转换为关节角度，插值会更平滑自然。
    
    Args:
        pre_action: shape (14,) - 前一个动作（已经是关节角度值）
        actions: shape (N, 14) - 当前动作序列（已经是关节角度值）
        num_inserts: 每对关键帧间插入的中间帧数量
    
    Returns:
        插值后的动作序列 shape = (T, 14)，其中 T = 1 + (num_inserts + 1) * N
    """
    assert pre_action.ndim == 1 and pre_action.shape[0] == 14, "pre_action 必须是 (14,)"
    assert actions.ndim == 2 and actions.shape[1] == 14, "actions 必须是 (N, 14)"

    keyframes = np.vstack([pre_action[None, :], actions])  # (N+1, 14)

    # N = 0 时，直接返回 pre_action
    if keyframes.shape[0] == 1:
        return keyframes.copy()  # shape (1, 14)

    pieces = [keyframes[0][None, :]]  # 先放起点，只放一次
    for i in range(keyframes.shape[0] - 1):
        start = keyframes[i]
        end   = keyframes[i + 1]
        # 生成 [start, ..., end] 共 num_inserts+2 个点；去掉起点，保留中间与终点
        # 所有维度（包括夹爪）都进行线性插值，保证动作平滑
        seg = np.linspace(start, end, num=num_inserts + 2, endpoint=True)[1:]
        pieces.append(seg)  # seg 形状 (num_inserts+1, 14)

    return np.vstack(pieces)  # (T, 14)


@dataclasses.dataclass
class Args:
    # WebSocket Client options
    host: str = "10.176.58.104"
    port: int = 8000
    api_key: Optional[str] = None
    # Control loop frequency (Hz). 建议和数据集 FPS 对齐（你这里是 20Hz）。
    rate: float = 20.0
    # Number of steps to run (0 for infinite)
    steps: int = 0
    
    # ROS Topics (Defaults from inference.py)
    img_front_topic: str = '/camera_f/color/image_raw'
    img_left_topic: str = '/camera_l/color/image_raw'
    img_right_topic: str = '/camera_r/color/image_raw'
    
    img_front_depth_topic: str = '/camera_f/depth/image_raw'
    img_left_depth_topic: str = '/camera_l/depth/image_raw'
    img_right_depth_topic: str = '/camera_r/depth/image_raw'
    
    puppet_arm_left_cmd_topic: str = '/master/joint_left'
    puppet_arm_right_cmd_topic: str = '/master/joint_right'
    puppet_arm_left_topic: str = '/puppet/joint_left'
    puppet_arm_right_topic: str = '/puppet/joint_right'
    
    robot_base_topic: str = '/odom_raw'
    robot_base_cmd_topic: str = '/cmd_vel'
    
    use_robot_base: bool = False
    use_depth_image: bool = False
    publish_rate: int = 30 # Used by RosOperator internally if needed
    arm_steps_length: List[float] = dataclasses.field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2])

    prompt: str = "pick up the cube"
    ros_master_uri: Optional[str] = None
    ros_hostname: Optional[str] = None
    ros_ip: Optional[str] = None
    
    # 插值和平滑参数
    # 你的 action 是“绝对关节角目标”，一般不需要在 client 里做二次平滑/插值，
    # 直接按 20Hz 发送每一步目标更稳定、更符合训练分布。
    use_actions_interpolation: bool = False
    num_interpolation_steps: int = 25

    # 如果 True，会在每个 action 内部循环多次逼近目标（容易把节拍搞乱、导致 timeout）
    use_continuous_smooth_publisher: bool = False
    
    # 夹爪处理参数（与训练 config 对齐）
    # 你的 `NormalizeGripperDims` 在训练/推理都会作用在 state/actions 上：
    # - 模型输出 gripper 维度是归一化 [0,1]
    # - client 需要用同一套 (raw_min/raw_max/flip) 反归一化回“原始量纲”，再发到机器人控制 topic
    binarize_gripper: bool = False
    gripper_threshold: float = 0.5
    left_gripper_raw_min: float = 0.0
    left_gripper_raw_max: float = 4.5
    left_gripper_flip: bool = True
    right_gripper_raw_min: float = -0.0505457
    right_gripper_raw_max: float = 4.5
    right_gripper_flip: bool = True

    # 复位位置 (左臂7维 + 右臂7维)
    reset_joints_open: List[float] = dataclasses.field(default_factory=lambda: [
        -0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, 
        -0.00286102294921875, 0.00095367431640625, 3.557830810546875,
        -0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, 
        -0.00476837158203125, -0.00209808349609375, 3.557830810546875
    ])
    reset_joints_closed: List[float] = dataclasses.field(default_factory=lambda: [
        -0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, 
        -0.00286102294921875, 0.00095367431640625, -0.3393220901489258,
        -0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, 
        -0.00286102294921875, 0.00095367431640625, -0.3397035598754883
    ])
    
    # 控制执行动作数量的参数（与当前 task1 model action_horizon=16 对齐）
    action_exec_horizon: int = 16
    # 动作执行参数
    max_action_chunk_size: int = 50  # 每次从模型接收的动作序列中，最多执行多少个动作
    # 等待动作执行完成的超时时间（秒），以及位置容差（弧度/米等）
    action_completion_timeout: float = 4.0
    action_completion_pos_tol: float = 0.02

    # ===== Action semantics / debugging =====
    # If True, treat joint dims (0-5 and 7-12) from the policy as DELTAS and add them to current observed joints
    # before publishing. This is a quick way to validate whether your served policy is outputting deltas vs absolutes.
    apply_joint_deltas_to_current: bool = False

    # Print current joints vs commanded joints at step 0 and last step of each executed chunk.
    log_commanded_targets: bool = True

    # Avoid replaying a repeated chunk from frame zero when the robot is
    # already near a later frame in that chunk.
    align_action_chunk_to_current: bool = True
    chunk_alignment_min_improvement: float = 0.05

    # Real-time chunking (RTC) / temporal action ensembling.
    # Re-plan after a short prefix and blend all chunks that predict the same
    # absolute control timestep.  This smooths boundaries between chunks while
    # keeping the controller responsive to new observations.
    rtc_enabled: bool = True
    rtc_execute_horizon: int = 8
    rtc_decay: float = 0.25
    rtc_max_chunks: int = 8


class TemporalActionEnsembler:
    """Fuse overlapping action chunks on a shared control-step timeline."""

    def __init__(self, decay: float = 0.25, max_chunks: int = 8):
        if decay < 0:
            raise ValueError("rtc_decay must be non-negative")
        if max_chunks < 1:
            raise ValueError("rtc_max_chunks must be at least 1")
        self.decay = float(decay)
        self.max_chunks = int(max_chunks)
        self._chunks = deque(maxlen=self.max_chunks)

    def add_chunk(self, start_step: int, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 14:
            raise ValueError(f"RTC expects a non-empty (T, 14) chunk, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("RTC received NaN or Inf in action chunk")
        self._chunks.append((int(start_step), actions.copy()))

    def ensemble(self, start_step: int, count: int) -> np.ndarray:
        if count < 1:
            raise ValueError("RTC ensemble count must be positive")

        fused = []
        for absolute_step in range(int(start_step), int(start_step) + int(count)):
            candidates = []
            weights = []
            for chunk_start, chunk in self._chunks:
                offset = absolute_step - chunk_start
                if 0 <= offset < len(chunk):
                    # Prefer newer predictions, while retaining older chunks to
                    # smooth the boundary between consecutive replans.
                    age = max(0, int(start_step) - chunk_start)
                    candidates.append(chunk[offset])
                    weights.append(np.exp(-self.decay * age))

            if not candidates:
                break
            fused.append(np.average(np.stack(candidates), axis=0, weights=np.asarray(weights)))

        # Chunks are always added before ensemble(), so an empty result means a
        # timeline/shape bug and must not silently command the robot.
        if not fused:
            raise RuntimeError("RTC found no action prediction for the current control step")

        # Drop chunks that cannot contribute to this or any future timestep.
        next_step = int(start_step) + len(fused)
        self._chunks = deque(
            ((s, a) for s, a in self._chunks if s + len(a) > next_step),
            maxlen=self.max_chunks,
        )
        return np.asarray(fused, dtype=np.float32)


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.last_log_time = 0.0  # throttle verbose diagnostics
        logger.info("init()")
        self.init()
        logger.info("init_ros()")
        self.init_ros()

    def _log_queue_status(self, reason: str) -> None:
        """Log queue lengths and latest timestamps when synchronization fails."""
        now = time.time()
        # Throttle to avoid spamming logs
        if now - self.last_log_time < 1.0:
            return
        self.last_log_time = now

        def _ts(q):
            try:
                return q[-1].header.stamp.to_sec()
            except Exception:
                return None

        status = {
            "img_left": (len(self.img_left_deque), _ts(self.img_left_deque)),
            "img_right": (len(self.img_right_deque), _ts(self.img_right_deque)),
            "img_front": (len(self.img_front_deque), _ts(self.img_front_deque)),
            "img_left_depth": (len(self.img_left_depth_deque), _ts(self.img_left_depth_deque)),
            "img_right_depth": (len(self.img_right_depth_deque), _ts(self.img_right_depth_deque)),
            "img_front_depth": (len(self.img_front_depth_deque), _ts(self.img_front_depth_deque)),
            "puppet_left": (len(self.puppet_arm_left_deque), _ts(self.puppet_arm_left_deque)),
            "puppet_right": (len(self.puppet_arm_right_deque), _ts(self.puppet_arm_right_deque)),
            "robot_base": (len(self.robot_base_deque), _ts(self.robot_base_deque)),
        }

        logger.warning("get_frame unavailable (%s). Queue status: %s", reason, status)

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def _get_latest_joint_names(self, side: str) -> Optional[List[str]]:
        """Best-effort: reuse the same joint name order as the robot publishes, to avoid controllers ignoring commands."""
        try:
            if side == "left" and len(self.puppet_arm_left_deque) > 0:
                names = list(self.puppet_arm_left_deque[-1].name)
            elif side == "right" and len(self.puppet_arm_right_deque) > 0:
                names = list(self.puppet_arm_right_deque[-1].name)
            else:
                return None
            return names if len(names) == 7 else None
        except Exception:
            return None

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = self._get_latest_joint_names("left") or ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.name = self._get_latest_joint_names("right") or joint_state_msg.name
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def puppet_arm_publish_continuous_smooth(self, left, right):
        """让机械臂按步长限制平滑移动到目标位置，参考inference.py中的puppet_arm_publish_continuous方法"""
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        
        # 获取当前位置
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        
        # 计算移动方向
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            
            # 限制每步的移动距离，参考inference.py中的逻辑
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            rospy.logdebug(f"puppet_arm_publish_continuous_smooth: step {step}")
            rate.sleep()

    def puppet_arm_publish_continuous(self, left, right):
        """让机械臂按步长限制平滑移动到目标位置"""
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        
        # 获取当前位置
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        
        # 计算移动方向
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            
            # 限制每步的移动距离
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            rospy.logdebug(f"puppet_arm_publish_continuous: step {step}")
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right, num_steps=100):
        """使用线性插值让机械臂平滑移动到目标位置"""
        rate = rospy.Rate(200)
        left_arm = None
        right_arm = None
        
        # 获取当前位置
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        
        # 生成线性插值轨迹
        traj_left_list = np.linspace(left_arm, left, num_steps)
        traj_right_list = np.linspace(right_arm, right, num_steps)
        
        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            # 保持夹爪状态
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            self._log_queue_status("missing images")
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            self._log_queue_status("left image too old")
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            self._log_queue_status("right image too old")
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            self._log_queue_status("front image too old")
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            self._log_queue_status("puppet left too old")
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            self._log_queue_status("puppet right too old")
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            self._log_queue_status("left depth too old")
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            self._log_queue_status("right depth too old")
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            self._log_queue_status("front depth too old")
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            self._log_queue_status("robot base too old")
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        try:
            img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'bgr8')
        except Exception as e:
            logger.error(f"Error converting left image: {e}")
            img_left = np.zeros((480, 640, 3), dtype=np.uint8)

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        try:
            img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'bgr8')
        except Exception as e:
            logger.error(f"Error converting right image: {e}")
            img_right = np.zeros((480, 640, 3), dtype=np.uint8)

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        try:
            img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'bgr8')
        except Exception as e:
            logger.error(f"Error converting front image: {e}")
            img_front = np.zeros((480, 640, 3), dtype=np.uint8)

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        logger.info("init_node")
        rospy.init_node('aloha_client', anonymous=True,disable_signals=True)
        rospy.loginfo("Subscriber")
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        rospy.loginfo("Publisher")
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


class AlohaController:
    def __init__(self, args: Args):
        logger.info("AlohaController.__init__: Starting initialization")
        self.args = args
        logger.info("AlohaController.__init__: Creating RosOperator")
        self.ros_operator = RosOperator(args)
        rospy.loginfo("AlohaController.__init__: RosOperator created")
        # Wait for ROS to connect and receive data
        rospy.loginfo("Waiting for ROS data (2 seconds)...")
        time.sleep(2.0)
        rospy.loginfo("ROS data wait complete.")
        
        # 初始化动作队列和前一动作
        self.action_queue = collections.deque()
        self.last_action = None
    
    def reset_to_default_pose(self, use_linear=False):
        """复位机械臂到默认姿态"""
        rospy.loginfo("Resetting arms to open gripper pose...")
        left_open = self.args.reset_joints_open[:7]
        right_open = self.args.reset_joints_open[7:14]
        
        if use_linear:
            self.ros_operator.puppet_arm_publish_linear(left_open, right_open, num_steps=100)
        else:
            # 使用新的平滑移动函数
            self.ros_operator.puppet_arm_publish_continuous_smooth(left_open, right_open)
        
        time.sleep(0.5)
        
        rospy.loginfo("Resetting arms to closed gripper pose...")
        left_closed = self.args.reset_joints_closed[:7]
        right_closed = self.args.reset_joints_closed[7:14]
        
        if use_linear:
            self.ros_operator.puppet_arm_publish_linear(left_closed, right_closed, num_steps=100)
        else:
            # 使用新的平滑移动函数
            self.ros_operator.puppet_arm_publish_continuous_smooth(left_closed, right_closed)
        
        # 更新 last_action
        self.last_action = np.array(left_closed + right_closed)
        rospy.loginfo("Reset complete.") 

    def read_observation(self) -> Optional[Dict[str, Any]]:
        rospy.loginfo("read_observation: Calling ros_operator.get_frame()...")
        ret = self.ros_operator.get_frame()
        if not ret:
            rospy.loginfo("read_observation: get_frame() returned False (no data yet)")
            return None
        rospy.loginfo("read_observation: Got frame data successfully")

        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = ret

        # Convert BGR to RGB if necessary (cv_bridge usually returns BGR for 'bgr8' or 'passthrough' if input is bgr)
        # Assuming cameras are RGB or BGR. Standard ROS is often BGR.
        # inference.py uses 'passthrough'. If raw images are bgr, we might need conversion.
        # Let's assume they are BGR and convert to RGB for the model.
        
        # Debug shapes before conversion
        logger.info(f"Raw shapes (HWC) - Front: {img_front.shape}, Left: {img_left.shape}, Right: {img_right.shape}")

        # Validate and convert images from BGR HWC to RGB CHW
        def process_image(img, name):
            # Validate size
            if img.size < 10000:
                logger.error(f"Image {name} is too small: {img.shape}. Replacing with black image.")
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Ensure RGB HWC format
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to CHW format (from HWC) for policy
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            
            # Ensure uint8
            img = img.astype(np.uint8)
            
            rospy.loginfo(f"Processed {name} shape (CHW): {img.shape}, dtype: {img.dtype}")
            return img

        img_front = process_image(img_front, "Front")
        img_left = process_image(img_left, "Left")
        img_right = process_image(img_right, "Right")

        # Resize if needed? inference.py doesn't seem to resize in get_frame, but maybe in model wrapper.
        # We'll send raw images and let the server handle transforms if possible, 
        # or resize to 224x224 if that's what the policy expects.
        # xarm_client resizes to 224x224. Let's do that to be safe/standard.
        # img_front = cv2.resize(img_front, (224, 224))
        # img_left = cv2.resize(img_left, (224, 224))
        # img_right = cv2.resize(img_right, (224, 224))
        # Actually, let's NOT resize here unless we know the policy requires it. 
        # The server usually has transforms.

        # IMPORTANT: 对齐你的训练数据处理逻辑。
        # 你在训练时用过 `NormalizeGripperDims` 把 raw gripper -> [0,1]（并可能 flip）。
        # 但该 transform 你放在 `repack_transforms` 里，而 `repack_transforms` 在推理 server 侧不会被应用。
        # 所以这里需要在 client 侧对 state 的 gripper 做同样的归一化，再发给 server。
        qpos_left_raw = np.array(puppet_arm_left.position)
        qpos_right_raw = np.array(puppet_arm_right.position)
        left_arm = qpos_left_raw[:6]
        right_arm = qpos_right_raw[:6]
        left_grip_raw = float(qpos_left_raw[6]) if qpos_left_raw.size > 6 else 0.0
        right_grip_raw = float(qpos_right_raw[6]) if qpos_right_raw.size > 6 else 0.0

        def _norm_gripper(raw: float, raw_min: float, raw_max: float, flip: bool) -> float:
            # match NormalizeGripperDims(new_min=0,new_max=1)
            denom = (raw_max - raw_min) if (raw_max - raw_min) != 0 else 1.0
            x = (raw - raw_min) / denom
            x = float(np.clip(x, 0.0, 1.0))
            if flip:
                x = 1.0 - x
            return x

        left_grip = _norm_gripper(
            left_grip_raw, self.args.left_gripper_raw_min, self.args.left_gripper_raw_max, self.args.left_gripper_flip
        )
        right_grip = _norm_gripper(
            right_grip_raw, self.args.right_gripper_raw_min, self.args.right_gripper_raw_max, self.args.right_gripper_flip
        )
        qpos = np.concatenate([left_arm, [left_grip], right_arm, [right_grip]])

        # Construct qvel (shape 14 when possible)
        qvel = np.zeros_like(qpos)
        if puppet_arm_left.velocity and puppet_arm_right.velocity:
            v_left = np.array(puppet_arm_left.velocity)
            v_right = np.array(puppet_arm_right.velocity)
            if v_left.size > 5 and v_right.size > 5:
                left_arm_v = v_left[:6]
                right_arm_v = v_right[:6]
                if v_left.size > 7:
                    left_grip_v = float(constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(float(v_left[7])))
                else:
                    left_grip_v = 0.0
                if v_right.size > 7:
                    right_grip_v = float(constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(float(v_right[7])))
                else:
                    right_grip_v = 0.0
                qvel = np.concatenate([left_arm_v, [left_grip_v], right_arm_v, [right_grip_v]])

        # Effort is not required by OpenPI policy; keep shape consistent.
        effort = np.zeros((14,), dtype=np.float32)

        obs = {
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "state": qpos,
            "qpos": qpos,
            "qvel": qvel,
            "effort": effort,
            "prompt": self.args.prompt,
        }
        
        # Also add flat keys if policy expects them (like xarm example)
        # obs["head_image"] = img_front
        # obs["wrist_image"] = img_left # or right?
        
        return obs

    def apply_action(
        self,
        action: np.ndarray,
        use_interpolation: bool = None,
        allow_chunk_alignment: bool = True,
    ) -> int:
        """执行动作，支持插值平滑"""
        if use_interpolation is None:
            use_interpolation = self.args.use_actions_interpolation
        
        logger.info(f"apply_action: Received action with shape {action.shape}")
        
        if action.ndim == 1:
            action = action[None, :] # Make it (1, D)
        
        # Ensure action is writable
        action = action.copy()

        # Compare only arm joints: grippers are normalized in `action`, while
        # the ROS JointState contains raw gripper positions.
        if allow_chunk_alignment and self.args.align_action_chunk_to_current and len(action) > 1:
            try:
                cur_left = np.asarray(self.ros_operator.puppet_arm_left_deque[-1].position, dtype=np.float64)[:6]
                cur_right = np.asarray(self.ros_operator.puppet_arm_right_deque[-1].position, dtype=np.float64)[:6]
                current_joints = np.concatenate([cur_left, cur_right])
                chunk_joints = np.concatenate([action[:, :6], action[:, 7:13]], axis=1)
                errors = np.mean(np.abs(chunk_joints - current_joints[None, :]), axis=1)
                start_idx = int(np.argmin(errors))
                improvement = float(errors[0] - errors[start_idx])
                if start_idx > 0 and improvement >= self.args.chunk_alignment_min_improvement:
                    logger.warning(
                        "Chunk continuity: skipping repeated prefix [0:%d]; joint MAE %.4f -> %.4f rad",
                        start_idx, float(errors[0]), float(errors[start_idx]),
                    )
                    action = action[start_idx:]
            except (IndexError, ValueError, TypeError) as exc:
                logger.warning("Chunk continuity check skipped: %s", exc)
        
        # ========== DEBUG: 记录原始gripper值 ==========
        left_gripper_raw = action[:, 6].copy()
        right_gripper_raw = action[:, 13].copy()
        
        logger.info("="*60)
        logger.info("DEBUG: Gripper Value Analysis")
        logger.info("="*60)
        logger.info(f"Raw gripper values (BEFORE any conversion):")
        logger.info(f"  Left gripper:  min={left_gripper_raw.min():.6f}, max={left_gripper_raw.max():.6f}, "
                   f"mean={left_gripper_raw.mean():.6f}, first={left_gripper_raw[0]:.6f}")
        logger.info(f"  Right gripper: min={right_gripper_raw.min():.6f}, max={right_gripper_raw.max():.6f}, "
                   f"mean={right_gripper_raw.mean():.6f}, first={right_gripper_raw[0]:.6f}")
        
        # 定义有效范围
        NORM_MIN = 0.0
        NORM_MAX = 1.0
        
        logger.info("Expected ranges:")
        logger.info(f"  Gripper normalized range (model output): [{NORM_MIN:.1f}, {NORM_MAX:.1f}]")
        logger.info(
            f"  Will unnormalize to RAW using config params: "
            f"left[{self.args.left_gripper_raw_min},{self.args.left_gripper_raw_max}], flip={self.args.left_gripper_flip}; "
            f"right[{self.args.right_gripper_raw_min},{self.args.right_gripper_raw_max}], flip={self.args.right_gripper_flip}"
        )

        # Model outputs gripper dims in normalized [0,1]. Convert back to RAW values that match the training dataset.
        def _unnorm_gripper(x: np.ndarray, raw_min: float, raw_max: float, flip: bool) -> np.ndarray:
            x = np.clip(x, 0.0, 1.0)
            if self.args.binarize_gripper:
                x = np.where(x >= self.args.gripper_threshold, 1.0, 0.0)
            if flip:
                x = 1.0 - x
            return raw_min + x * (raw_max - raw_min)

        action[:, 6] = _unnorm_gripper(action[:, 6], self.args.left_gripper_raw_min, self.args.left_gripper_raw_max, self.args.left_gripper_flip)
        action[:, 13] = _unnorm_gripper(action[:, 13], self.args.right_gripper_raw_min, self.args.right_gripper_raw_max, self.args.right_gripper_flip)
        
        logger.info("="*60)

        # 如果启用插值且有前一动作
        if use_interpolation and self.last_action is not None:
            logger.info(f"Applying action interpolation with {self.args.num_interpolation_steps} steps")
            interpolated_actions = actions_interpolation(
                self.last_action, 
                action, 
                num_inserts=self.args.num_interpolation_steps
            )
            logger.info(f"Interpolated actions shape: {interpolated_actions.shape}")
            actions_to_execute = interpolated_actions
        else:
            actions_to_execute = action
        
        # 只执行前args.action_exec_horizon个动作
        actions_to_execute = actions_to_execute[:self.args.action_exec_horizon]
        logger.info(f"Trimmed to first {self.args.action_exec_horizon} actions. Executing {len(actions_to_execute)} action steps...")
        
        # 使用固定频率发布动作（建议与数据集 20Hz 对齐）
        action_rate = rospy.Rate(float(self.args.rate))
        exec_start = time.time()
        
        for i, act in enumerate(actions_to_execute):
            left_action = act[:7]
            right_action = act[7:14]

            # Optional: interpret joint dims as deltas.
            if getattr(self.args, "apply_joint_deltas_to_current", False):
                try:
                    if len(self.ros_operator.puppet_arm_left_deque) > 0:
                        cur_left = np.array(self.ros_operator.puppet_arm_left_deque[-1].position)[:7]
                        left_action[:6] = left_action[:6] + cur_left[:6]
                    if len(self.ros_operator.puppet_arm_right_deque) > 0:
                        cur_right = np.array(self.ros_operator.puppet_arm_right_deque[-1].position)[:7]
                        right_action[:6] = right_action[:6] + cur_right[:6]
                except Exception as e:
                    logger.warning(f"apply_joint_deltas_to_current failed: {e}")
            
            if i == 0 or i == len(actions_to_execute) - 1 or i % 10 == 0:
                # 显示位置(前3维)和夹爪(第6维)
                logger.info(f"  Step {i}/{len(actions_to_execute)}: "
                           f"Left_pos={left_action[:3].tolist()} Left_gripper={left_action[6]:.4f} | "
                           f"Right_pos={right_action[:3].tolist()} Right_gripper={right_action[6]:.4f}")

            if getattr(self.args, "log_commanded_targets", False) and (i == 0 or i == len(actions_to_execute) - 1):
                try:
                    cur_left = np.array(self.ros_operator.puppet_arm_left_deque[-1].position)[:7] if len(self.ros_operator.puppet_arm_left_deque) else None
                    cur_right = np.array(self.ros_operator.puppet_arm_right_deque[-1].position)[:7] if len(self.ros_operator.puppet_arm_right_deque) else None
                    logger.info(
                        f"    current_left[:7]={cur_left.tolist() if cur_left is not None else None} -> cmd_left[:7]={left_action.tolist()}"
                    )
                    logger.info(
                        f"    current_right[:7]={cur_right.tolist() if cur_right is not None else None} -> cmd_right[:7]={right_action.tolist()}"
                    )
                except Exception:
                    pass
            
            # 直接发布每一步目标（绝对关节角），让控制器跟踪即可。
            # 注意：puppet_arm_publish_continuous_smooth 会在每步内部循环多次，容易让节拍错乱。
            if getattr(self.args, "use_continuous_smooth_publisher", False):
                self.ros_operator.puppet_arm_publish_continuous_smooth(left_action, right_action)
            else:
                self.ros_operator.puppet_arm_publish(left_action, right_action)
            action_rate.sleep()
        
        exec_elapsed = time.time() - exec_start
        if exec_elapsed > 0:
            hz = len(actions_to_execute) / exec_elapsed
            logger.info(
                "Action block executed in %.3fs (%.2f Hz effective publish rate; target %.2f Hz)",
                exec_elapsed,
                hz,
                float(self.args.rate),
            )

        # 更新 last_action 为最后执行的动作
        self.last_action = actions_to_execute[-1]
        logger.info(f"Completed executing {len(actions_to_execute)} actions")
        return len(actions_to_execute)

    def wait_for_action_completion(self, target_action: Optional[np.ndarray], timeout: Optional[float] = None, pos_tol: Optional[float] = None) -> bool:
        """等待机器人执行到目标动作位置（基于末端关节位置），直到超时。

        Args:
            target_action: 形状 (14,) 或 (1,14) 的目标动作（左右臂7维+7维）。
            timeout: 最大等待时间（秒），None 则使用 `self.args.action_completion_timeout`。
            pos_tol: 位置容差（每个关节的绝对误差上限），None 则使用 `self.args.action_completion_pos_tol`。

        Returns:
            True 如果在超时时间内达到目标，否则 False。
        """
        if timeout is None:
            timeout = getattr(self.args, 'action_completion_timeout', 8.0)
        if pos_tol is None:
            pos_tol = getattr(self.args, 'action_completion_pos_tol', 0.02)

        if target_action is None:
            logger.warning("wait_for_action_completion: target_action is None, nothing to wait for.")
            return True

        # 规范化目标动作为一维 (14,)
        tgt = np.array(target_action).reshape(-1)
        if tgt.size >= 14:
            tgt = tgt[:14]
        else:
            logger.warning("wait_for_action_completion: target_action has unexpected size %s", tgt.size)
            return True

        tgt_left = tgt[:7]
        tgt_right = tgt[7:14]

        start_t = time.time()
        rate = rospy.Rate(20)  # poll 20 Hz
        rospy.loginfo(f"Waiting for action completion: timeout={timeout}s, pos_tol={pos_tol}")

        while not rospy.is_shutdown():
            if time.time() - start_t > timeout:
                rospy.logwarn("wait_for_action_completion: timeout after %.2fs", timeout)
                return False

            # 获取最新的关节状态
            if len(self.ros_operator.puppet_arm_left_deque) == 0 or len(self.ros_operator.puppet_arm_right_deque) == 0:
                rate.sleep()
                continue

            try:
                cur_left = np.array(self.ros_operator.puppet_arm_left_deque[-1].position)
                cur_right = np.array(self.ros_operator.puppet_arm_right_deque[-1].position)
            except Exception as e:
                rospy.logwarn(f"Error reading current puppet arm positions: {e}")
                rate.sleep()
                continue

            diff_left = np.max(np.abs(cur_left - tgt_left))
            diff_right = np.max(np.abs(cur_right - tgt_right))
            max_diff = max(diff_left, diff_right)

            if max_diff <= pos_tol:
                rospy.loginfo(f"Action reached target within tol: max_diff={max_diff:.4f}")
                return True

            rate.sleep()

        return False


def run_client(args: Args) -> None:
    logger.info("="*60)
    logger.info("Starting Aloha Client...")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info("="*60)

    inference_stats = RollingStats(window=200)
    step_stats = RollingStats(window=200)
    apply_stats = RollingStats(window=200)
    loop_wall_start = time.time()

    # Override ROS environment variables if provided via CLI
    if args.ros_master_uri:
        os.environ['ROS_MASTER_URI'] = args.ros_master_uri
        logger.info(f"Overriding ROS_MASTER_URI -> {args.ros_master_uri}")
    if args.ros_hostname:
        os.environ['ROS_HOSTNAME'] = args.ros_hostname
        logger.info(f"Overriding ROS_HOSTNAME -> {args.ros_hostname}")
    if args.ros_ip:
        os.environ['ROS_IP'] = args.ros_ip
        logger.info(f"Overriding ROS_IP -> {args.ros_ip}")
    
    # Check ROS environment
    ros_master = os.environ.get('ROS_MASTER_URI', 'Not set')
    logger.info(f"ROS_MASTER_URI: {ros_master}")
    
    # Test ROS master connectivity first
    logger.info("Testing ROS master connectivity...")
    try:
        import subprocess
        result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=3)
        if result.returncode != 0:
            logger.error("Cannot connect to ROS master!")
            logger.error(f"rostopic list failed with: {result.stderr}")
            logger.error(f"ROS_MASTER_URI is: {ros_master}")
            logger.error("Please ensure roscore is running: roscore")
            return
        logger.info("ROS master is accessible.")
    except subprocess.TimeoutExpired:
        logger.error("ROS master connection timed out!")
        logger.error("This usually means roscore is not running or network issues.")
        return
    except Exception as e:
        logger.error(f"Error testing ROS master: {e}")
        return

    # Clean up any lingering nodes before initializing
    logger.info("Cleaning up any existing aloha_client nodes before initialization...")
    try:
        import subprocess
        result = subprocess.run(['rosnode', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'aloha_client' in line:
                    node_name = line.strip()
                    if not node_name:
                        continue
                    logger.warning(f"Found existing node: {node_name}, killing it...")
                    subprocess.run(['rosnode', 'kill', node_name], timeout=5)
            time.sleep(1.0)
    except Exception as e:
        logger.warning(f"Could not clean existing nodes: {e}")

    logger.info("Creating AlohaController...")
    controller = AlohaController(args)
    rospy.loginfo("AlohaController created.")
    
    # 询问是否需要复位
    rospy.loginfo("Do you want to reset arms to default pose? (Press Enter to skip, 'y' to reset)")
    try:
        # 使用非阻塞方式等待输入，超时 5 秒
        import select
        i, o, e = select.select([sys.stdin], [], [], 5.0)
        if i:
            response = sys.stdin.readline().strip().lower()
            if response == 'y':
                controller.reset_to_default_pose(use_linear=False)
        else:
            rospy.loginfo("No input received, skipping reset.")
    except Exception as e:
        rospy.logwarn(f"Could not read input for reset: {e}, skipping reset.")
    
    # Create websocket client
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
    rospy.loginfo(f"Connected to policy server at {args.host}:{args.port}")
    rospy.loginfo(f"Server metadata: {policy.get_server_metadata()}")

    # Warmup
    rospy.loginfo("Starting warmup (2 iterations)...")
    for i in range(2):
        rospy.loginfo(f"Warmup iteration {i+1}/2...")
        obs = controller.read_observation()
        if obs:
            rospy.loginfo(f"Warmup {i+1}: Got observation, calling infer...")
            policy.infer(obs)
            rospy.loginfo(f"Warmup {i+1}: Infer complete.")
        else:
            logger.warning(f"Warmup {i+1}: Failed to get observation")
    rospy.loginfo("Warmup complete.")

    if args.rtc_execute_horizon < 1:
        raise ValueError("--rtc-execute-horizon must be at least 1")
    if args.rtc_execute_horizon > args.action_exec_horizon:
        raise ValueError(
            "--rtc-execute-horizon cannot exceed --action-exec-horizon "
            f"({args.rtc_execute_horizon} > {args.action_exec_horizon})"
        )
    rtc = (
        TemporalActionEnsembler(decay=args.rtc_decay, max_chunks=args.rtc_max_chunks)
        if args.rtc_enabled
        else None
    )
    action_step = 0
    rospy.loginfo(
        "RTC temporal smoothing: enabled=%s, execute_horizon=%d, decay=%.3f, max_chunks=%d",
        args.rtc_enabled,
        args.rtc_execute_horizon,
        args.rtc_decay,
        args.rtc_max_chunks,
    )

    rate = rospy.Rate(args.rate)
    step = 0
    rospy.loginfo(f"Starting main control loop at {args.rate} Hz...")
    
    try:
        while not rospy.is_shutdown():
            step_start = time.time()
            if args.steps > 0 and step >= args.steps:
                break
                
            rospy.loginfo(f"Step {step}: Reading observation...")
            obs = controller.read_observation()
            if obs is None:
                rospy.loginfo(f"Step {step}: No observation available, sleeping...")
                rate.sleep()
                continue
                
            # Inference
            t0 = time.time()
            rospy.loginfo(f"Step {step}: Calling policy.infer()...")
            result = policy.infer(obs)
            # result is usually a dict with 'actions'
            actions = np.asarray(result['actions'], dtype=np.float32)
            inference_time = time.time() - t0

            if rtc is not None:
                rtc.add_chunk(action_step, actions)
                actions = rtc.ensemble(action_step, args.rtc_execute_horizon)
                rospy.loginfo(
                    "RTC fused action steps [%d, %d) from overlapping chunks",
                    action_step,
                    action_step + len(actions),
                )
            
            # Print detailed action information
            rospy.loginfo(f"\n{'='*60}")
            rospy.loginfo(f"Step {step}: Inference completed")
            rospy.loginfo(f"  Inference time: {inference_time:.3f}s")
            rospy.loginfo(f"  Action shape: {actions.shape}")
            rospy.loginfo(f"  Action dtype: {actions.dtype}")
            rospy.loginfo(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")
            
            # Print first action in detail
            if len(actions) > 0:
                first_action = actions[0]
                rospy.loginfo(f"  First action (14 dims):")
                rospy.loginfo(f"    Left arm  (0-6): {first_action[:7]}")
                rospy.loginfo(f"    Right arm (7-13): {first_action[7:14]}")
                # 特别检查夹爪值范围（应该是关节角度值）
                rospy.loginfo(f"    Gripper values - Left: {first_action[6]:.4f}, Right: {first_action[13]:.4f}")
                rospy.loginfo(f"    Expected range: OPEN={constants.PUPPET_GRIPPER_JOINT_OPEN:.4f}, CLOSE={constants.PUPPET_GRIPPER_JOINT_CLOSE:.4f}")
                if len(first_action) > 14:
                    rospy.loginfo(f"    Extra dims: {first_action[14:]}")
            
            rospy.loginfo(f"  Total actions in chunk: {len(actions)}")
            rospy.loginfo(f"{'='*60}\n")
            
            # Apply action
            rospy.loginfo(f"Step {step}: Applying actions...")
            apply_start = time.time()
            executed_actions = controller.apply_action(
                actions,
                allow_chunk_alignment=(rtc is None),
            )
            action_step += executed_actions
            apply_duration = time.time() - apply_start
            apply_stats.add(apply_duration)

            # 等待动作执行完成再进行下一次推理请求
            target = controller.last_action
            ok = controller.wait_for_action_completion(target, timeout=args.action_completion_timeout, pos_tol=args.action_completion_pos_tol)
            if not ok:
                rospy.logwarn("Action did not reach target within timeout; continuing to next inference anyway.")

            step_duration = time.time() - step_start
            step_stats.add(step_duration)
            inference_stats.add(inference_time)

            if step_stats.count() % 10 == 0:
                avg_step = step_stats.avg()
                avg_infer = inference_stats.avg()
                avg_apply = apply_stats.avg()
                logger.info(
                    "Perf (rolling %d): step %.3fs (~%.2f Hz), infer %.3fs (~%.2f Hz), apply %.3fs",
                    step_stats.count(),
                    avg_step,
                    (1.0 / avg_step) if avg_step > 0 else 0.0,
                    avg_infer,
                    (1.0 / avg_infer) if avg_infer > 0 else 0.0,
                    avg_apply,
                )

            step += 1
            # rate.sleep() # apply_action might already sleep if executing chunks. 
            # If action is single step, we need to sleep.
            # If action is chunk, apply_action sleeps between steps.
            # But we also need to account for inference time.
            # For simplicity, if we execute a chunk, we assume it takes time.
            # If we want closed loop control (MPC style), we might only execute first action.
            # But here we execute what we get.
            
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        wall_elapsed = time.time() - loop_wall_start
        total_steps = step_stats.count()
        if total_steps > 0:
            avg_step = step_stats.avg()
            avg_infer = inference_stats.avg()
            avg_apply = apply_stats.avg()
            logger.info(
                "Final perf: steps=%d, wall=%.2fs, step_avg=%.3fs (~%.2f Hz), infer_avg=%.3fs (~%.2f Hz), apply_avg=%.3fs",
                total_steps,
                wall_elapsed,
                avg_step,
                (1.0 / avg_step) if avg_step > 0 else 0.0,
                avg_infer,
                (1.0 / avg_infer) if avg_infer > 0 else 0.0,
                avg_apply,
            )
        else:
            logger.info("Final perf: no steps executed; wall=%.2fs", wall_elapsed)


def main():
    logging.basicConfig(
        level=logging.DEBUG,  # ← 改成 DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    args = tyro.cli(Args)
    run_client(args)


if __name__ == '__main__':
    main()
