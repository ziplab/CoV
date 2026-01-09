import logging
import os
from pathlib import Path

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis
from natsort import natsorted
from PIL import Image

from cov.utils import extract_patterns

log = logging.getLogger(__name__)


class Camera:
    def __init__(
        self,
        ply_path: Path,
        pose_path: Path,
        rgb_img_path: Path,
    ):
        ply_path = str(ply_path)  # Because habitat-sim can't read PosixPath object.
        self.view_pose_list = natsorted(
            [
                pose_file
                for pose_file in pose_path.iterdir()
                if pose_file.is_file() and pose_file.suffix == ".txt"
            ],
            key=lambda x: x.stem,
        )

        self.view_img_list = natsorted(
            [
                img_file
                for img_file in rgb_img_path.iterdir()
                if img_file.is_file() and img_file.suffix == ".png"
            ],
            key=lambda x: x.stem,
        )

        if "hm3d" in ply_path:
            sample_rate = 10
        else:
            sample_rate = 60

        self.view_pose_list = self.view_pose_list[::sample_rate]
        self.view_img_list = self.view_img_list[::sample_rate]
        self.cur_view_idx = -1
        self.screen_shot_cnt = 0
        self.on_traj = False

        # 初始化habitat-sim仿真器
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = ply_path
        backend_cfg.enable_physics = False

        # 配置相机传感器
        sensor_cfg = habitat_sim.CameraSensorSpec()
        sensor_cfg.uuid = "color_sensor"
        sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
        sensor_cfg.resolution = [1080, 1920]
        sensor_cfg.position = [0.0, 0.0, 0.0]
        sensor_cfg.hfov = 90

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor_cfg]

        # 配置动作空间
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.4)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.4)
            ),
            "move_left": habitat_sim.agent.ActionSpec(
                "move_left", habitat_sim.agent.ActuationSpec(amount=0.4)
            ),
            "move_right": habitat_sim.agent.ActionSpec(
                "move_right", habitat_sim.agent.ActuationSpec(amount=0.4)
            ),
            "move_up": habitat_sim.agent.ActionSpec(
                "move_up", habitat_sim.agent.ActuationSpec(amount=0.4)
            ),
            "move_down": habitat_sim.agent.ActionSpec(
                "move_down", habitat_sim.agent.ActuationSpec(amount=0.4)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.get_agent(0)

    def _go_to_camera_view(self, pose):
        """
        切换到指定pose矩阵的视角
        """
        self.on_traj = True

        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]

        agent_state = self.agent.get_state()
        agent_state.position = position.astype(np.float32)

        # 将旋转矩阵转换为Magnum四元数，然后转换为habitat-sim可接受的格式
        rotation_quat = mn.Quaternion.from_matrix(mn.Matrix3x3(rotation_matrix))
        # 转换为 [x, y, z, w] 数组格式
        quat_array = np.array(
            [
                rotation_quat.vector.x,
                rotation_quat.vector.y,
                rotation_quat.vector.z,
                rotation_quat.scalar,
            ]
        )
        agent_state.rotation = quat_array

        self.agent.set_state(agent_state)

    def shot_birdeye_view(self, img_dir: str):
        """
        生成场景鸟瞰图并保存
        """
        # 计算场景边界
        scene = self.sim.get_active_scene_graph().get_root_node()
        bb = scene.cumulative_bb
        min_point = np.array([bb.min.x, bb.min.y, bb.min.z])
        max_point = np.array([bb.max.x, bb.max.y, bb.max.z])
        scene_center = (min_point + max_point) / 2
        scene_size = max_point - min_point

        # 自适应计算相机高度
        horizontal_size = max(scene_size[0], scene_size[2])
        pitch_angle = 60
        pitch_rad = np.radians(pitch_angle)
        view_distance = (horizontal_size / 2) / np.tan(pitch_rad)
        best_height = max_point[1] + view_distance * 0.3
        min_height = scene_center[1] + horizontal_size * 0.6
        best_height = max(best_height, min_height)

        # 设置相机位置和旋转
        agent_state = self.agent.get_state()
        agent_state.position = np.array(
            [scene_center[0], 1.5 * best_height, scene_center[2] + 0.6 * scene_size[2]]
        )
        pitch_quat = quat_from_angle_axis(np.radians(-pitch_angle), np.array([1, 0, 0]))
        agent_state.rotation = pitch_quat
        self.agent.set_state(agent_state)

        return self.screen_shot(img_dir, "birdeye_view")

    def switch_back_view(self):
        """
        Restore history view in case that agent moves into blank view, or performs an non-existing action.
        """
        pose = np.loadtxt(self.view_pose_list[self.cur_view_idx])
        self._go_to_camera_view(pose)

    def switch_to_view(self, idx):
        """
        切换到指定索引的视角
        """
        if idx < 0 or idx >= len(self.view_pose_list):
            raise ValueError("Index out of range")
        self.cur_view_idx = idx
        pose = np.loadtxt(self.view_pose_list[self.cur_view_idx])
        self._go_to_camera_view(pose)

    def move_camera(self, direction):
        """
        移动相机位置，使用habitat-sim原生API
        """
        self.on_traj = False

        # 映射按键到habitat-sim原生动作
        action_map = {
            "a": "move_forward",
            "s": "move_backward",
            "d": "move_left",
            "f": "move_right",
            "j": "move_up",
            "k": "move_down",
        }

        action = action_map.get(direction.lower())
        if action:
            self.agent.act(action)

    def rotate_horizontal(self, angle_deg):
        """
        水平旋转相机（绕Y轴），使用habitat-sim原生API
        """
        self.on_traj = False

        # 计算需要执行的旋转次数（每次10度）
        rotation_step = 10.0
        steps = int(abs(angle_deg) / rotation_step)

        # 确定旋转方向
        action = "turn_left" if angle_deg > 0 else "turn_right"

        # 执行旋转
        for _ in range(steps):
            self.agent.act(action)

        # 处理余数（小于10度的部分）
        remainder = abs(angle_deg) % rotation_step
        if remainder > 0:
            # 对于余数部分，使用手动设置状态的方式
            agent_state = self.agent.get_state()
            angle_rad = np.radians(remainder if angle_deg > 0 else -remainder)
            rotation_delta = quat_from_angle_axis(angle_rad, np.array([0, 1, 0]))
            new_rotation = rotation_delta * agent_state.rotation
            agent_state.rotation = new_rotation
            self.agent.set_state(agent_state)

    def screen_shot(self, img_dir: str, img_name: str = None):
        """
        截取当前视角的图像
        返回截图路径
        """
        self.screen_shot_cnt += 1
        if self.on_traj:
            return self.view_img_list[self.cur_view_idx]

        os.makedirs(img_dir, exist_ok=True)
        name = os.path.join(
            img_dir,
            f"{self.screen_shot_cnt}.png" if not img_name else f"{img_name}.png",
        )

        observations = self.sim.get_sensor_observations()
        rgb = observations["color_sensor"]

        img = Image.fromarray(rgb)
        img.save(name)
        return os.path.abspath(name)

    def exec_instruction(self, action: str):
        """
        执行指令字符串
        """
        if "done" in action.lower():
            return

        insts = extract_patterns(action)
        move_insts = [inst for inst in insts if inst["type"] == "movement"]
        rotate_insts = [inst for inst in insts if inst["type"] == "rotation"]
        switch_insts = [inst for inst in insts if inst["type"] == "switch"]
        DIRECTION_TO_KEY = {
            "forward": "a",
            "backward": "s",
            "left": "d",
            "right": "f",
            "upward": "j",
            "downward": "k",
        }
        for inst in move_insts:
            for _ in range(inst["value"]):
                self.move_camera(DIRECTION_TO_KEY[inst["direction"]])
            log.info(f"Moving camera {inst['direction']} by {inst['value']} steps")

        for inst in rotate_insts:
            angle = inst["value"]
            direction = inst["direction"]
            angle = -angle if "right" in direction.lower() else angle
            self.rotate_horizontal(angle)
            log.info(f"Rotating camera by {angle} degrees")

        for inst in switch_insts:
            if inst["target"] is not None:
                self.switch_to_view(inst["target"])
            else:
                self.switch_back_view()

    def __del__(self):
        if hasattr(self, "sim"):
            self.sim.close()
