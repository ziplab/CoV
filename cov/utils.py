import logging
import re
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader
from PIL import Image

from cov.config import OpenEQAConfig

log = logging.getLogger(__name__)


def extract_patterns(log_text: str):
    """
    解析日志文本中的多个命令
    返回命令对象列表
    """
    commands = []
    patterns = {
        "movement": re.compile(r"(\w+)-movement\+(\d+)"),
        "rotation": re.compile(r"(\w+)-rotation\+(\d+)"),
        "switch": re.compile(r"switch(?:ing)?(?: to view (\d+))?"),
    }

    # 查找所有movement命令
    for match in patterns["movement"].finditer(log_text):
        commands.append(
            {
                "type": "movement",
                "direction": match.group(1),
                "value": int(match.group(2)),
            }
        )

    # 查找所有rotation命令
    for match in patterns["rotation"].finditer(log_text):
        commands.append(
            {
                "type": "rotation",
                "direction": match.group(1),
                "value": int(match.group(2)),
            }
        )

    # 查找所有switch命令
    for match in patterns["switch"].finditer(log_text):
        commands.append(
            {
                "type": "switch",
                "target": int(match.group(1))
                if match.group(1) is not None
                else None,  # 如果没有目标，默认 None
            }
        )

    return commands


def is_mostly_blank(image_path, threshold=0.9, blank_value=255):
    """
    检测图片是否大部分为空白
    :param image_path: 图片路径
    :param threshold: 空白像素占比阈值，默认
    :param blank_value: 视为空白的像素值(对于灰度图)，RGB则为(255,255,255)
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    # 判断是否为RGB/RGBA图像
    if len(img_array.shape) == 3:
        # 对于彩色图像，检查是否所有通道都是blank_value
        if img_array.shape[2] == 4:  # RGBA图像
            # 先检查RGB通道是否都是blank_value，再检查alpha通道是否完全不透明
            rgb_blank = np.all(img_array[:, :, :3] == blank_value, axis=2)
            alpha_opaque = img_array[:, :, 3] == 255
            blank_pixels = rgb_blank & alpha_opaque
        else:  # RGB图像
            blank_pixels = np.all(img_array == blank_value, axis=2)
    else:  # 灰度图像
        blank_pixels = img_array == blank_value

    blank_ratio = np.mean(blank_pixels)
    return blank_ratio >= threshold


def process_openeqa_path(episode_history: str):
    """
    Route an episode_history like hm3d-v0/000-hm3d-BFRyYbPCCPE to its corresponding path(relative to data/frames).
    """
    # TODO: Unify hm3d and scannet folder structure
    # NOTE: Now we use compressed image folder "comp_color"
    scene_id = episode_history.split("-")[-1]
    if "hm3d" in episode_history:
        glb_path = f"{episode_history}/{scene_id}.glb"
        pose_path = f"{episode_history}/pose/"
        rgb_img_path = f"{episode_history}/comp_color/"
    else:
        glb_path = f"{episode_history}/{scene_id}_vh_clean.glb"
        pose_path = episode_history
        rgb_img_path = episode_history
    return glb_path, pose_path, rgb_img_path


def get_model_name(config: OpenEQAConfig) -> str:
    """从完整模型路径中提取模型名称，并保证路径安全。"""
    return config.model.model_name.split("/")[-1].replace(":", "-")


def build_agent_output_paths(
    config: OpenEQAConfig, agent_name: str, episode_history: str, question_id: str
) -> tuple[Path, Path, Path]:
    """
    构建 agent 执行的输出路径

    Returns:
        base_dir: 基础输出目录
        screen_shot_dir: 截图目录
        html_path: HTML 文件路径
    """
    model_name = get_model_name(config)

    if agent_name == "cov":
        base_dir = (
            config.dataset.output_dir
            / model_name
            / agent_name
            / str(config.min_action_step)
            / episode_history
            / question_id
        )
    else:
        base_dir = (
            config.dataset.output_dir
            / model_name
            / agent_name
            / episode_history
            / question_id
        )
    screen_shot_dir = base_dir / "shots"
    html_path = base_dir / "history.html"

    return base_dir, screen_shot_dir, html_path


def get_results_path(config: OpenEQAConfig) -> Path:
    """获取结果 JSON 文件路径"""
    model_name = get_model_name(config)
    agent_type = config.agent

    if agent_type == "cov":
        return (
            config.dataset.output_dir
            / model_name
            / agent_type
            / str(config.min_action_step)
            / f"{agent_type}-results.json"
        )

    return (
        config.dataset.output_dir
        / model_name
        / agent_type
        / f"{agent_type}-results.json"
    )


def extract_answer(text: str) -> str:
    """
    从大模型输出中提取 answer。

    支持以下格式:
    - "done+answer text" -> "answer text"
    - "done answer text" -> "answer text"
    - "some text done+answer" -> "answer"
    - "answer text" (没有 done 标记时返回原文本)

    Args:
        text: 大模型输出的文本

    Returns:
        提取出的 answer 文本
    """
    if not text:
        return "Text is None"

    text = text.strip()
    text_lower = text.lower()

    done_idx = text_lower.find("done")

    if done_idx != -1:
        after_done = text[done_idx + 4 :].strip()

        if after_done.startswith("+"):
            answer = after_done[1:].strip()
        else:
            answer = after_done

        answer = answer.strip("[]")
        log.info(answer)
        return answer

    answer = text.strip("[]")
    log.info(answer)
    return answer


def load_prompt_template(name: str) -> str:
    """
    Render prompts.
    Params:
        name: str, like baseline_bot.j2 or baseline_bot
    """
    prompts_dir = Path(__file__).parent / "prompts"
    env = Environment(loader=FileSystemLoader(prompts_dir))

    return env.get_template(f"{name.replace('.j2', '')}.j2")
