import logging
import os
import re

from cov.bots import BaselineBot, Chatbot, ViewSelectionBot
from cov.camera import Camera
from cov.config import OpenEQAConfig
from cov.utils import (
    build_agent_output_paths,
    extract_answer,
    is_mostly_blank,
    process_openeqa_path,
)
from tools.html_generator import HTMLGenerator

log = logging.getLogger(__name__)


def cov_agent(
    episode_history: str = "hm3d-v0/000-hm3d-BFRyYbPCCPE",
    question_id: str = "f2e82760-5c3c-41b1-88b6-85921b9e7b32",
    question: str = "What is the white object on the wall above the TV?",
    gts=["Air conditioning unit"],
    config: OpenEQAConfig = None,
):
    """
    Agent query for one question.
    """
    log.info(f"Question ID: {question_id}")

    html_generator = HTMLGenerator(question_id, config.model.model_name)
    html_generator.set_question(question)
    if gts:
        html_generator.set_gts(gts)

    # TODO: Refactor path design for better organization
    base_dir, screen_shot_dir, local_html_path = build_agent_output_paths(
        config, config.agent, episode_history, question_id
    )
    os.makedirs(screen_shot_dir, exist_ok=True)

    glb_path, pose_path, rgb_img_path = map(
        lambda x: config.dataset_dir / x, process_openeqa_path(episode_history)
    )

    log.info(f"Loading GLB from: {glb_path}")

    cam1 = Camera(ply_path=glb_path, pose_path=pose_path, rgb_img_path=rgb_img_path)

    selbot = ViewSelectionBot(
        question=question,
        rgb_img_list=cam1.view_img_list,
        max_views=config.max_views_k,
        model_config=config.model,
    )

    selection = selbot.invoke()
    pattern = r"selected\s*views?\s*[:=]?\s*\[?([\d,\s]+)\]?"
    match = re.search(pattern, selection, re.IGNORECASE)
    sel_views = []
    if match:
        sel_views = [int(v.strip()) for v in match.group(1).split(",")]
    else:
        log.error("No matching pattern found for 'selected views: '")
    sel_views = sel_views[: config.max_views_k]
    sel_view_path_list = {
        sel_view: cam1.view_img_list[sel_view] for sel_view in sel_views
    }

    best5_urls = list(sel_view_path_list.values())
    html_generator.set_best5(best5_urls)

    birdeye_path = cam1.shot_birdeye_view(screen_shot_dir)
    html_generator.set_birdeye(birdeye_path)

    answer = None
    action_repetition = 0
    prev_action = None
    total_action_cnt = 0
    switch_to_birdeye = False

    chatbot = Chatbot(
        question=question,
        view_ids=list(range(len(cam1.view_pose_list))),
        best5_view_list=sel_view_path_list,
        bird_eye_view=birdeye_path,
        max_views=config.max_views_k,
        min_action_step=config.min_action_step,
        model_config=config.model,
    )

    # query loop
    while total_action_cnt <= 65:
        image_path = (
            birdeye_path if switch_to_birdeye else cam1.screen_shot(screen_shot_dir)
        )
        switch_to_birdeye = False
        total_action_cnt += 1

        try:
            if is_mostly_blank(image_path):
                cam1.switch_back_view()
                image_path = cam1.screen_shot(screen_shot_dir)
                text = "You are moving to a blank view and I switched back. Please resume from the view I provided and continue to give adjustment instructions or provide answer."
                action = chatbot.invoke_in_text(text=text, img_path=image_path)
            else:
                action = chatbot.invoke(image_path, total_action_cnt)

            # 检测重复动作
            if action == prev_action and "switch" not in action:
                action_repetition += 1

            if action_repetition >= 10:
                print(f"Too many times with action: {action}, changing to another...")
                text = "You have repeated this instruction too many times. Please try to use other instructions to get the proper view or answer the question if you can."
                action = chatbot.invoke_in_text(text=text, img_path=image_path)
                action_repetition = 0

            prev_action = action

            html_generator.add_step(image_path, action)

            if "switch to bird-eye-view" in action:
                switch_to_birdeye = True
            else:
                cam1.exec_instruction(action)

            if "done" in action.lower():
                answer = extract_answer(action)
                html_generator.set_answer(answer)
                log.info(f"{question_id} token usage: {chatbot.get_token_usage()}")
                break
        except Exception as e:
            raise e

    # If answer is None, it means exceeding maximum turns
    if answer is None:
        raise Exception("Exceeds maximum turns")

    # Save query history html
    html_content = html_generator.generate_html()
    with open(local_html_path, "w") as f:
        f.write(html_content)
    log.info(f"Local HTML saved to: {local_html_path}")

    return {
        "question_id": question_id,
        "answer": answer,
        "action_steps": total_action_cnt,
        "token_consumption": chatbot.usage_info,
    }


def baseline_agent(
    episode_history: str = "hm3d-v0/000-hm3d-BFRyYbPCCPE",
    question_id: str = "f2e82760-5c3c-41b1-88b6-85921b9e7b32",
    question: str = "What is the white object on the wall above the TV?",
    gts=["Air conditioning unit"],
    config: OpenEQAConfig = None,
):
    """
    Baseline agent that provides all images to the model without COV framework.
    """
    log.info(f"Question ID: {question_id}")

    html_generator = HTMLGenerator(question_id, config.model.model_name)
    html_generator.set_question(question)
    if gts:
        html_generator.set_gts(gts)

    base_dir, screen_shot_dir, local_html_path = build_agent_output_paths(
        config, config.agent, episode_history, question_id
    )
    os.makedirs(screen_shot_dir, exist_ok=True)

    glb_path, pose_path, rgb_img_path = map(
        lambda x: config.dataset_dir / x, process_openeqa_path(episode_history)
    )

    log.info(f"Loading GLB from: {glb_path}")

    cam1 = Camera(ply_path=glb_path, pose_path=pose_path, rgb_img_path=rgb_img_path)

    img_path_list = cam1.view_img_list

    baseline_bot = BaselineBot(
        question=question,
        rgb_img_list=img_path_list,
        model_config=config.model,
    )

    answer = extract_answer(baseline_bot.invoke())

    for img_path in img_path_list:
        html_generator.add_step(img_path, "Image provided to model")

    html_generator.set_answer(answer)

    log.info(f"{question_id} token usage: {baseline_bot.get_token_usage()}")

    html_content = html_generator.generate_html()
    with open(local_html_path, "w") as f:
        f.write(html_content)
    log.info(f"Local HTML saved to: {local_html_path}")

    return {
        "question_id": question_id,
        "answer": answer,
        "token_consumption": baseline_bot.usage_info,
    }
