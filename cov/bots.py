import base64
import logging
import os

from litellm import completion

from cov.config import ModelConfig
from cov.utils import load_prompt_template

log = logging.getLogger(__name__)


class EvalBot:
    def __init__(
        self,
        answer: str = None,
        gts: list = [],
        question: str = None,
        *,
        model_config: ModelConfig,
    ):
        self.model_config = model_config
        self.messages = []
        self.usage_info = {}

        template = load_prompt_template("eval_bot.j2")

        user_prompt = template.render(
            question=question,
            answer=answer,
            gts=gts,
        )

        self.messages.append({"role": "user", "content": user_prompt})

    def invoke(self):
        response = completion(
            model=self.model_config.model_name,
            api_base=os.environ[self.model_config.api_base_env],
            api_key=os.environ[self.model_config.api_key_env],
            custom_llm_provider="openai",
            messages=self.messages,
            temperature=0,
        )

        if hasattr(response, "usage"):
            self.usage_info = response.usage

        content = response.choices[0].message.content
        log.info(content)
        return content.split("</think>")[1] if "</think>" in content else content

    def get_token_usage(self):
        return self.usage_info


class ViewSelectionBot:
    def __init__(
        self,
        question: str = None,
        rgb_img_list: list = [],
        pose_matrix_list: list = [],
        max_views: int = 5,
        *,
        model_config: ModelConfig,
    ):
        self.model_config = model_config
        self.messages = []
        self.usage_info = {}

        template = load_prompt_template("view_selection_bot.j2")

        system_prompt = template.render(
            question=question,
            view_ids=list(range(len(rgb_img_list))),
            max_views=max_views,
        )

        self.messages.append({"role": "system", "content": system_prompt})

        # Add image messages
        for view_id, img_path in enumerate(rgb_img_list):
            with open(img_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            content = [
                {
                    "type": "text",
                    "text": f"This is the image corresponding to view id: {view_id}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}",
                    },
                },
            ]
            self.messages.append({"role": "user", "content": content})

    def invoke(self):
        response = completion(
            model=self.model_config.model_name,
            api_base=os.environ[self.model_config.api_base_env],
            api_key=os.environ[self.model_config.api_key_env],
            custom_llm_provider="openai",
            messages=self.messages,
            temperature=0,
        )

        # Extract usage information
        if hasattr(response, "usage"):
            self.usage_info = response.usage

        content = response.choices[0].message.content
        log.info(content)
        return content.split("</think>")[1] if "</think>" in content else content

    def get_token_usage(self):
        return self.usage_info


class Chatbot:
    def __init__(
        self,
        question: str = None,
        view_ids: list = [],
        best5_view_list: dict = {},
        bird_eye_view: str = None,
        max_views: int = 5,
        *,
        model_config: ModelConfig,
        min_action_step: int = 3,
    ):
        self.model_config = model_config
        self.messages = []
        self.usage_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.min_action_step = min_action_step

        template = load_prompt_template("chatbot.j2")

        system_prompt = template.render(
            question=question,
            view_ids=view_ids,
            max_views=max_views,
            min_action_step=min_action_step,
        )

        self.messages.append({"role": "system", "content": system_prompt})

        for view_id, img_path in best5_view_list.items():
            with open(img_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            content = [
                {
                    "type": "text",
                    "text": f"This is one of the best images, corresponding to view id: {view_id}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}",
                    },
                },
            ]
            self.messages.append({"role": "user", "content": content})

        with open(bird_eye_view, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        content = [
            {
                "type": "text",
                "text": "This is the image of the scene from bird eye's view for your reference",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                },
            },
        ]
        self.messages.append({"role": "user", "content": content})

    def invoke(self, img_path: str, step: int):
        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        content = [
            {
                "type": "text",
                "text": f"Here is the provided view image based on your adjustment. Currently you are in step {step}. Perform ONLY ONE action per step. Remember your minium action step budget is {self.min_action_step}. If you have reached minimum step budget and you are sure you have collected enough information, give your answer following pattern 'done+[answer]'.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                },
            },
        ]

        self.messages.append({"role": "user", "content": content})

        response = completion(
            model=self.model_config.model_name,
            api_base=os.environ[self.model_config.api_base_env],
            api_key=os.environ[self.model_config.api_key_env],
            custom_llm_provider="openai",
            messages=self.messages,
            temperature=0,
        )

        if hasattr(response, "usage"):
            usage = response.usage
            self.usage_info["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self.usage_info["completion_tokens"] += getattr(
                usage, "completion_tokens", 0
            )
            self.usage_info["total_tokens"] += getattr(usage, "total_tokens", 0)

        assistant_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_content})

        log.info(assistant_content)
        return (
            assistant_content.split("</think>")[1]
            if "</think>" in assistant_content
            else assistant_content
        )

    def invoke_in_text(self, text: str, img_path: str):
        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        content = [
            {
                "type": "text",
                "text": text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                },
            },
        ]

        self.messages.append({"role": "user", "content": content})

        response = completion(
            model=self.model_config.model_name,
            api_base=os.environ[self.model_config.api_base_env],
            api_key=os.environ[self.model_config.api_key_env],
            custom_llm_provider="openai",
            messages=self.messages,
            temperature=0,
        )

        if hasattr(response, "usage"):
            usage = response.usage
            self.usage_info["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self.usage_info["completion_tokens"] += getattr(
                usage, "completion_tokens", 0
            )
            self.usage_info["total_tokens"] += getattr(usage, "total_tokens", 0)

        assistant_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_content})

        log.info(assistant_content)
        return (
            assistant_content.split("</think>")[1]
            if "</think>" in assistant_content
            else assistant_content
        )

    def get_token_usage(self):
        return self.usage_info


class BaselineBot:
    """
    Baseline chat bot without COV framework.
    """

    def __init__(
        self,
        question: str = None,
        rgb_img_list: list = [],
        *,
        model_config: ModelConfig,
    ):
        self.model_config = model_config
        self.messages = []
        self.usage_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        template = load_prompt_template("baseline_bot.j2")

        system_prompt = template.render(question=question)

        self.messages.append({"role": "system", "content": system_prompt})

        # NOTE There is a bug in litellm or llm providers, so that you must pass image like f"data:image/png;base64,{image_data}". Or it fails.
        for img_path in rgb_img_list:
            with open(img_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            content = [
                {
                    "type": "text",
                    "text": "Here is one of the images of the scene",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_data}",
                },
            ]
            self.messages.append({"role": "user", "content": content})

    def invoke(self):
        query_message = {
            "role": "user",
            "content": "Your answer should STRICTLY FOLLOW the pattern 'done+[answer]'. Please give your answer:",
        }

        messages_to_send = self.messages + [query_message]
        response = completion(
            model=self.model_config.model_name,
            messages=messages_to_send,
            api_base=os.environ[self.model_config.api_base_env],
            api_key=os.environ[self.model_config.api_key_env],
            custom_llm_provider="openai",
            temperature=0,
        )

        if hasattr(response, "usage"):
            usage = response.usage
            self.usage_info["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
            self.usage_info["completion_tokens"] = getattr(
                usage, "completion_tokens", 0
            )
            self.usage_info["total_tokens"] = getattr(usage, "total_tokens", 0)

        content = response.choices[0].message.content
        return content.split("</think>")[1] if "</think>" in content else content

    def get_token_usage(self):
        return self.usage_info
