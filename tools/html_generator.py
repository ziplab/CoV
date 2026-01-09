"""Generate a web page for query history."""

import time

import jinja2


class HTMLGenerator:
    def __init__(self, question_id, model_name):
        self.question_id = question_id
        self.model_name = model_name
        self.template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Query History of Question: {{ question_id }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 30px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .step { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                .step-header { display: flex; justify-content: space-between; margin-bottom: 15px; }
                .step-number { font-weight: bold; }
                .image-container { text-align: center; }
                img { max-width: 100%; max-height: 600px; border: 1px solid #eee; }
                .caption { margin-top: 15px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
                .question { font-size: 1.2em; font-weight: bold; margin-bottom: 20px; }
                .answer { margin-top: 30px; padding: 20px; background-color: #e6f7ff; border-radius: 5px; font-weight: bold; }
                .aieval { margin-top: 30px; padding: 20px; background-color: #16f68d; border-radius: 5px; font-weight: bold; }
                .top-images {
                margin-bottom: 30px;
                }
                .first-image {
                text-align: center;
                margin-bottom: 20px;
                }
                .first-image img {
                max-width: 100%;
                max-height: 400px;
                border: 1px solid #ccc;
                border-radius: 5px;
                }
                .grid-images {
                display: flex;
                justify-content: center;
                gap: 15px;
                flex-wrap: wrap;
                }
                .grid-image-item {
                flex: 1 1 150px;
                max-width: 200px;
                text-align: center;
                }
                .grid-image-item img {
                max-width: 100%;
                border: 1px solid #ccc;
                border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Agent History</h1>
                    <h2>{{ question_id }}</h2>
                    {% if model_name %}
                    <div style="text-align: center; font-size: 1.1em; margin-bottom: 10px;">Model: {{ model_name }}</div>
                    {% endif %}
                    <div class="question">Question: {{ question }}</div>
                </div>
                {% if birdeye or best5 %}
                <div class="top-images">
                    {% if birdeye %}
                    <h2 style="text-align: center;">Bird eye view</h2>
                    <div class="first-image">
                    <img src="{{ birdeye }}" alt="Top Image 1">
                    </div>
                    {% endif %}
                    {% if best5 %}
                    <h2 style="text-align: center;">Best 5 views</h2>
                    <div class="grid-images">
                    {% for img_url in best5 %}
                    <div class="grid-image-item">
                        <img src="{{ img_url }}" alt="Top Image">
                    </div>
                    {% endfor %}
                    </div>
                    {% endif %}
                </div>
                {% endif %}
                {% for step in steps %}
                <div class="step">
                    <div class="step-header">
                    <div class="step-number">Step {{ step.number }}</div>
                    <div class="timestamp">{{ step.timestamp }}</div>
                    </div>
                    <div class="image-container">
                    <img src="{{ step.image_url }}" alt="Step {{ step.number }}">
                    </div>
                    <div class="caption">{{ step.caption }}</div>
                </div>
                {% endfor %}
                {% if answer %}
                <div class="answer">
                    Final Answer: {{ answer }}
                </div>
                {% endif %}
                {% for gt in gts %}
                <div class="answer">
                    Ground Truth: {{ gt }}
                </div>
                {% endfor %}
                {% if aieval %}
                <div class="aieval">
                    AI Evaluation: {{ aieval }}
                </div>
                {% endif %}
            </div>
        </body>
        </html>
        """
        self.steps = []
        self.question = ""
        self.answer = ""
        self.gts = []
        self.ai_eval = ""
        self.best5 = []
        self.birdeye = ""

    def _process_image_url(self, image_url: str) -> str:
        # NOTE: To view local html, use a live server. Image paths are relative to the results directory.
        if "shots" in image_url:
            return f"shots/{image_url.split('shots/')[-1]}"
        else:
            # Like data/frames/hm3d-v0/002-hm3d-wcojb4TFT35/comp_color/comp_00010-rgb.png
            return f"/{image_url}"

    def add_step(self, image_url, caption):
        """Add a step to the HTML"""
        # Convert OSS URL to custom domain URL if needed
        image_url = str(image_url)
        
        # Local image url needs to be pre-processed.
        image_url = self._process_image_url(image_url)

        self.steps.append(
            {
                "number": len(self.steps) + 1,
                "image_url": image_url,
                "caption": caption,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def set_question(self, question):
        """Set the question for the session"""
        self.question = question

    def set_gts(self, gts):
        """Set the ground truth for the session"""
        self.gts = gts

    def set_answer(self, answer):
        """Set the final answer"""
        self.answer = answer

    def set_ai_eval(self, ai_eval):
        """Set the AI evaluation"""
        self.ai_eval = ai_eval

    def set_best5(self, best5):
        if "oss" not in str(best5[0]):
            best5 = list(map(lambda x: self._process_image_url(str(x)), best5))
        self.best5 = best5

    def set_birdeye(self, birdeye):
        birdeye = str(birdeye)
        if "oss" not in birdeye:
            birdeye = self._process_image_url(birdeye)
        self.birdeye = birdeye

    def generate_html(self):
        """Generate the HTML content"""
        template = jinja2.Template(self.template)
        return template.render(
            question_id=self.question_id,
            question=self.question,
            steps=self.steps,
            answer=self.answer,
            gts=self.gts,
            aieval=self.ai_eval,
            best5=self.best5,
            birdeye=self.birdeye,
            model_name=self.model_name,
        )
