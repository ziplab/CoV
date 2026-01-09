"""
Main script for open-eqa experiments.
"""

import json
import logging

import hydra
from dotenv import load_dotenv

from cov.agents import cov_agent, baseline_agent
from cov.config import OpenEQAConfig
from cov.utils import get_results_path

load_dotenv()
log = logging.getLogger(__name__)

AGENT_REGISTRY = {"cov": cov_agent, "baseline": baseline_agent}

@hydra.main(version_base=None, config_name="openeqa")
def main(cfg: OpenEQAConfig):
    log.info(cfg)

    # Load OpenEQA questions
    with open(cfg.dataset.question_file, "r") as f:
        questions = json.load(f)

    log.info(f"Loaded {len(questions)} questions from {cfg.dataset.question_file}")

    result_path = get_results_path(cfg)

    # Load exsiting questions
    processed_ids = set()
    results = []
    if result_path.exists():
        with open(result_path, "r") as f:
            results = json.load(f)
            processed_ids = {r["question_id"] for r in results}
        log.info(f"Found {len(processed_ids)} already processed questions")

    agent_func = AGENT_REGISTRY[cfg.agent]
    for idx, item in enumerate(questions):
        question_id = item["question_id"]

        # Skip exsiting
        if question_id in processed_ids:
            log.info(
                f"Skipping already processed question {idx + 1}/{len(questions)}: {question_id}"
            )
            continue

        log.info(f"Processing question {idx + 1}/{len(questions)}: {question_id}")

        try:
            result = agent_func(
                episode_history=item["episode_history"],
                question_id=question_id,
                question=item["question"],
                gts=[item["answer"]] if "answer" in item else None,
                config=cfg,
            )
            results.append(result)
            # Store data instantly in case of losing result data accidently.
            # Use w mode because original results have been stored.
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            log.info(
                f"Successfully processed {question_id}, total completed: {len(results)}"
            )

        except Exception as e:
            log.exception(f"Failed to process question {question_id}: {e}")
            continue

    log.info(f"All processing complete. Total results: {len(results)}")
    log.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
