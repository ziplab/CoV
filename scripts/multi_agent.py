from lc_demo import agent
import json
import os
from tqdm import tqdm
import argparse
import wandb

completed_scenes = ['scene0015_00', 'scene0063_00', 'scene0064_00',
                    'scene0077_00', 'scene0081_00', 'scene0100_00',
                    'scene0278_00', 'scene0304_00', 'scene0314_00', 'scene0328_00',
                    'scene0329_00', 'scene0334_00', 'scene0338_00']

def main(resume: bool = False):
    # NOTE: This is an example script. Configure paths according to your setup.
    root_dir = os.environ.get('COV_ROOT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load the JSON file
    json_file = os.environ.get('SCANQA_JSON', os.path.join(root_dir, 'data/ScanQA_v1.0_val.json'))
    with open(json_file, 'r') as file:
        data = json.load(file)
    # Save the answer to a file
    output_dir = os.path.join(root_dir, 'results', 'scanqa_results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ans.json")
    stats_log = os.path.join(output_dir, 'stats.log')
    # correctness_table = wandb.Table(columns=["scene_id", "correct", "total"])
    aieval_correct_cnt = 0
    total = 0
    if resume:
        with open(stats_log, 'r') as f:
            line1 = f.readline()
            line1 = line1.replace('correct/total: ','')
            aieval_correct_cnt, total = line1.split('/')
            aieval_correct_cnt = int(aieval_correct_cnt)
            total = int(total)
    else:
        with open(output_file, 'a') as f:
            f.write('[')

    for i, item in tqdm(enumerate(data)):
        if resume and i+1 <= total:
            continue
        scene_id = item['scene_id']
        question = item['question']
        question_id = item['question_id']
        gt = item['answers']

        # if scene_id in completed_scenes:
        #     continue
        # Call the main function from lc_demo
        answer, turns, url, ai_eval = agent(scene_id=scene_id, question_id=question_id, question=question, gts=gt)

        if 'incorrect' not in ai_eval.lower():
            aieval_correct_cnt += 1
        output_info = {
            "scene_id": scene_id,
            "question_id": question_id,
            "question": question,
            "pred_ans": answer,
            "gt": gt,
            "interaction turns": turns,
            "aieval": ai_eval,
            "url": url
        }
        with open(output_file, 'a') as f:
            f.write(json.dumps(output_info, indent=4) + ',')
        
        with open(stats_log, 'w') as f:
            f.write(f'correct/total: {aieval_correct_cnt}/{i+1}\nMost recently Finished question: {question_id}\n')
        total = i+1
        print(f"Saved answer for question ID {question_id} to {output_file}")
        
    with open(output_file, 'w') as f:
        f.write(f'\ntotal AI eval correctness: {aieval_correct_cnt} / {len(data)}\n]')
    # correctness_table.add_data(scene_id, aieval_correct_cnt, len(data))
    # wandb.log({f"ScanQA-total-AI-eval": correctness_table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--resume", action='store_true', help="resume from exception")

    # run = wandb.init(
    #     project="CoV-ScanQA-trace",   
    #     entity="adelaiseg",
    #     )
    args = parser.parse_args()
    main(args.resume)
    # wandb.finish()
    