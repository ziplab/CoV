from lc_demo import agent
import json
import os
import argparse
import wandb
import multiprocessing as mp
from multiprocessing import queues
from functools import partial
from tqdm import tqdm
import time
import queue as queue_module
import threading

# Import the scene to task number mapping
# from multi_agent import scene2tasknum

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def process_question(item):
    """Process a single question using the agent."""
    
    scene_id = item['scene_id']
    question = item['question']
    question_id = item['question_id']
    gt = item['answers']
    
    # Call the agent function
    try:
        answer, turns, url, ai_eval = agent(scene_id=scene_id, question_id=question_id, question=question, gts=gt)
        
        # Create the output info
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
        
        return output_info, question_id, url
    except Exception as e:
        print(f"Error processing question {question_id}: {str(e)}")
        return None, question_id, None

def write_to_file(output_file, output_info, is_first=False, is_last=False):
    """Write the output info to the file with proper JSON formatting."""
    with open(output_file, 'a') as f:
        if is_first:
            f.write('[')
        f.write(json.dumps(output_info, indent=4))
        if not is_last:
            f.write(',')
        else:
            f.write(']')

def process_batch(batch):
    """Process a batch of questions in a single process."""
    results = []
    for item in batch:
        result = process_question(item)
        if result[0] is not None:
            results.append(result)
    return results

def create_wandb_table_from_results(results_file):
    """Create a wandb table from the results in ans.json."""
    with open(results_file, 'r') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading results file: {e}")
            return None
    
    # Create the table
    table = wandb.Table(columns=["step", "question_id", "question_link"])
    
    # Add each result to the table
    for i, result in enumerate(results):
        question_id = result.get("question_id", "unknown")
        url = result.get("url", "")
        table.add_data(i, question_id, url)
    
    return table
def process_100_item(index, data, num_processes=4, use_threads=False):
     # Filter data for the specified scene_id
    data = [item for item in data if item['scene_id'] != 'scene0414_00']

    print(f"Found {len(data)} items")
    
    filtered_data = data
    
    print(f"After filtering, {len(filtered_data)} questions remain")
    
    if len(filtered_data) == 0:
        print("No questions to process after filtering. Exiting.")
        return []
    
    # Create output directory
    output_dir = f'{root_dir}/cov/mp_outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ans_{index}.json")
    output_qid_url_file = os.path.join(output_dir, "urls_{index}.log")
    if os.path.exists(output_file):
        return 
    print(f"Output will be saved to {output_file}")
    
    # Initialize the output file
    with open(output_file, 'w') as f:
        f.write('[')
    
    # Initialize wandb
    # print("Initializing wandb")
    # run = wandb.init(
    #     project="CoV-ScanQA-trace",   
    #     entity="adelaiseg",
    # )
    
    # Determine the number of items per batch
    num_batches = min(num_processes, len(filtered_data))
    if num_batches <= 0:
        num_batches = 1
    
    batch_size = len(filtered_data) // num_batches
    if batch_size <= 0:
        batch_size = 1
    
    # Create batches
    batches = []
    for i in range(0, len(filtered_data), batch_size):
        batches.append(filtered_data[i:i + batch_size])
    
    print(f"Created {len(batches)} batches with approximately {batch_size} items each")
    
    # Process batches
    all_results = []
    step = 0
    
    if use_threads:
        # Use threads
        from concurrent.futures import ThreadPoolExecutor
        print("Using ThreadPoolExecutor")
        
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            # Process results as they complete
            for future in tqdm(futures, total=len(futures), desc="Processing batches"):
                batch_results = future.result()
                for output_info, question_id, url in batch_results:
                    if output_info is not None:
                        # Write to file
                        with open(output_qid_url_file, 'a') as f:
                            f.write(f'{question_id}: {url}\n')
                        # Write to file
                        is_last = (step == len(filtered_data) - 1)
                        write_to_file(output_file, output_info, is_first=False, is_last=is_last)
                        print(f"Saved answer for question ID {question_id}")
                        all_results.append(output_info)
                        step += 1
    else:
        # Use multiprocessing
        print("Using multiprocessing.Pool")
        with mp.Pool(processes=num_processes) as pool:
            # Use partial to fix the scene_id parameter
            process_func = partial(process_batch)
            
            # Map the function over the batches
            for batch_results in tqdm(pool.imap_unordered(process_func, batches), 
                                    total=len(batches), 
                                    desc="Processing batches"):
                for output_info, question_id, url in batch_results:
                    if output_info is not None:
                        # Write to file
                        with open(output_qid_url_file, 'a') as f:
                            f.write(f'{question_id}: {url}\n')
                        # Write to file
                        is_last = (step == len(filtered_data) - 1)
                        write_to_file(output_file, output_info, is_first=False, is_last=is_last)
                        print(f"Saved answer for question ID {question_id}")
                        all_results.append(output_info)
                        step += 1
    
    # Finalize the output file if it's not already done
    with open(output_file, 'rb+') as f:
        f.seek(-1, os.SEEK_END)
        last_char = f.read(1)
        if last_char != b']':
            f.seek(0, os.SEEK_END)
            f.write(b']')
    
    print(f"Processing complete. {len(all_results)} results saved.")
    
    # Create and log the wandb table at the end
    # print("Creating wandb table from results file")
    # table = create_wandb_table_from_results(output_file)
    # if table is not None:
    #     wandb.log({"generated_questions": table})
    
    # wandb.finish()
    return all_results
    
def main(num_processes=4, use_threads=False):
    # NOTE: This is an example script. Configure paths according to your setup.
    root_dir = os.environ.get('COV_ROOT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Starting with num_processes={num_processes}, use_threads={use_threads}")
    
    # Load the JSON file
    json_path = os.path.join(root_dir, 'data/ScanQA_v1.0_val.json')
    print(f"Loading data from {json_path}")
    with open(json_path, 'r') as file:
        data = json.load(file)
    # Process data in chunks of 100
    chunk_size = 100
    total_chunks = (len(data) + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"Will process data in {total_chunks} chunks of {chunk_size} items each")
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(data))
        chunk_data = data[start_idx:end_idx]
        
        print(f"\n--- Processing chunk {chunk_idx + 1}/{total_chunks} (items {start_idx+1} to {end_idx}) ---")
        print(f"Chunk size: {len(chunk_data)}")
        
        # Process this chunk
        process_100_item(
            index=chunk_idx + 1,  # Start from 1 for output file naming
            data=chunk_data,
            num_processes=num_processes,
            use_threads=use_threads
        )
        

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ScanQA tasks using multiple CPU processes or threads")
    parser.add_argument("--num_processes", type=int, help="Number of processes/threads to use", default=16)
    parser.add_argument("--use_threads", action="store_true", help="Use threads instead of processes")
    args = parser.parse_args()
    
    main(num_processes=args.num_processes, use_threads=args.use_threads) 