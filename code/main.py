import argparse
import os
import json
import csv
import time
import re
from tqdm import tqdm
from utils import generate_content_and_process, continous_detection_optimization1
from prompt import zero_shot_cot, mdcot_step1, clinical_cot, ltm_cot_step1, ltm_cot_step2
from accelerate import Accelerator
import google.generativeai as genai
import sys

# Set up the model configurations
genai.configure(api_key='xxx')
safety_settings = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

config = genai.types.GenerationConfig(
    stop_sequences=['<|endoftext|>'],
    max_output_tokens=2048,
    temperature=0.6,
    top_k=1,
    top_p=1
)

model = genai.GenerativeModel(model_name="gemini-pro-vision", safety_settings=safety_settings, generation_config=config)
model3 = genai.GenerativeModel(model_name="gemini-1.5-flash", safety_settings=safety_settings)

# Setup accelerator for device
accelerator = Accelerator()
device = accelerator.device
print(f'Using device: {device}')
print('Loading model..')


# Define the evaluation function
def gemini_evaluate(csv_path, model, method, log_dir_path, image_folder):
    os.makedirs(log_dir_path, exist_ok=True)

    # Create log files
    log_file_path = os.path.join(log_dir_path, 'mdcot_log_1.5.json')
    results_file_path = os.path.join(log_dir_path, 'mdcot_results_1.5.json')

    class_labels = ['AD', 'MCI', 'CN']
    metrics = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in class_labels}

    # Log file setup
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as file:
            file.write('')
    if not os.path.exists(results_file_path):
        with open(results_file_path, 'w') as file:
            json.dump({'total_count': 0, 'correct_count': 0, 'accuracy': 0}, file)

    logs = []
    print(f"Begin evaluation, the dataset file path: {csv_path}")

    with open(csv_path, 'r', newline='') as csvfile:
        total_lines = sum(1 for line in csvfile)
        csvfile.seek(0)
        csvreader = csv.DictReader(csvfile)
        progress_bar = tqdm(desc='Evaluating', total=total_lines)
        total_count = 0
        correct_count = 0

        # Loading image description
        image_des_file = '../ADNI_img/trans.jsonl'

        image_des = []
        with open(image_des_file, 'r', encoding='utf-8') as file:
            for line in file:
                image_des.append(json.loads(line))

        for i, row in enumerate(csvreader):
            if i % 10 == 0 and i > 0:
                print("Wait one minute...")
                time.sleep(60)

            progress_bar.update(1)

            patient_info = row['patient_information']
            image_path = row['ImageName']
            true_label = row['Label']
            image_path = os.path.join(image_folder, image_path)

            # Zero-shot COT
            if method == 'zero_shot_cot':
                prediction = 'Unknown'
                try:
                    prompt = zero_shot_cot(patient_info, image_path)
                    response_text = generate_content_and_process(model3, prompt)
                    print(f"Generated response: {response_text}")
                    # Processing the reasoning and answer
                    split_reasoning = response_text.split('Reasoning:')
                    split_choice_answer = response_text.split('Answer:')
                    reasoning = split_reasoning[1].strip() if len(split_reasoning) > 1 else 'No reasoning provided'
                    prediction = re.search(r'([A-C])\. (\w+)\(', split_choice_answer[-1].strip()).group(2) if len(split_choice_answer) > 1 else 'Unknown'
                except Exception as e:
                    print(f"Error processing {row['ImageName']}: {str(e)}")
                print(f'Prediction: {prediction}, True label: {true_label}')

                log_entry = {"id": total_count, "Current data": row, "generated reasoning": reasoning, "extracted prediction": prediction, "true label": row['Label']}
                logs.append(json.dumps(log_entry, ensure_ascii=False))

            # mdcot method
            elif method == 'mdcot':
                cot_prompt = mdcot_step1(patient_info, image_path)
                cot_response = generate_content_and_process(model3, cot_prompt)
                time.sleep(10)
                initial_diagnosis = 'Unknown'
                try:
                    
                    match = re.search(r'Answer:\s*[A-Z]\.\s*([A-Z]+)', cot_response)
                    if match:
                        initial_diagosis = match.group(1)  
                    
                except Exception as e:
                    print(f"An error occurred while processing: {str(e)}")

                print(f"generate response：{cot_response}")


                # Review and refine initial diagnosis and reasoning
                final_response = continous_detection_optimization1(cot_response)
                print(f"final response: {final_response}")
                
                prediction = 'Unkown'
                try:
                    
                    match = re.search(r'Answer:\s*[A-Z]\.\s*([A-Z]+)', cot_response)
                    if match:
                        prediction = match.group(1)  
                except Exception as e:
                    print(f"Error: {str(e)}")

                    

                print(f'final prediction：{prediction}, true label：{true_label}')

                log_entry = {"id": total_count, "Current data": row, "initial response": cot_response, "initial diagnosis": initial_diagnosis, "Optimized Reasoning": final_response, "final diagnosis": prediction, "Actual label": true_label}
                logs.append(json.dumps(log_entry, ensure_ascii=False))


            elif method == 'clinical_cot':

                prediction = 'Unkown'
                try:
                    # few-shot
                    prompt = clinical_cot(patient_info, image_des[total_count]['analysis_result'])
                    response_text = generate_content_and_process(model3, prompt)
                    print(f'generate response：{response_text}')
                    
                    
                    # attain the reasoning part and the answer choice part
                    split_reasoning = response_text.split('Medical Rationale:')
                    split_choice_answer = response_text.split('Answer:')

                          
                    if len(split_reasoning) > 1:
                        reasoning = split_reasoning[1].strip()
                    else:
                        reasoning = 'No reasoning provided'

                    if len(split_choice_answer) > 1:
                        last_answer_segment = split_choice_answer[-1].strip()
                        match = re.search(r'([A-C])\. (\w+)\(', last_answer_segment)
                        if match:
                            prediction = match.group(2)  
                except Exception as e:
                    print(f"while processing {row['ImageName']}, an error occurred: {str(e)}")

                print(f'prediction：{prediction}，true label：{true_label}')

                


                log_entry = {"id": total_count, "Current data": row, "generated response": response_text," generated reasoning": reasoning, "extracted prediction": prediction, "true label": true_label}
                logs.append(json.dumps(log_entry, ensure_ascii=False))



            # Least-to-most cot
            elif method == 'ltm':
                prediction = 'Unknown'  
           
                try:
                    prompt1 = ltm_cot_step1(patient_info, image_path)
                    response_text1 = generate_content_and_process(model3, prompt1)
                    time.sleep(5)
                    print(f"the response of ltm_step1 ：{response_text1}")

                    prompt2 = ltm_cot_step2(patient_info, image_path, response_text1)
                    response_text2 = generate_content_and_process(model3, prompt2)
                    print(f"the response of ltm_step2：{response_text2}")

                    # attain the reasoning part and the answer choice part
                    split_reasoning = response_text2.split('Reasoning:')
                    split_choice_answer = response_text2.split('Answer:')


                    # extract the reasoning       
                    if len(split_reasoning) > 1:
                        
                        reasoning = split_reasoning[1].strip()
                    else:
                        reasoning = 'No reasoning provided'

                    # extract the prediction
                    if len(split_choice_answer) > 1:
                        last_answer_segment = split_choice_answer[-1].strip()
                        match = re.search(r'([A-C])\. (\w+)\(', last_answer_segment)
                        if match:
                            prediction = match.group(2)  
                except Exception as e:
                    print(f"while processing {row['ImageName']}, an error occurred: {str(e)}")

                print(f'prediction：{prediction}, true label：{true_label}')

                # record logs
                log_entry = {"id": total_count, "Current data": row,"generated response1": response_text1, "generated response2":response_text2, "extracted reasoning": prediction, "true label": true_label}
                logs.append(json.dumps(log_entry, ensure_ascii=False))






            # Calculating metrics
            for label in class_labels:
                if prediction == true_label == label:
                    metrics[label]['TP'] += 1
                elif prediction == label and true_label != label:
                    metrics[label]['FP'] += 1
                elif prediction != label and true_label == label:
                    metrics[label]['FN'] += 1

            # Calculate accuracy
            if prediction == row['Label']:
                correct_count += 1
            total_count += 1

        # Writing logs to file
        with open(log_file_path, 'w', encoding="utf8") as log_file:
            log_file.write("[\n" + ",\n".join(logs) + "\n]")

        progress_bar.close()

    # Results calculation
    results = {}
    Accuracy = correct_count / total_count if total_count > 0 else 0
    results['accuracy'] = Accuracy

    for label, counts in metrics.items():
        tp, fp, fn = counts['TP'], counts['FP'], counts['FN']
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        results[label] = {'Precision': precision, 'Recall': recall}

    print(results)

    # Save results to JSON
    with open(results_file_path, 'w') as results_file:
        json.dump(results, results_file, indent=4)

    return results


# Main function to parse arguments and run the evaluation
def main():
    parser = argparse.ArgumentParser(description="Run Gemini evaluation with different methods.")
    parser.add_argument('csv_path', type=str, help="Path to the CSV dataset")
    parser.add_argument('method', choices=['zero_shot_cot', 'mdcot', 'ltm', 'clinical_cot'], help="The method to use for evaluation")
    parser.add_argument('--log_dir', type=str, default='../log/test_logs', help="Directory to store logs")
    parser.add_argument('--image_folder', type=str, default='../dataset/ADNI_img/ADNI_JPG', help="Directory containing images")

    args = parser.parse_args()

    gemini_evaluate(args.csv_path, model, args.method, args.log_dir, args.image_folder)

if __name__ == '__main__':
    main()
