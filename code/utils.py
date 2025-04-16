import os
import csv
import re
import PIL
import time
from pathlib import Path
from google.api_core import exceptions
import google.generativeai as genai
from prompt import zero_shot_cot

genai.configure(api_key='xxx')  # api_key

safety_settings = [
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
]
# config settings
config = genai.types.GenerationConfig(
        stop_sequences=['<|endoftext|>'],
        max_output_tokens=2048,
        temperature=0.8,
        top_k=1,
        top_p=1
        )
config1 = genai.types.GenerationConfig(
        stop_sequences=['<|endoftext|>'],
        max_output_tokens=2048,
        temperature=0.2,
        top_k=1,
        top_p=1
        )

model = genai.GenerativeModel(model_name = "gemini-pro-vision", safety_settings=safety_settings, generation_config=config)
model1 = genai.GenerativeModel(model_name = "gemini-pro", safety_settings=safety_settings, generation_config=config1)
model2 = genai.GenerativeModel(model_name = "gemini-1.5-pro", safety_settings=safety_settings, generation_config=config)
model3 = genai.GenerativeModel(model_name = "gemini-1.5-flash", safety_settings=safety_settings, generation_config=config)


# profile paths
csv_path = '../dataset/test_data.csv'
image_folder = '../dataset/ADNI_img/ADNI_JPG'


image_folder = Path(image_folder)


    

def clean_generation(response):
    """
    for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
    """
    return response.replace('<unk> ', '').strip()


def is_correct(response, true_label):
    diagnosis_result = response.split('Answer:')[-1].strip().split('.')[0]
    return diagnosis_result == true_label






# Perform iterative checks and corrections to obtain the final correct diagnosis result only based on the initial diagnosis
def verification_and_optimization(initial_response):
    instruction = (
        f'''As a medical review expert, assess the given medical rationale and diagnosis included in the initial_response. If there is no diagnosis, you need to make a new diagnosis, choosing the most appropriate category from the 3 options below: A. AD(Alzheimer's disease) or B. CN(Normal) or C. MCI(mild cognitive impairment). Check if the reasoning is medically sound and free from errors. If the reasoning or diagnosis contains any inaccuracies, specify what they are and suggest corrections. Otherwise, confirm the diagnosis as accurate. Please use 'Valid: Yes' if the diagnosis is correct, otherwise use 'Valid: No'. For any inaccuracies, please specify them under 'Errors: [detailed errors here]'.'''
    )

    prompt = (
        f"Initial_response: {initial_response}\n"
        f"{instruction}\n"
    )

    

    return prompt  


# final decision procession
def decision_model_prompt(input_response, Valid, Errors):

    instruction = (
        "You are the final reviewer in the medical diagnosis process. Please review the validation output from the previous step. "
        "Depending on the validity of the diagnosis, you may need to make corrections."
    )
    
    # If the review shows that the diagnosis is not reasonable, provide details of the error and request a revision
    if Valid == 'No':
        prompt = (
            f"Initial_responese: {input_response}\n"
            f"Identified Errors: {Errors}\n"
            f"{instruction}\n"
            "Based on the identified errors, please correct the errors in the initial_response. "
            "You must provide a diagnosis, and the diagnosis must be in the following format: 'Diagnosis: A. AD(Alzheimer's disease)' or 'Diagnosis: B. CN(Normal)' or 'Diagnosis: C. MCI(mild cognitive impairment)."
        )
  

    return prompt





def continous_detection_optimization1(initial_response, max_iteration=3):
    iteration = 0
    current_response = initial_response  

    while iteration < max_iteration:
        # Review and Optimization
        detect_prompt = verification_and_optimization(current_response)
        detect_response = generate_content_and_process(model3, detect_prompt)
        print(f"reward after detect: {detect_response}")

        # Extracting feedback after review
        Valid, Errors = extract_detect(detect_response)
        
        # If the review shows that the diagnosis is not reasonable, provide details of the error and request a revision
        if Valid == 'No':
            corrected_prompt = decision_model_prompt(current_response, Valid, Errors)
            corrected_output = generate_content_and_process(model3, corrected_prompt)
            #diagnosis, reasoning = extract_answer(corrected_output)
            current_response = corrected_output  
            iteration += 1
        
        # Otherwise, if the review shows that it is reasonable, the final diagnosis result and reasoning will be directly output
        else:
            return current_response  

    # If maximum iterations are reached, return the last corrected output
    return current_response  
    



# Extract the contents of Valid and Errors in the review results
def extract_detect(detect_response):
   
    valid_search = re.search(r"Valid: (Yes|No)", detect_response)
    if valid_search:
        Valid = valid_search.group(1)
    else:
        Valid = "Yes"  
    
   
    error_search = re.search(r"Errors: (.+)", detect_response)
    Errors = error_search.group(1) if error_search else ""

    return Valid, Errors


# Extract the diagnostic results and reasoning from the final output
def extract_answer(response):

    diagnosis = 'Unknown'
    try:
        
        split_reasoning = response.split('Reasoning:')
        split_choice_answer = response.split('Answer:')

        # extract the reasoning part
        if len(split_reasoning) > 1:
            
            reasoning = split_reasoning[1].strip()
        else:
            reasoning = 'No reasoning provided'

        # extract the answer part
        if len(split_choice_answer) > 1:
            
            last_answer_segment = split_choice_answer[-1].strip()
            
            match = re.search(r'([A-C])\. (\w+)\(', last_answer_segment)
            if match:
                diagnosis = match.group(2)  # extract the diagnosis"AD", "CN", or "MCI"

    except Exception as e:
                print(f"Error: {str(e)}")
    

    return diagnosis, reasoning

def extract_diagnosis(optimized_reasoning):
    pattern = r'Answer:\s*([A-Z])\.\s*(\w+)'
    match = re.search(pattern, optimized_reasoning)
    if match:
        return match.group(2)  
    return 'Unkown'



def iterative_feedback_prompt(patient_info, image_path):
    img = PIL.Image.open(image_path)
    
    review_instruction = (
        "Now, review the initial diagnosis and rationale. "
        "Check for any logical inconsistencies or missing information that might affect the diagnosis accuracy. "
        "Consider whether additional information or a reevaluation of the data is necessary."
    )
    
    final_decision_instruction = (
        "Based on the review, make any necessary adjustments to the diagnosis. "
        "Provide a final diagnosis with a revised and detailed medical rationale that addresses any issues identified during the review."
    )
    
    # Placeholder for demonstrations
    test_question = (f"Tester information: {patient_info} Image: {img}. "
                     "Initially: What is your preliminary diagnosis and rationale?\n"
                     "Review: Are there any inconsistencies or adjustments needed?\n"
                     "Final decision: Provide your final diagnosis and updated rationale.")

    prompt = ([f"Review: {review_instruction}\n"
               f"Final decision: {final_decision_instruction}\n"
               f"Test question: {test_question}.", img])

    return prompt




def experiment_one_generate_rationale(patient_info, image_path, diagnosis_result):
    img = PIL.Image.open(image_path)

    instruction = ("You are an expert in Alzheimer's disease. Based on the provided patient information, brain MRI image, and the given diagnosis result, generate detailed medical rationales for the diagnosis ('Medical Rationale:'). ")

    test_question = (f"Tester information: {patient_info} Diagnosis: {diagnosis_result} Image: {img}. Question: Based on the provided text information, image, and diagnosis result, What is the most reasonable medical rationale for the diagnosis? ")

    prompt = ([f"{instruction}\\n"
            f"Test question: {test_question}. You must provide an answer in the following format: 'Medical Rationale:...'. Let's think step by step!", img])

    return prompt






def extract_diagnosis(model_output):
    match = re.search(r'Answer:\s+[A-Z]\.\s+(\w+)', model_output)
    if match:
        return match.group(1)  
    return 'Unkown'



def extract_medical_rationale(model_output):
    """
    This function extracts the 'Medical Rationale' from the model's output.
    
    Args:
    model_output (str): The output text from the model.
    
    Returns:
    str: The extracted 'Medical Rationale' or an empty string if not found.
    """
    # Regular expression to match the 'Medical Rationale' section
    rationale_pattern = re.compile(r'Medical Rationale:(.*?)$', re.DOTALL)
    
    # Search for the pattern in the model's output
    match = rationale_pattern.search(model_output)
    
    if match:
        # Extract the rationale text
        medical_rationale = match.group(1).strip()
        return medical_rationale
    else:
        # Return an empty string if the pattern is not found
        return "UNKOWN"






# Model output processing
def generate_content_and_process(model, prompt):
    
    try:
        
        response = model.generate_content(prompt, stream=True)
        response.resolve()  
        print("The content is generated and the results can be accessed securely.")
        if response.text is not None:
            return response.text  
        else:
            print("The generated content is empty.")
            return "Answer: Unknown"  
    except Exception as e:
        print(f"generation error: {e}")
        return "Answer: Unknown"  # Return default value in case of exception
    
def extract_diagnosis_and_reasoning(output):
    # extract the diagnosis
    diagnosis_start = output.find("Diagnosis:") + len("Diagnosis: ")
    diagnosis_end = output.find(",", diagnosis_start)
    diagnosis = output[diagnosis_start:diagnosis_end].strip()

    # extract the reasoning
    reasoning_start = output.find("Medical Rationale:") + len("Medical Rationale: ")
    reasoning = output[reasoning_start:].strip()

    return diagnosis, reasoning







