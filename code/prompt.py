from PIL import Image



# few-shot testers
Test_info1 = '''Tester information: This is a 78-year-old female, MMSE score is 23.0, CDGLOBAL score is 1.0, NPISCORE is 2.0, FAQTOTAL score is 9.0.'''
Test_image1 = '../ADNI_img/ADNI_JPG/ADNI_023_S_1289_MR_Axial_PD-T2_TSE__br_raw_20070212155103656_13_S26372_I39481.jpg'
Test_info2 = '''Tester information: This is an 83-year-old female, APOE A1=3, APOE A2=4, MMSE score is 23.0, CDGLOBAL score is 1.0, NPISCORE is 2.0, FAQTOTAL score is 16.0.'''
Test_image2 = '../ADNI_img/ADNI_JPG/ADNI_052_S_0952_MR_MP-RAGE__br_raw_20080514150616694_56_S50105_I105582.jpg'
Test_info3 = '''Tester information: This is a 79-year-old female, APOE A1=2, APOE A2=3, MMSE score is 28.0, CDGLOBAL score is 0.0, NPISCORE is 0.0, FAQTOTAL score is 0.0.'''
Test_info4 = '''This is a 73-year-old male, MMSE score is 29.0, CDGLOBAL score is 0.5.'''
Test_info5 = '''Tester information: This is a 86-year-old male, MMSE score is 21.0, CDGLOBAL score is 1.0.'''
Test_info6 = '''Tester information: This is a 80-year-old female, MMSE score is 30.0, CDGLOBAL score is 0.0, NPISCORE is 0.0, FAQTOTAL score is 0.0.'''
Test_image3 = '../ADNI_img/ADNI_JPG/ADNI_006_S_0681_MR_MP-RAGE__br_raw_20081017130027140_136_S57565_I121918.jpg'
Test_image4 = '../ADNI_img/ADNI_JPG/032_S_0978_ADNI_032_S_0978_MR____________MPRAGE__br_raw_20061017112557325_72_S20185_I26408.jpg'
Test_image5 = '../ADNI_img/ADNI_JPG/ADNI_128_S_0805_MR_MPRAGE_br_raw_20060908113231531_83_S18658_I23999.jpg'
Test_image6 = '../ADNI_img/ADNI_JPG/ADNI_014_S_0520_MR_MP-RAGE__br_raw_20080627073640535_98_S52566_I111582.jpg'
ltm_image1 = '../ADNI_img/ADNI_JPG/ltm_image1.jpg'
ltm_image2 = '../ADNI_img/ADNI_JPG/ltm_image2.jpg'
ltm_image3 = '../ADNI_img/ADNI_JPG/ltm_image3.jpg'










# testers‘ image
Test_image1 = Image.open(Test_image1)
Test_image2 = Image.open(Test_image2)
Test_image3 = Image.open(Test_image3)
Test_image4 = Image.open(Test_image4)
Test_image5 = Image.open(Test_image5)
Test_image6 = Image.open(Test_image6)





# construct zero-shot cot prompt
def zero_shot_cot(patient_info, image_path):

    image_path = Image.open(image_path)

    Instruction = '''You are a doctor. Fully analyze the information and the MRI image given, generate medical rationale('Reasoning:...') and diagnose('Answer: ...') the patient. You can utilize the medical rationale. Answer me with only either "A. AD(Alzheimer's disease)", "B. CN(Normal)", or "C. MCI(mild cognitive impairment)".\n'''

    cot_format = '''You can't misdiagnose. Let's think step by step! And the answer must be in the following format: 'Reasoning:...\n Answer:...'mmm\n'''

    Test_question = f'''Question: Tester information:{patient_info}, Based on the provided Tester information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result?'''
    
    return [Instruction + Test_question + cot_format, image_path]

# construct clinical cot prompt
def clinical_cot(patient_info, image_des):
    # prompt = f'''You are a doctor. Generate medical rationale and diagnose the patient based on the information your colleague gave you. You can utilize the medical rationale. Answer me with only either "A. AD(Alzheimer's disease)", "B. CN(Normal)", or "C. MCI(mild cognitive impairment)". You should follow the style of the history.
    demonstration = f'''
    Example 1
    Patient Description: {Test_info1} This MRI image is an axial view of the brain. The ventricles, which are the fluid-filled spaces in the brain, are enlarged. The white matter, which is the tissue that connects different parts of the brain, is decreased. The gray matter, which is the tissue that makes up the outer layer of the brain, is also decreased. These changes are consistent with Alzheimer's disease.\n\nThe image shows atrophy in the medial temporal lobe, which is a region of the brain that is affected in Alzheimer's disease. The atrophy is more pronounced on the left side than on the right side. This asymmetry is also consistent with Alzheimer's disease.\n\nOverall, the features of this MRI image are consistent with a diagnosis of Alzheimer's disease.\n What is the diagnosis of this patient?
    A. AD(Alzheimer's disease) B. CN(Normal) C. MCI(mild cognitive impairment)
    Medical Rationale: The patient is 78 years old, presenting with a MMSE score of 23.0, which indicates mild cognitive impairment or mild dementia. The CDGLOBAL score of 1.0 and FAQTOTAL score of 9.0 support the presence of mild functional impairment and dementia. The presence of the APOE4 allele increases the risk of developing Alzheimer's disease. The MRI image shows signs consistent with Alzheimer's disease, such as possible medial temporal lobe atrophy and ventricular enlargement, which are hallmarks of AD. The combined clinical and imaging findings point towards a diagnosis of Alzheimer's disease. By synthesizing the clinical information and imaging features, we can reasonably conclude that the patient is likely suffering from Alzheimer's disease.\n Answer: A. AD(Alzheimer's disease)

    Example 2
    Patient Description: {Test_info2} The MRI image shows atrophy in the hippocampus, which is a brain region associated with memory. The ventricles, which are the fluid-filled spaces in the brain, are also enlarged. These are signs of Alzheimer's disease (AD). Therefore, this image belongs to the AD category.
    What is the diagnosis of this patient?
    A. AD(Alzheimer's disease) B. CN(Normal) C. MCI(mild cognitive impairment)
    Medical Rationale: To generate a plausible diagnosis based on the provided patient information and corresponding brain MRI image, let's follow the structured approach mentioned previously. Step 1: Patient Information Summary: The presence of APOE4 allele is associated with a higher risk of developing Alzheimer's disease. MMSE score between 20-24 suggests mild dementia. A CDGLOBAL score of 1.0 indicates mild dementia.\n Answer: C. MCI(Mild cognitive impairment)

    Example 3
    Patient Description: {Test_info3} The MRI image shows significant atrophy in the hippocampus, which is a brain region associated with memory. The ventricles, which are the fluid-filled spaces in the brain, are also enlarged. These are signs of Alzheimer's disease (AD). Therefore, this image belongs to the AD category.
    What is the diagnosis of this patient?
    A. AD(Alzheimer's disease) B. CN(Normal) C. MCI(mild cognitive impairment)
    Medical Rationale: In the case of this 79-year-old female, the absence of significant brain atrophy and a relatively high MMSE score of 28.0 indicates preserved cognitive function. The MRI image does not exhibit the typical hallmarks of advanced neurodegenerative diseases, such as pronounced hippocampal atrophy or extensive white matter hyperintensities. There is slight ventricular enlargement, which is commonly observed in normal aging. In light of these observations, the patient's condition can be classified as cognitively normal (CN), with no clear evidence of neurodegenerative disease or significant vascular cognitive impairment.\n Answer: B. CN(Normal)


    Patient Description: {patient_info} {image_des}
    What is the diagnosis of this patient?
    A. AD(Alzheimer's disease) B. CN(Normal) C. MCI(mild cognitive impairment)
    '''
    prompt = f'''You are a doctor. Generate detailed medical rationales for the diagnosis ("Diagnosis:") based on the patient description. These rationales should be the crucial cue for the diagnosis. The answer must be in the following format: 'Reasoning: ...\nAnswer: ...\n'. The diagnosis result(Answer: ...) includes the following three options: 'A. AD(Alzheimer's disease)', 'B. CN(Normal)', 'C. MCI(mild cognitive impairment)'.
    These are some examples:{demonstration}
    Question:
    Patient Description: {patient_info} {image_des}
    What is the diagnosis of this patient?
    A. AD(Alzheimer's disease) B. CN(Normal) C. MCI(mild cognitive impairment)
    '''

    return prompt



# construct the prompt for step 1
def mdcot_step1(patient_info, image_path):

    image_path = Image.open(image_path)
    #  
    Instruction = '''You are a virtual medical expert specializing in Alzheimer's disease. First, thoroughly analyze the provided patient information, which includes age, gender, MMSE score, CDGLOBAL score, NPISCORE, and FAQTOTAL score. Next, examine the corresponding brain MRI image for any critical features like hippocampal atrophy, ventricular enlargement, and cortical changes and so on. Combine these observations to generate a detailed and structured medical rationale for your diagnosis. The rationale should clearly link the observed symptoms and image features with the diagnostic conclusion. All three options are 'A. AD(Alzheimer's disease)', 'B. CN(Normal)', or 'C. MCI(Mild cognitive impairment)'. Note the differences between these three diagnostic options and select the most appropriate diagnosis, even if the evidence is not conclusive. Your diagnostic output should strictly follow this format for clarity and precision.
    '''
    # Instruction = ''' You are a virtual medical expert specializing in Alzheimer's disease. First, examine the corresponding brain MRI image for any critical features like hippocampal atrophy, ventricular enlargement, and cortical changes. Combine these observations to generate a detailed and structured medical rationale for your diagnosis. The rationale should clearly link the observed symptoms and image features with the diagnostic conclusion. Choose the most probable diagnosis from the three options, even if the evidence is not conclusive. The possible outcomes are 'A. AD(Alzheimer's disease)', 'B. CN(Normal)', or 'C. MCI(Mild cognitive impairment)'. Your diagnostic output should strictly follow this format for clarity and precision.
    # '''
    # # wothout special task knowledge
    # Instruction = ''' Generate corresponding Alzheimer's disease medical rationale and diagnosis result based on the given tester information input and questions. All three options are 'A. AD(Alzheimer's disease)', 'B. CN(Normal)', or 'C. MCI(Mild cognitive impairment)'. Note the differences between these three diagnostic options and select the most appropriate diagnosis, even if the evidence is not conclusive. Your diagnostic output should strictly follow this format for clarity and precision.
    # '''

    cot_format = '''Let's think step by step! And the answer must be in the following format: 'Reasoning:...\n Answer:...'mmm\n'''

    
    Test_question = f'''Question: Tester information:{patient_info}, Based on the provided Tester information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result?{cot_format}'''
    
    prompt = [f'Task Instruction:{Instruction}\n',f'Based on the demonstrations above, answer the following diagnostic questions:\n Test_question: {Test_question}', image_path]
    # return [Test_question + cot_format, image_path]
    # return [Test_question, image_path]
    return prompt


# construct few-shot cot prompt
def few_shot_cot(patient_info, image_path):

    image_path = Image.open(image_path)

    Instruction = '''You are a doctor. Fully analyze the information and the MRI image given, generate medical rationale('Reasoning:...') and diagnose('Answer: ...') the patient. You can utilize the medical rationale. Answer me with only either "A. AD(Alzheimer's disease)", "B. CN(Normal)", or "C. MCI(mild cognitive impairment)".\n'''

    cot_format = '''Let's think step by step! The generated reasoning must fully consider the test subject's text information description and MRI image features and generate detailed diagnostic reasoning, and the answer must be in the following format: 'Reasoning: ...\nAnswer: ...\n'. The diagnosis result(Answer: ...) includes the following three options: 'A. AD(Alzheimer's disease)', 'B. CN(Normal)', 'C. MCI(mild cognitive impairment)'.\n '''


    Demonstration_1 = (
        f'''{Instruction} Question: Tester information:{Test_info1}, MRI image: <image1>. Based on the provided text information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result? {cot_format} 
        Reasoning: The patient is 78 years old, presenting with a MMSE score of 23.0, which indicates mild cognitive impairment or mild dementia. The CDGLOBAL score of 1.0 and FAQTOTAL score of 9.0 support the presence of mild functional impairment and dementia. The MRI image shows signs consistent with Alzheimer's disease, such as possible medial temporal lobe atrophy and ventricular enlargement, which are hallmarks of AD. The combined clinical and imaging findings point towards a diagnosis of Alzheimer's disease. By synthesizing the clinical information and imaging features, we can reasonably conclude that the patient is likely suffering from Alzheimer's disease.\n
        Answer: A. AD(Alzheimer's disease)''')

    Demonstration_2 = (
        f'''{Instruction} Question: Tester information:{Test_info2}, MRI image: <image2>. Based on the provided text information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result? {cot_format}
        Reasoning: For this 83-year-old female patient with an MMSE score of 23.0, which points to moderate cognitive deficits, the MRI image shows moderate ventricular enlargement and sulcal widening, without significant hippocampal atrophy. This pattern is more suggestive of vascular contributions to cognitive impairment or a mixed pathology rather than pure Alzheimer's disease. However, given the patient's age and the absence of severe cortical atrophy or significant hippocampal volume loss, it's reasonable to conclude that the cognitive decline has not yet reached the severity of dementia. Therefore, the patient's condition can be categorized as mild cognitive impairment (MCI).
        Answer: C. MCI(Mild cognitive impairment)''')

    Demonstration_3 = (
        f'''{Instruction} Question: Tester information:{Test_info3}, MRI image: <image3>. Based on the provided text information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result? {cot_format} 
        Reasoning: In the case of this 79-year-old female, the absence of significant brain atrophy and a relatively high MMSE score of 28.0 indicates preserved cognitive function. The presence of APOE3 allele is neutral with respect to Alzheimer's risk. And the MRI image does not exhibit the typical hallmarks of advanced neurodegenerative diseases, such as pronounced hippocampal atrophy or extensive white matter hyperintensities. There is slight ventricular enlargement, which is commonly observed in normal aging. In light of these observations, the patient's condition can be classified as cognitively normal (CN), with no clear evidence of neurodegenerative disease or significant vascular cognitive impairment.\n
        Answer: B. CN(Normal)''')

    Test_question = (f'''{Instruction} Question: Tester information:{patient_info}, MRI image: <image4>. Based on the provided text information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result? {cot_format}''')

    prompt = [f'Demonstration1: {Demonstration_1}', Test_image1,f'Demonstration2: {Demonstration_2}', Test_image2, f'Demonstration3: {Demonstration_3}', Test_image3, f'Based on the demonstrations above, answer the following diagnostic questions:\n Test_question: {Test_question}', image_path]

    return prompt

# construct ps_cot prompt
def ps_cot(patient_info, image_path):

    image_path = Image.open(image_path)

    Instruction = '''You are a doctor of neurologists. Analyze the information and the MRI image provided. Your response should focus on presenting evidence from the data and MRI image, considering patient age, cognitive test scores, and MRI findings. Diagnose with one of the following: "A. AD(Alzheimer's disease)", "B. CN(Normal)", or "C. MCI(mild cognitive impairment)".\n'''

    cot_format = '''Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer.The diagnosis should reflect the likelihood and should not be absolute unless the data conclusively supports it. The answer format should be: 'Reasoning: ...\n Answer: ...\n' with the diagnosis options being 'A. AD(Alzheimer's disease)', 'B. CN(Normal)', 'C. MCI(mild cognitive impairment)'.\n '''

   

    Test_question = (f'''{Instruction} Question: Tester information:{patient_info}. Based on the provided text information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result? {cot_format}''', image_path)

    
    prompt = [f'{Test_question}', image_path]

    return prompt




# construct ltm_cot prompt
def ltm_cot_step1(patient_info, image_path):

    image_path = Image.open(image_path)

    Instruction = '''You are a neurologist. Given a complex medical case, your task is to break it down into simpler subquestions that can be solved step-by-step. Each subquestion should explore a specific aspect of the given information and MRI findings. The final sub-question should be about the most appropriate diagnosis. Be cautious with the diagnosis and consider all possibilities.\n'''

    cot_format = '''Step-by-step breakdown: 1. First, identify the necessary steps to solve the problem. 2. Then, generate a subquestion for each step.You must answer in the following format:\n"Subquestions: 1. [subquestion]\n2. [subquestion]\n ...\n"'''

   

    Test_question = f'''{Instruction}Original Question: Tester information:{patient_info}, Based on the provided text information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result?{cot_format}'''


    prompt = [f'Test_question: {Test_question}', image_path]

    return prompt

# construct ltm_cot step2 prompt
def ltm_cot_step2(patient_info, image_path, subquestions):

    image_path = Image.open(image_path)

    Instruction = '''You are a neurologist, and given a series of sub-questions and their answers, your task is to solve each sub-question step by step and analyze the impact of the results on the final diagnosis. Finally, you must combine the answers to all sub-questions for a comprehensive analysis, taking into account various factors as much as possible, especially the impact of significant features in the subject's intelligence test information or significant features in the MRI image on the diagnosis. When features in text information conflict with image features, the more significant features are more decisive for the final diagnosis. For example, when the MMSE score is high, the diagnosis is less likely to be AD and tends to be CN. Finally, you must choose the most appropriate diagnosis from the following three diagnoses as the answer to the initial question and output: "A. AD (Alzheimer's disease)", "B. CN (normal)" or "C. MCI (mild cognitive impairment)." You must output the answer to this initial question, choosing from the three options above..\n'''

    cot_format = '''You must answer in the following format:\nOriginal Question: {Original Question}\n\nSubquestion 1: {subquestion_1}\nAnswer 1: {answer_1}\n\nSubquestion 2: {subquestion_2}\nAnswer 2: {answer_2}\n\n...\n\nReasoning: ...\nAnswer: ...\nFinal the most reasonable diagnosis is [option]. The options include the following three types: 'A. AD(Alzheimer's disease)', 'B. CN(Normal)', 'C. MCI(mild cognitive impairment)'. You can't misdiagnose.\n'''

   

    Test_question = f'''{Instruction}Original Question: Tester information:{patient_info}, Based on the provided text information and the image, What do you think is the most reasonable diagnosis rationales and diagnosis result?{subquestions}{cot_format}'''

    prompt = [f'Test_question: {Test_question}', image_path]

    return prompt








