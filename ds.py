import pandas as pd
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import BitsAndBytesConfig
import torch
from google import genai 
from prompts import evaluation_context, question_context
import re
import os 
from dotenv import load_dotenv

# umath dataset for evaluation 
df = pd.read_parquet("hf://datasets/toloka/u-math/data/test-00000-of-00001.parquet")



load_dotenv()
google_api_key = os.getenv("google_api_key")

device = "cuda"
torch.cuda.empty_cache()


torch.set_float32_matmul_precision("high")


class Gemini:
    def __init__(self):
        self.client  = genai.Client(api_key=google_api_key)
    
    def __call__(self,prompt):
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents = [prompt]
        )
        return response.text.strip() 



class LocalModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="./offload",
        )


    def __call__(self,prompts):
        encoded = self.tokenizer(prompts,padding=True,return_tensors="pt").to(device)
        outputs = self.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask = encoded["attention_mask"]
        )
        decoded = [self.tokenizer.decode(o,skip_special_tokens=True) for o in outputs]
        return decoded
    #
    # def __call__(self, prompt):
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
    #     outputs = self.model.generate(
    #         **inputs,
    #         max_new_tokens=150,
    #         temperature=0.9,
    #         top_p =0.9,
    #         repetition_penalty=1.2,
    #         do_sample=True 
    #     )
    #
    #     text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return text
    #
    #


# model = Gemini(gemini_key)
# evaluator = Gemini(gemini_key, "gemini-1.5-pro")
#
# TODO: speed up through parralelization


def clean_dataset(df):
# TODO: remove rows with has_image to true 
    return df.loc[df["has_image"] == True]

# TODO: parralelize all ops below. Possibly with taichi.
def generate_question_prompts(questions):
    context = question_context
    res = []
    for question in questions: 
        res.append(f"<context>  {context} </context> Answer the following: <Question> {question} </Question>")
    return res


def generate_response_prompts(questions, model_responses, golden_responses):
    context = evaluation_context
    res = []
    for i in range(len(questions)):
        res.append(f"<context>{context}</context> <Question>{questions[i]}</Question> \n <GoldAnswer> {golden_responses[i]} </GoldAnswer> <ModelAnswer> {model_responses[i]} </ModelAnswer>")
    return res


def evaluate_responses(evaluator, questions, responses, golden_responses):
    prompts = generate_response_prompts(questions, responses, golden_responses)
    responses = []
    for prompt in prompts:
        print(len(responses))
        res = evaluator(prompt)
        responses.append(res)
    return responses 


def extract_evals(texts):
    res = []
    for text in texts: 
        pattern = r"<Evaluation>(.*?)</Evaluation>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            res.append(match.group(1).strip())  # return content inside tags.
        else:
            res.append(None)
    return res 


def extract_reasonings(texts):
    res = []
    for text in texts:
        pattern = r"<Reasoning>(.*?)</Reasoning>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            res.append(match.group(1))
        else: 
            res.append(None)
    return res 


def log_evals(isCorrects,batch_num,total_batches):
    count = sum(isCorrects) 
    print(f"{batch_num}/{total_batches} complete with score {count}/{len(isCorrects)}")

# TODO: make the models handle batching 

df["model_response"] = None
df["evaluation_reason"] = None
df['eval'] = None 

df = clean_dataset(df)

evaluator = Gemini()
model = LocalModel("meta-llama/Llama-3.1-8B-Instruct")



batch_size = 32
num_rows = df.shape[0]
num_batches = (num_rows + batch_size - 1) // batch_size 

for i in range(0, num_rows, batch_size):
    try:
        j = min(i + batch_size, num_rows)  
        questions = df.iloc[i:j]["problem_statement"].to_list()
        golden_responses = df.iloc[i:j]["golden_answer"].to_list()
        question_prompts = generate_question_prompts(questions)
        answers = model(question_prompts)
        evaluations = evaluate_responses(evaluator, questions, answers, golden_responses)

        evals = extract_evals(evaluations)
        reasonings = extract_reasonings(evaluations)


        assert len(answers) == len(questions), f"{len(answers)} != {len(questions)}"
        assert len(reasonings) == len(questions), f"{len(reasonings)} != {len(questions)}"
        df.iloc[i:j,df.columns.get_loc("model_response")] = answers
        df.iloc[i:j,df.columns.get_loc("evaluation_reason")] = reasonings

        isCorrects = list(map(lambda eval: eval == "Correct", evals))
        df.iloc[i:j,df.columns.get_loc("eval")] = isCorrects  

        log_evals(isCorrects,i // batch_size + 1, num_batches)
    except Exception as e:
        print("failed")
        print(f"Batch {i}:{j} -> {len(questions)} questions")
        print(f"answers={len(answers)}, reasonings={len(reasonings)}, evals={len(evals)}")
        print(e) 
        continue 


df.to_csv("gemma_2b_it_val.csv", index=False)

