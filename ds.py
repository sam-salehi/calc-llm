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
    def __init__(self, model_name, max_new_tokens=128, batch_size=8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0  
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder="./offload",
            quantization_config=quant_config
        )
    # FIXME: make reliant on outside batch only.
    def __call__(self, prompts):
        results = []
        # Process in batches to limit GPU memory usage
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i+self.batch_size]
            encoded = self.tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
            
            output = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=self.max_new_tokens
            )

            for j in range(len(batch_prompts)):
                decoded = self.tokenizer.decode(output[j], skip_special_tokens=True)
                results.append(decoded)
            
            torch.cuda.empty_cache()
            
        return results

def clean_dataset(df):
    return df.loc[df["has_image"] == False]

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



def evaluate_model(df,model_id,save_path):
    df["model_response"] = None
    df["evaluation_reason"] = None
    df['eval'] = None 

    evaluator = Gemini()
    model = LocalModel(model_id)

    batch_size = 32
    num_rows = df.shape[0]
    num_batches = (num_rows + batch_size - 1) // batch_size 


    start_time = time.time()
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


    end_time = time.time()
    print(f"Took {end_time - start_time} seconds")

    df.to_csv(save_path,index=False)


if __name__ == "__main__":
    df = pd.read_parquet("hf://datasets/toloka/u-math/data/test-00000-of-00001.parquet")
    df = clean_dataset(df)
    hugging_face_id = "google/gemma-7b-it"
    evaluate_model(df,hugging_face_id,"gemma_7b_it.csv")
