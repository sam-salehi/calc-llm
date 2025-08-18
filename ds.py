import pandas as pd
import numpy as np
import time
import torch
from google import genai 
import re
import os 
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from transformers import Trainer, TrainingArguments 
from datasets import Dataset 

from prompts import evaluation_context, question_context
from clp import CLP_PATH, train_test_split
from models import Gemini, LocalModel

# umath dataset for evaluation 








def clean_dataset(df):
    return df.loc[df["has_image"] == False]

def generate_question_prompts(questions):
    context = question_context
    res = []
    for question in questions: 
        res.append(f"<context>  {context} </context> Answer the following: <Question> {question} </Question>")
    return res


def generate_evaluation_prompts(questions, model_responses, golden_responses):
    context = evaluation_context
    res = []
    for i in range(len(questions)):
        res.append(f"<context>{context}</context> <Question>{questions[i]}</Question> \n <GoldAnswer> {golden_responses[i]} </GoldAnswer> <ModelAnswer> {model_responses[i]} </ModelAnswer>")
    return res

def generate_training_text(question,golden_response):
   return f"<Question>{question}</Question><solution>{golden_response}</solution>" 
    

def evaluate_responses(evaluator, questions, responses, golden_responses):
    prompts = generate_evaluation_prompts(questions, responses, golden_responses)
    responses = []
    responses = evaluator(prompts)
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



def evaluate_model(df,model_id,save_path,batch_size):
    df["model_response"] = None
    df["evaluation_reason"] = None
    df['eval'] = None 

    evaluator = Gemini()
    model = LocalModel(model_id)

    num_rows = df.shape[0]
    print("Num problems: ", num_rows)
    num_batches = (num_rows + batch_size - 1) // batch_size 


    start_time = time.time()
    for i in range(0, num_rows, batch_size):
        try:
            j = min(i + batch_size, num_rows)  
            questions = df.iloc[i:j]["problem_statement"].to_list()
            golden_responses = df.iloc[i:j]["golden_answer"].to_list()
            question_prompts = generate_question_prompts(questions)
            answers = model(question_prompts)
            print("Model Response generated")
            evaluations = evaluate_responses(evaluator, questions, answers, golden_responses)
            print("Evalations generated")

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
    #
    #

def train_model(training_data,hugging_face_id):
    dataset = Dataset.from_pandas(training_data)

    tokenizer = AutoTokenizer.from_pretrained(hugging_face_id)

    # TODO: refactor with abovce code.



    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0  
    )


    model = AutoModelForCausalLM.from_pretrained(
        hugging_face_id,
        torch_dtype="auto", 
        device_map="auto",
        offload_folder="./offload",
        quantization_config=quant_config,
    )


    def tokenize_fn(row):
        text = generate_training_text(row["problem_statement"],row["golden_answer"])
        tokens = tokenizer(text)
        return tokens
    tokenized_data = dataset.map(tokenize_fn,batched=False)
    
    # should be handled differently
    
    training_args = TrainingArguments(
        output_dir="./sft_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        save_strategy="epoch",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        tokenizer=tokenizer
    )

    trainer.train()




# df = pd.read_parquet("hf://datasets/toloka/u-math/data/test-00000-of-00001.parquet")
# df = clean_dataset(df)
# hugging_face_id = "google/gemma-7b-it"
# evaluate_model(df,hugging_face_id,"gemma_7b_it.csv")
#

GEMMA_2B_ID = "google/gemma-2B-it"
GEMMA_7B_ID = "google/gemma-7B-it"

if __name__ == "__main__":
    df = pd.read_csv(CLP_PATH)
    train,test = train_test_split(df)
    
    print("Testing set size: ",len(test))
    print("Testing set column names:", list(test))

    #evaluate_model(test,GEMMA_7B_ID,"gemma_7b_it_clp.csv",batch_size=2)
    train_model(train,GEMMA_2B_ID)


# https://huggingface.co/blog/gemma-peft


# Super vised learnign with training.
# Epochs of entire data.
# Epoch of each category.

# should try fine tuning on one category and testng that single one first.
