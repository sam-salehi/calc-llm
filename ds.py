import pandas as pd
import time
from google import genai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from google import genai
import requests
from prompts import evaluation_context, question_context
import re


df = pd.read_parquet("hf://datasets/toloka/u-math/data/test-00000-of-00001.parquet")
num_rows = df.shape[0]


class Model:
    def __init__(self, model_name="tiiuae/falcon-rw-1b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            offload_folder="/content/offload",
            offload_state_dict=True,
        )

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=300, do_sample=False)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text


class Gemini:
    def __init__(self, api_key, model="gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def __call__(self, prompt, max_new_tokens=256):
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return resp.text


gemini_key = "AIzaSyBRfwiNmkhOlwcX94E5eQ96xMLF5asdH8s"  # TODO: remove to env file
# model = Gemini(gemini_key)
# evaluator = Gemini(gemini_key, "gemini-1.5-pro")
#


def generate_question_prompt(question):
    context = question_context
    res = f"<context>  {context} </context> Answer the following: <Question> {question} </Question>"
    return res


def generate_response_prompt(question, model_response, golden_response):
    context = evaluation_context
    res = f"<context>{context}</context> <Question>{question}</Question> \n <GoldAnswer> {golden_response} </GoldAnswer> <ModelAnswer> {model_response} </ModelAnswer>"
    return res


def evaluate_response(evaluator, question, response, golden_response):
    prompt = generate_response_prompt(question, response, golden_response)
    response = evaluator(prompt)
    return response


def extract_eval(text):
    pattern = r"<Evaluation>(.*?)</Evaluation>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()  # return content inside tags.
    return None


def extract_reasoning(text):
    pattern = r"<Reasoning>(.*?)</Reasoning>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


df["model_response"] = None
df["evaluation_reason"] = None
df["eval"] = None

model = Gemini(api_key=gemini_key)
evaluator = model

for i in range(num_rows):
    try:
        question = df.loc[i, "problem_statement"]
        golden_response = df.loc[i, "golden_answer"]

        question_prompt = generate_question_prompt(question)
        answer = model(question_prompt)

        evaluation = evaluate_response(evaluator, question, answer, golden_response)

        eval = extract_eval(evaluation)
        reasoning = extract_reasoning(evaluation)

        df.loc[i, "model_response"] = answer
        df.loc[i, "evaluaton_reason"] = reasoning
        df.loc[i, "eval"] = eval == "Correct"

        print(f"{i + 1}/{num_rows} response was {eval}.")
    except Exception as e:
        print(e)
        break


df.to_csv("gemini_val.csv", index=False)
