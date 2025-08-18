
# preprocessing of clp textbook. 
import pandas as pd 

def score(df):
    count = (df["eval"] == True).sum()
    return count

def format_cell(row):
    return (
        r"$$\begin{aligned}"
        rf"\textbf{{Q:}} &\ {row['problem_statement']} \\"
        rf"\textbf{{A:}} &\ {row['model_response']} \\"
        rf"\textbf{{Eval:}} &\ {row['eval']} "
        r"\end{aligned}$$"
    )

def log_random_response(df):
    raise NotImplementedError()



df = pd.read_csv("gemma_7b_it_clp.csv")
score = score(df)

print(score)
print(len(df))
