import os
from random import randint
import pandas as pd
from TexSoup import TexSoup
import re

CLP_PATH = "CLP.csv"

def has_no_image(text):
    # ensures the question is just text based and doesn't include images.  
    return  not ("includegraphics" in text)
def extract_question_with_solutions(file):
    with open(file, "r") as f:
        soup = TexSoup(f.read())

    questions_with_solutions = []
    nodes = list(soup.contents)

    i = 0
    while i < len(nodes):
        node = nodes[i]
        if getattr(node, "name", None) == "question":
            # Search ahead for the next solution before the next question
            solution_text = None
            for j in range(i + 1, len(nodes)):
                next_node = nodes[j]
                if getattr(next_node, "name", None) == "solution":
                    solution_text = str(next_node)
                    break
                if getattr(next_node, "name", None) == "question":
                    break  # no solution found before the next question
            if has_no_image(str(node)) and has_no_image(solution_text):
                questions_with_solutions.append(
                    {"question": str(node), "solution": solution_text}
                )
        i += 1

    return questions_with_solutions


books = ["CLP3", "CLP2", "CLP1", "CLP4"]
categories = {
    "CLP1": ["Limits","Derivatives","Applications of Derivatives", "Integral Calculus"],
    "CLP2": ["Integration","Applications of Integration", "Sequences and Series"],
    "CLP3": ["Geometry in two and three dimensions", "Partial Derivatves", "Multiple Integrals"],
    "CLP4": ["Curves","Vector Fields", "Surface Integrals", "Integral Theorems"]
}
# ordered_categories = [cat for cat in book for book in categories.values()]

def create_clp_dataset():
    rows = []

    section = 0 

    for book in books:
        abs_path = os.path.join(os.getcwd(), book, "latex", "problembook", "problems")
        for section in os.listdir(abs_path):
            print(f"Extraction {book} section {section}")
            match = re.search(r"_s(.)", section) 
            if not match.group(1) or not match.group(1).isdigit():
                continue # file is related to TrueFalse ,etc. 
            chapter_num =int(match.group(1)) - 1 
            chapter_idx = section 
            section_path = abs_path + "/" + section
            try:
                qs = extract_question_with_solutions(section_path)
            except Exception as e:
                print("Failed to extract: ", section_path)
                # print("With error: ",e)
            for q in qs:
                rows.append(
                    {
                        "file": section,
                        "problem_statement": q["question"],
                        "golden_answer": q["solution"],
                        "category": categories[book][chapter_num],
                        "book": book 
                    }
                )

    df = pd.DataFrame(rows, columns=["file", "problem_statement", "golden_answer", "category","book"])
    df.to_csv(CLP_PATH,index=False)


def train_test_split(df,shuffle=False,frac=0.9):
    train_list = []
    test_list = []
    for cat, group in df.groupby("category"):
        train = group.sample(frac=frac,random_state=42)
        test = group.drop(train.index)
        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    if shuffle:
        train_df = train_df.sample(frac=1,random_state=42).reset_index(drop=True)

    return train_df, test_df 


def get_random_question():
    return df.loc[randint(0, len(df)), "question"]


if __name__ == "__main__":
    create_clp_dataset()
