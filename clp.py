import os
from random import randint
import pandas as pd
from TexSoup import TexSoup


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

            questions_with_solutions.append(
                {"question": str(node), "solution": solution_text}
            )
        i += 1

    return questions_with_solutions


rows = []
books = ["CLP3", "CLP2", "CLP1", "CLP4"]
for book in books:
    abs_path = os.path.join(os.getcwd(), book, "latex", "problembook", "problems")
    for section in os.listdir(abs_path):
        print("Extracting: ", book + ": " + section)
        section_path = abs_path + "/" + section
        try:
            qs = extract_question_with_solutions(section_path)
        except Exception as e:
            print("Failed to extract: ", section_path)
            print(e)

        for q in qs:
            rows.append(
                {
                    "file": section,
                    "question": q["question"],
                    "solution": q["solution"],
                    "category": book,
                }
            )

df = pd.DataFrame(rows, columns=["file", "question", "solution", "category"])


def get_random_question():
    return df.loc[randint(0, len(df)), "question"]


q = get_random_question()


df.to_csv("CLP.csv", index=False)

