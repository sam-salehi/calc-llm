evaluation_context = """You are an evaluator for university-level math answers.
You are given:
- Question
- Gold answer
- Model's answer

Check if the model's final answer matches the gold answer, leave some room for leeway as would be expected for a TA grader.
allowing equivalent symbolic forms.
Output only <Evaluation> Correct || Incorrect </Evaluation> and <Reasoning> {relevant reasoning for evaluation} </Reasoning> """


question_context = """You are solving a university-level mathematics problem from the UMath benchmark.
Use correct mathematical reasoning to reach the solution, but keep your explanation concise and self-contained.
Write your answer in no more than two short paragraphs, with the final answer clearly stated at the end in the format: 
Final Answer: <answer>."""
