from datasets import load_dataset, load_from_disk
import os, shutil

# TODO load proper dotenv and import necessary LLM systems

# Define function to be executed for each problem
def run_test(problem):
    # print(f"Running test for problem: {problem}\n\n")

    # Create a directory to save the problem and proposed solution(s)
    problemname = f"{problem['source']}_{problem['problemid']}".strip().replace(' ', '_').replace('__', '_')
    problem_dir = os.path.join(os.path.dirname(__file__), "evaluation", problemname)
    os.makedirs(problem_dir, exist_ok=True)

    # Save problem description and correct solution in the problem directory for reference
    with open(os.path.join(problem_dir, "problem_description.txt"), "w", encoding="utf-8") as f:
        f.write(f"Problem Description:\n{problem['problem_text']}\n\n")
        f.write("--------------------------------------------------------------------------------------\n\n")
        f.write(f"Reference Explanation:\n{problem['solution']}\n\n")
        f.write("--------------------------------------------------------------------------------------\n\n")
        f.write(f"Final Answer:\n{problem['answer_number']} {problem['unit']}\n")

    # Run problem through FlexibleAgents system (default config)
    # TODO

    # Run problem through basic GPT-4o Agent (only backbone!)
    # TODO

    # Perplexity excluded as I do not have API access - some problems are manually tested separately :()

    # Run each Solution through the critic agent to evaluate against the correct solution
    # TODO

    # Save correctness results, critic agent ratings, and critic agent comments for each proposed solution in the problem directory
    # TODO



# Clear evaluation folder if it exists, then create a new one
evaluation_dir = os.path.join(os.path.dirname(__file__), "evaluation")
if os.path.exists(evaluation_dir):
    shutil.rmtree(evaluation_dir)
os.makedirs(evaluation_dir, exist_ok=True)

# Load scibench (locall or from web)
if not os.path.exists(os.path.join(os.path.dirname(__file__), "problems")):
    problems = load_dataset("xw27/scibench")
    problems.save_to_disk(os.path.join(os.path.dirname(__file__), "problems"))
else:
    problems = load_from_disk(os.path.join(os.path.dirname(__file__), "problems"))

# Process each problem in the dataset, placing the result in the evaluation folder
problems.map(run_test)