from datasets import load_dataset, load_from_disk
import os

# Define function to be executed for each problem
def run_test(problem):
    print(f"Running test for problem: {problem}")
    # Here you can add code to run the test for the given problem
    # For example, you could call a function that takes the problem as input and returns the result
    # result = run_problem(problem)
    # print(f"Result: {result}")

# Load scibench (locall or from web)
if not os.path.exists(os.path.join(os.path.dirname(__file__), "problems")):
    problems = load_dataset("xw27/scibench")
    problems.save_to_disk(os.path.join(os.path.dirname(__file__), "problems"))
else:
    problems = load_from_disk(os.path.join(os.path.dirname(__file__), "problems"))

# Show dataset
problems.map(run_test)