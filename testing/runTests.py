import os, shutil, sys, pickle

# Add parent directory to path for imports as testing is in subdir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexibleAgents import agentChat
from autogen import LLMConfig, ConversableAgent
from llmconfig import local_llm_config, commercial_llm_config
from criticAgent import solutionCriticAgent
from datasets import load_dataset, load_from_disk

from dotenv import load_dotenv
load_dotenv()



class Tester:
    def __init__(self):
        self.llmconfig = commercial_llm_config
        self.resultsPath = os.path.join(os.path.dirname(__file__), "scores.json")

        resultformat = {
            "correctness": [],
            "scores": [],
            "critic_comments": [],
        }

        self.results = {
            "flexibleChat": resultformat.copy(),
            "basicAgent": resultformat.copy()
        }

        print("Initializing Tester instance...")
        self.folderSetup()
        print("Folder setup complete.")
        self.loadProblems()
        print("Problems loaded successfully.")
        self.setupSolvers()
        print("Solvers set up successfully.")


    def folderSetup(self):
        # Create problems folder
        self.problem_dir = os.path.join(os.path.dirname(__file__), "problems")
        os.makedirs(self.problem_dir, exist_ok=True)

    def loadProblems(self):
        # Load scibench (local or from web)
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "dataset")):
            self.problems = load_dataset("xw27/scibench")
            self.problems.save_to_disk(os.path.join(os.path.dirname(__file__), "dataset"))
        else:
            self.problems = load_from_disk(os.path.join(os.path.dirname(__file__), "dataset"))

    # Instantiate solvers (flexibleChat vs. basic GPT-4o agent vs. cmbagent (?)) and critic agent here to avoid re-instantiating for each problem
    def setupSolvers(self):
        self.flexibleChat = agentChat.flexibleAgentChat(
            configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
            llm_config=self.llmconfig,
            maxRounds=20
        )

        # Baseline "Agent" (backbone LLM)
        self.basicAgent = ConversableAgent(
            name = "BasicAgent",
            llm_config = self.llmconfig,
            human_input_mode="NEVER"
        )

        # Critic agent for solution evaluation
        self.criticAgent = solutionCriticAgent(
            name = "CriticAgent",
            llm_config=self.llmconfig,
        )

    # Process each problem in the dataset with map()
    def runTests(self):
        self.problems.map(self.run_test)

    # Save critic responses to file for record keeping and potential qualitative analysis of critic comments, in addition to saving the key results (correctness and score) to a dictionary for easy visualization and comparison across problems and solvers.
    def saveCriticResponseToFile(self, critic_response, solver_name, problem_dir):
        with open(os.path.join(problem_dir, f"critic_evaluation_{solver_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Correctness (1=correct, 0=incorrect): {critic_response.isAnswerCorrect}\n")
            f.write(f"Explanation Rating (1-10): {critic_response.explanationRating}\n")
            f.write(f"Critic Comments:\n{critic_response.comments}\n")

    # Separately save them to a dictionary for easy visualization and comparison across problems and solvers.
    def saveCriticResponsesToDict(self, critic_responses, solver_name):
        for critic_response in critic_responses.items():
            self.results[solver_name]["correctness"].append(critic_response.isAnswerCorrect)
            self.results[solver_name]["scores"].append(critic_response.explanationRating)
            self.results[solver_name]["critic_comments"].append(critic_response.comments)

    # Define function to be executed for each problem (needed for using MAP on the HF dataset)
    def run_test(self, problem):
        print(f"Running test for problem: {problem}\n\n")

        # Create a directory (if not existent yet) to save the problem and proposed solution(s)
        problemname = f"{problem['source']}_{problem['problemid']}".strip().replace(' ', '_').replace('__', '_')
        problem_dir = os.path.join(os.path.dirname(__file__), "evaluation", problemname)
        os.makedirs(problem_dir, exist_ok=True)

        # Save problem description and correct solution in the problem directory for reference
        if not os.path.exists(os.path.join(problem_dir, "problem_description.txt")):
            with open(os.path.join(problem_dir, "problem_description.txt"), "w", encoding="utf-8") as f:
                f.write(f"Problem Description:\n{problem['problem_text']}\n\n")
                f.write("--------------------------------------------------------------------------------------\n\n")
                f.write(f"Reference Explanation:\n{problem['solution']}\n\n")
                f.write("--------------------------------------------------------------------------------------\n\n")
                f.write(f"Final Answer:\n{problem['answer_number']} {problem['unit']}\n")

        # Run problem through FlexibleAgents system (default config)
        flexibleChatResponse = self.flexibleChat.startConversation(problem['problem_text'])
        print(f"FlexibleAgents proposed solution:\n{flexibleChatResponse}\n\n")

        # Run problem through basic GPT-4o Agent (only backbone!)
        basicAgentResponse = self.basicAgent.run(problem['problem_text'], max_turns=1, human_input_mode="NEVER")
        print(f"BasicAgent proposed solution:\n{basicAgentResponse}\n\n")

        quit()

        # Run each Solution by the critic agent to evaluate against the correct solution
        # TODO Check messages format: what part of the conversation history do we want to pass to the critic agent for evaluation?
        # TODO

        # Save correctness results, critic agent ratings, and critic agent comments for each proposed solution in the problem directory
        self.saveCriticResponsesToFile(flexibleChatResponse, "flexibleChat")
        self.saveCriticResponsesToFile(basicAgentResponse, "basicAgent")

        # Save correctness and score into dictionary for easy visualization of results
        self.saveCriticResponsesToDict(flexibleChatResponse, "flexibleChat")
        self.saveCriticResponsesToDict(basicAgentResponse, "basicAgent")


if __name__ == "__main__":
    tester = Tester()
    tester.runTests()