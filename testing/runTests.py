import json
import os, shutil, sys, pickle
from collections import defaultdict

from sympy import content

# Add parent directory to path for imports as testing is in subdir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexibleAgents import agentChat
from autogen import LLMConfig, ConversableAgent
from llmconfig import local_llm_config, commercial_llm_config
from datasets import load_dataset, load_from_disk

from dotenv import load_dotenv
load_dotenv()



class Tester:
    def __init__(self):
        self.llmconfig = commercial_llm_config

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

    # Instantiate solvers (flexibleChat vs. basic agent (single-agent instance of flexibleChat)) and critic agent here to avoid re-instantiating for each problem
    def setupSolvers(self):
        self.flexibleChat = agentChat.flexibleAgentChat(
            configPath="flexibleAgents/agentConfigs/testingConfig.txt",
            llm_config=self.llmconfig,
            maxRounds=20,
            trackTokens=True
        )

        # Baseline "Agent" (backbone LLM only, single-agent config)
        self.basicChat = agentChat.flexibleAgentChat(
            configPath="flexibleAgents/agentConfigs/basicAgent.txt",
            llm_config=self.llmconfig,
            maxRounds=2,
            trackTokens=True
        )

        # Critic agent for solution evaluation
        self.criticAgentChat = agentChat.flexibleAgentChat(
            configPath="flexibleAgents/agentConfigs/solutionCritic.txt",
            llm_config=self.llmconfig,
            maxRounds=2
        )

    # Process each problem in the dataset with map()
    def runTests(self):
        self.problems.map(self.run_test)
        
    # Separate last msg content from group chat response
    def fetchLastMsgContent(self, messageList):
        if len(messageList) == 0:
            return ""
        lastMsg = messageList[-1]
        content = lastMsg.get("content", "")
        return content
    
    # Separate Query from message list (first message content by definition)
    def fetchQueryFromMsgList(self, messageList):
        if len(messageList) == 0:
            return ""
        firstMsg = messageList[0]
        content = firstMsg.get("content", "")
        return content

    # Save group chat responses to text file for easy reading and record keeping, especially for proposed solutions by the flexibleChat and basicChat agents. 
    def saveGroupChatResponseToFile(self, messageList, filepath):
        query = self.fetchQueryFromMsgList(messageList)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Conversation Log for query: {query}\n\n")
            for msg in messageList:
                name = msg.get("name", "unknown")
                content = msg.get("content", "")

                # Try to pretty-print JSON content when possible, otherwise write raw content
                formatted = None
                if isinstance(content, (dict, list)):
                    formatted = json.dumps(content, indent=4)
                else:
                    try:
                        parsed = json.loads(content)
                        formatted = json.dumps(parsed, indent=4)
                    except Exception:
                        formatted = str(content)

                # Write to file and also print so manager/summary messages are visible in the terminal
                f.write(f"{name}:\n{formatted}\n\n")

    def saveTokenUsageToFile(self, tokenUsage, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(tokenUsage, f)

    # Save critic responses to text normally, but also as pkl file
    # --> Done for record keeping, easy reading and potential qualitative analysis of critic comments.
    def saveCriticResponseToFile(self, critic_response, filepath):
        self.saveGroupChatResponseToFile(critic_response, filepath)

        # Infer pkl filename and save file
        filename_pkl = os.path.join(os.path.dirname(filepath), f"evaluation_{os.path.basename(filepath).split(".")[0]}.pkl")
        with open(filename_pkl, "wb") as f:
            pickle.dump(self.fetchLastMsgContent(critic_response), f)

    # Run agent system and critic evaluation, saving results as they go
    def runAndEvaluateProblem(self, agentChatInstance, problem, solution_file, evaluation_file, cost_file):
        # Run Agent system
        response, tokenUsage = agentChatInstance.startConversation(problem["problem_text"])
        
        # Save Output and token usage
        self.saveGroupChatResponseToFile(response, solution_file)
        self.saveTokenUsageToFile(tokenUsage, cost_file)

        # Run solution by the critic agent to evaluate against the correct solution
        criticEvaluation = self.criticAgentChat.startConversation(query=f"""
            Evaluate the following proposed solution to the given problem:
            Problem Description: {problem["problem_text"]}
            Reference Explanation: {problem["solution"]}
            Correct Final Answer: {problem["answer_number"]} {problem["unit"]}    
            Proposed Solution: {self.fetchLastMsgContent(response)}                                        
        """)

        # Save correctness results, critic agent ratings, and critic agent comments for each proposed solution in the problem directory
        self.saveCriticResponseToFile(criticEvaluation, evaluation_file)

    # Define function to be executed for each problem (needed for using MAP on the HF dataset)
    def run_test(self, problem):
        print(f"Running test for problem: {problem}\n\n")

        # Create a directory (if not existent yet) to save the problem and proposed solution(s)
        problemname = f"{problem["source"]}_{problem["problemid"]}".strip().replace(" ", "_").replace("__", "_")
        problem_dir = os.path.join(os.path.dirname(__file__), "problems", problemname)
        os.makedirs(problem_dir, exist_ok=True)

        print(f"Problem directory created at: {problem_dir}")

        # Filenames for problem description, saving proposed solutions, saving critic evaluations and token usage for both flexibleChat and basicChat agents
        problem_description_file = os.path.join(problem_dir, "problem_description.txt")

        flexibleChat_solution_file = os.path.join(problem_dir, f"solution_flexibleChat_{self.llmconfig["model"]}.txt")
        flexibleChat_evaluation_file = os.path.join(problem_dir, f"evaluation_flexibleChat_{self.llmconfig["model"]}.txt")
        flexibleChat_cost_file = os.path.join(problem_dir, f"cost_flexibleChat_{self.llmconfig["model"]}.pkl")
        
        basicChat_solution_file = os.path.join(problem_dir, f"solution_basicChat_{self.llmconfig["model"]}.txt")
        basicChat_evaluation_file = os.path.join(problem_dir, f"evaluation_basicChat_{self.llmconfig["model"]}.txt")
        basicChat_cost_file = os.path.join(problem_dir, f"cost_basicChat_{self.llmconfig["model"]}.pkl")

        # Save problem description and correct solution in the problem directory for reference
        if not os.path.exists(problem_description_file):
            with open(problem_description_file, "w", encoding="utf-8") as f:
                f.write(f"Problem Description:\n{problem["problem_text"]}\n\n")
                f.write("--------------------------------------------------------------------------------------\n\n")
                f.write(f"Reference Explanation:\n{problem["solution"]}\n\n")
                f.write("--------------------------------------------------------------------------------------\n\n")
                f.write(f"Correct Final Answer:\n{problem["answer_number"]} {problem["unit"]}\n")

        # Run problem through fully configured FlexibleAgents system (testing config) and evaluate
        if not os.path.exists(flexibleChat_solution_file) and not os.path.exists(flexibleChat_evaluation_file):
            self.runAndEvaluateProblem(self.flexibleChat, problem, flexibleChat_solution_file, flexibleChat_evaluation_file, flexibleChat_cost_file)

        # Run problem through basic agent system (backbone LLM only) and evaluate
        if not os.path.exists(basicChat_solution_file) and not os.path.exists(basicChat_evaluation_file):
            self.runAndEvaluateProblem(self.basicChat, problem, basicChat_solution_file, basicChat_evaluation_file, basicChat_cost_file)

if __name__ == "__main__":
    tester = Tester()
    tester.runTests()