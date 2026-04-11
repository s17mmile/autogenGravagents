from dataclasses import dataclass
import json
import os, shutil, sys, pickle, time
from collections import defaultdict
import multiprocessing as mp

# Add parent directory to path for imports as testing is in subdir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexibleAgents import agentChat
from autogen import LLMConfig, ConversableAgent
from llmconfig import local_llm_config, commercial_llm_config_4o_mini, commercial_llm_config_5_nano, commercial_llm_config_5_4_nano
from datasets import load_dataset, load_from_disk

from dotenv import load_dotenv
load_dotenv()

# UTF-8 Encoding
sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class PathConfig:
	solution_dir: str
	evaluation_dir: str
	evaluation_pkl: str
	
	solution_cost_pkl: str
	evaluation_cost_pkl: str

class Tester:
	def __init__(self):
		self.llmconfig = commercial_llm_config_5_nano
		self.model = self.llmconfig["model"]

		self.numTested = 0
		self.numProblems = 0

	def prepareForTest(self):
		print("Initializing Tester instance...")
		self.folderSetup()
		print("Folder setup complete.")
		self.setupSolvers()
		print("Solvers set up successfully.")

	def folderSetup(self):
		# Create problems folder
		self.problem_dir = os.path.join(os.path.dirname(__file__), "problems")
		os.makedirs(self.problem_dir, exist_ok=True)

	# Instantiate solvers (flexibleChat vs. basic agent (single-agent instance of flexibleChat)) and critic agent here to avoid re-instantiating for each problem
	def setupSolvers(self):
		print("Setting up solver: flexibleChat...")
		self.flexibleChat = agentChat.flexibleAgentChat(
			configPath="flexibleAgents/agentConfigs/testingConfig.txt",
			llm_config=self.llmconfig,
			maxRounds=20,
			trackTokens=True
		)

		print("Setting up solver: basicChat...")
		# Baseline "Agent" (backbone LLM only, single-agent config)
		self.basicChat = agentChat.flexibleAgentChat(
			configPath="flexibleAgents/agentConfigs/basicAgent.txt",
			llm_config=self.llmconfig,
			maxRounds=2,
			trackTokens=True
		)

		print("Setting up critic agent: solutionCritic...")
		# Critic agent for solution evaluation
		self.criticAgentChat = agentChat.flexibleAgentChat(
			configPath="flexibleAgents/agentConfigs/solutionCritic.txt",
			llm_config=self.llmconfig,
			maxRounds=2,
			trackTokens=True
		)
		
		return

	# Separate last msg content from group chat response
	def fetchLastMsgContent(self, messageList):
		if len(messageList) == 0:
			return ""
		lastMsg = messageList[-1]
		content = lastMsg.get("content", "")
		return content

	def saveTokenUsageToFile(self, tokenUsage, filepath):
		with open(filepath, "wb") as f:
			pickle.dump(tokenUsage, f)

	# Save critic responses to text normally, but also as pkl file
	# --> Done for record keeping, easy reading and potential qualitative analysis of critic comments.
	def saveCriticResponseToFile(self, critic_response, eval_pkl):
		with open(eval_pkl, "wb") as f:
			pickle.dump(self.fetchLastMsgContent(critic_response), f)
		return

	def isTested(self, paths: PathConfig):
		return (
			os.path.exists(paths.solution_dir) 
			and os.path.exists(paths.evaluation_dir)
			and os.path.exists(paths.evaluation_pkl)
			and os.path.exists(paths.solution_cost_pkl)
			and os.path.exists(paths.evaluation_cost_pkl)
		)

	def clearPathConfig(self, paths: PathConfig):
		if os.path.exists(paths.solution_dir):
			shutil.rmtree(paths.solution_dir)
		if os.path.exists(paths.evaluation_dir):
			shutil.rmtree(paths.evaluation_dir)
		if os.path.exists(paths.evaluation_pkl):
			os.remove(paths.evaluation_pkl)
		if os.path.exists(paths.solution_cost_pkl):
			os.remove(paths.solution_cost_pkl)
		if os.path.exists(paths.evaluation_cost_pkl):
			os.remove(paths.evaluation_cost_pkl)

	# Run agent system and critic evaluation, saving results as they go
	def runAndEvaluateProblem(self, agentChatInstance, problem, paths: PathConfig):
		# Set agent system output path for this run and save token usage
		agentChatInstance.setConversationPath(paths.solution_dir)
		response, tokenUsage = agentChatInstance.startConversation(problem["problem_text"])
		self.saveTokenUsageToFile(tokenUsage, paths.solution_cost_pkl)

		# Set eval dir and run solution by the critic agent to evaluate against reference solution
		self.criticAgentChat.setConversationPath(paths.evaluation_dir)
		criticEvaluation, criticTokenUsage = self.criticAgentChat.startConversation(query=f"""
			Evaluate the following proposed solution to the given problem:
			Problem Description: {problem["problem_text"]}
			Reference Explanation: {problem["solution"]}
			Correct Final Answer: {problem["answer_number"]} {problem["unit"]}	
			Proposed Solution: {json.dumps(response)}										
		""")

		# Save correctness results, critic agent ratings, and critic agent comments for each proposed solution in the problem directory
		self.saveCriticResponseToFile(criticEvaluation, paths.evaluation_pkl)
		self.saveTokenUsageToFile(criticTokenUsage, paths.evaluation_cost_pkl)

	def setupPathsForProblem(self, problem):
		self.problemname = f"{problem["source"]}_{problem["problemid"]}".strip().replace(" ", "_").replace("__", "_")
		self.problem_dir = os.path.join(os.path.dirname(__file__), "problems", self.problemname)
		self.solutions_dir = os.path.join(self.problem_dir, "solutions")
		self.evaluations_dir = os.path.join(self.problem_dir, "evaluations")
		self.results_dir = os.path.join(self.problem_dir, "results")
		
		os.makedirs(self.problem_dir, exist_ok=True)
		os.makedirs(self.solutions_dir, exist_ok=True)
		os.makedirs(self.evaluations_dir, exist_ok=True)
		os.makedirs(self.results_dir, exist_ok=True)

	def makePathConfigForProblem(self, solverName):
		return PathConfig(
			solution_dir = os.path.join(self.solutions_dir, f"{solverName}_{self.model}"),
			evaluation_dir = os.path.join(self.evaluations_dir, f"{solverName}_{self.model}"),
			evaluation_pkl = os.path.join(self.results_dir, f"evaluation_{solverName}_{self.model}.pkl"),
			solution_cost_pkl = os.path.join(self.results_dir, f"{solverName}_{self.model}_cost.pkl"),
			evaluation_cost_pkl = os.path.join(self.results_dir, f"evaluation_{solverName}_{self.model}_cost.pkl")
		)

	def printProgress(self):
		print(f"Progress: {self.numTested}/{self.numProblems} problems fully tested with model {self.model}.")

	# Define function to be executed for each problem (needed for using MAP on the HF dataset)
	def run_test(self, problem):
		# Create a directory (if not existent yet) to save the problem and proposed solution(s)
		self.setupPathsForProblem(problem)

		print(f"Running test for problem {os.path.basename(os.path.normpath(self.problem_dir))} with model {self.model}...")

		# Filenames for problem description, saving proposed solutions, saving critic evaluations and token usage for both flexibleChat and basicChat agents
		self.problem_description_filepath = os.path.join(self.problem_dir, "problem_description.txt")

		# Save problem description and correct solution in the problem directory for reference
		if not os.path.exists(self.problem_description_filepath):
			with open(self.problem_description_filepath, "w", encoding="utf-8") as f:
				f.write(f"Problem Description:\n{problem["problem_text"]}\n\n")
				f.write("--------------------------------------------------------------------------------------\n\n")
				f.write(f"Reference Explanation:\n{problem["solution"]}\n\n")
				f.write("--------------------------------------------------------------------------------------\n\n")
				f.write(f"Correct Final Answer:\n{problem["answer_number"]} {problem["unit"]}\n")

		# Define output paths for flexible agent system
		flexibleChatPaths = self.makePathConfigForProblem("flexibleChat")
		
		# Run problem through fully configured FlexibleAgents system (testing config) and evaluate
		if not self.isTested(paths=flexibleChatPaths):
			self.clearPathConfig(flexibleChatPaths)
			self.runAndEvaluateProblem(self.flexibleChat, problem, flexibleChatPaths)
		else:
			print("Skipping test for flexibleChat as results already exist.")

		# Define output paths for basic agent (backbone LLM only, single-agent config) for comparison with flexible agent system
		basicChatPaths = self.makePathConfigForProblem("basicChat")
			
		# Run problem through basic agent system (backbone LLM only) and evaluate
		if not self.isTested(paths=basicChatPaths):
			self.clearPathConfig(basicChatPaths)
			self.runAndEvaluateProblem(self.basicChat, problem, basicChatPaths)
		else:
			print("Skipping test for basicChat as results already exist.")

		print("\n\n")

	# Purely count which tests have already been completed
	# This might even work with just map() but there were some issues with hashing the function.
	def checkTestProgress(self, problem):
		self.setupPathsForProblem(problem)

		flexibleChatPaths = self.makePathConfigForProblem("flexibleChat")
		basicChatPaths = self.makePathConfigForProblem("basicChat")

		self.numProblems += 1
		if self.isTested(flexibleChatPaths) and self.isTested(basicChatPaths):
			self.numTested += 1


# Global definition necessary for variable access in subprocesses
tester = None

# Top-level function is required for multiprocessing with map() on the HuggingFace dataset.
# Also, each process will create a separate Tester instance to avoid issues with shared state and potential concurrency problems.
def testPassthrough(example):
	global tester
	if tester is None:
		tester = Tester()
		tester.prepareForTest()
	tester.run_test(example)

def checkTestProgress(example):
	global tester
	if tester is None:
		tester = Tester()
	return tester.checkTestProgress(example)

if __name__ == "__main__":
	# Load scibench (local or from web)
	if not os.path.exists(os.path.join(os.path.dirname(__file__), "dataset")):
		problems = load_dataset("xw27/scibench")
		problems.save_to_disk(os.path.join(os.path.dirname(__file__), "dataset"))
	else:
		problems = load_from_disk(os.path.join(os.path.dirname(__file__), "dataset"))

	# Check completion and reset tester afterwards
	problems.map(
		checkTestProgress,
		desc = "Checking test progress..."
	)
	print(f"Progress: {tester.numTested}/{tester.numProblems} problems fully tested with model {tester.model}.")
	tester = None

	# Process each problem in the dataset with map(). Uses multiple processes at the same time for speedup - this will clutter the output.
	problems.map(
		testPassthrough,
		num_proc=6,
		desc = "Running tests on SciBench..."
	)

	# FUCK RIGHT OFF WHY DO THESE TESTS JUST KEEP STALLING FUCK OFFGFGFFFFF