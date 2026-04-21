from dataclasses import dataclass
import json
import os, shutil, sys, pickle, time
from collections import defaultdict
import multiprocessing as mp

# Add parent directory to path for imports as testing is in subdir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexibleAgents import agentChat
from autogen import LLMConfig, ConversableAgent
from llmconfig import *
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
	def __init__(self, llmconfig):
		self.llmconfig = llmconfig

		self.modelname = self.llmconfig["model"].replace("/", "_")

		self.numTested = 0
		self.numProblems = 0

	# Instantiate solvers (flexibleChat vs. basic agent (single-agent instance of flexibleChat)) and critic agent here to avoid re-instantiating for each problem
	def setupSolvers(self):
		print("Setting up solver: flexibleChat...")
		self.flexibleChat = agentChat.flexibleAgentChat(
			configPath="flexibleAgents/agentConfigs/testingConfig.txt",
			llm_config=self.llmconfig,
			maxRounds=20,
			trackTokens=True,
			resetAfterConversation=True
		)

		print("Setting up solver: basicChat...")
		# Baseline "Agent" (backbone LLM only, single-agent config)
		self.basicChat = agentChat.flexibleAgentChat(
			configPath="flexibleAgents/agentConfigs/basicAgent.txt",
			llm_config=self.llmconfig,
			maxRounds=2,
			trackTokens=True,
			resetAfterConversation=True
		)

		print("Setting up critic agent: solutionCritic...")
		# Critic agent for solution evaluation
		# The critic agent always uses the exact same GPT-4o-mini config for evaluation for consistency.
		self.criticAgentChat = agentChat.flexibleAgentChat(
			configPath="flexibleAgents/agentConfigs/solutionCritic.txt",
			llm_config=local_llm_config_4o_mini,
			maxRounds=2,
			trackTokens=True,
			resetAfterConversation=True
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

	def clearAndRebuildPathConfig(self, paths: PathConfig):
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

		os.makedirs(paths.solution_dir, exist_ok=True)
		os.makedirs(paths.evaluation_dir, exist_ok=True)



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

	def makePathConfig(self, solverName):
		return PathConfig(
			solution_dir = os.path.join(self.solutions_dir, f"{solverName}_{self.modelname}"),
			evaluation_dir = os.path.join(self.evaluations_dir, f"{solverName}_{self.modelname}"),
			evaluation_pkl = os.path.join(self.results_dir, f"evaluation_{solverName}_{self.modelname}.pkl"),
			solution_cost_pkl = os.path.join(self.results_dir, f"{solverName}_{self.modelname}_cost.pkl"),
			evaluation_cost_pkl = os.path.join(self.results_dir, f"evaluation_{solverName}_{self.modelname}_cost.pkl")
		)

	# This is a utility function to reset all results for a specific model in case of issues with the API or if you just want to re-run tests for a specific model.		
	def hardResetResultsForCurrentModel(self):
		for problem in os.listdir(os.path.join(os.path.dirname(__file__), "problems")):
			problem_dir = os.path.join(os.path.dirname(__file__), "problems", problem)
			if os.path.isdir(problem_dir):
				solutions_dir = os.path.join(problem_dir, "solutions")
				evaluations_dir = os.path.join(problem_dir, "evaluations")
				results_dir = os.path.join(problem_dir, "results")

				for item in os.listdir(solutions_dir):
					if self.modelname in item:
						shutil.rmtree(os.path.join(solutions_dir, item))
						print(f"Deleted {os.path.join(solutions_dir, item)}")
				for item in os.listdir(evaluations_dir):
					if self.modelname in item:
						shutil.rmtree(os.path.join(evaluations_dir, item))
						print(f"Deleted {os.path.join(evaluations_dir, item)}")
				for item in os.listdir(results_dir):
					if self.modelname in item:
						os.remove(os.path.join(results_dir, item))
						print(f"Deleted {os.path.join(results_dir, item)}")

	# Define function to be executed for each problem (needed for using MAP on the HF dataset)
	def run_test(self, problem):
		# Create a directory (if not existent yet) to save the problem and proposed solution(s)
		self.setupPathsForProblem(problem)

		print(f"Running test for problem {os.path.basename(os.path.normpath(self.problem_dir))} with model {self.modelname}...")

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
		flexibleChatPaths = self.makePathConfig("flexibleChat")
		
		# Run problem through fully configured FlexibleAgents system (testing config) and evaluate
		if not self.isTested(paths=flexibleChatPaths):
			self.clearAndRebuildPathConfig(flexibleChatPaths)
			self.runAndEvaluateProblem(self.flexibleChat, problem, flexibleChatPaths)
		else:
			print("Skipping test for flexibleChat as results already exist.")

		# Define output paths for basic agent (backbone LLM only, single-agent config) for comparison with flexible agent system
		basicChatPaths = self.makePathConfig("basicChat")
			
		# Run problem through basic agent system (backbone LLM only) and evaluate
		if not self.isTested(paths=basicChatPaths):
			self.clearAndRebuildPathConfig(basicChatPaths)
			self.runAndEvaluateProblem(self.basicChat, problem, basicChatPaths)
		else:
			print("Skipping test for basicChat as results already exist.")

		print("\n\n")

	# Run agent system and critic evaluation, saving results as they go
	def runAndEvaluateProblem(self, agentChatInstance, problem, paths: PathConfig):
		# Small prompt sanitization to remove some LaTex stuff. Testing to see if this  leads to less silent failures/stalls of the API.
		problemTextSanitized = problem["problem_text"].replace("$", "").replace("\\", "")

		# Set agent system output path for this run and save token usage
		agentChatInstance.setConversationPath(paths.solution_dir)
		response, tokenUsage = agentChatInstance.startConversation(problemTextSanitized)
		self.saveTokenUsageToFile(tokenUsage, paths.solution_cost_pkl)

		# Set eval dir and run solution by the critic agent to evaluate against reference solution
		self.criticAgentChat.setConversationPath(paths.evaluation_dir)
		criticEvaluation, criticTokenUsage = self.criticAgentChat.startConversation(query=f"""
			Evaluate the following proposed solution to the given problem:
			Problem Description: {problemTextSanitized}
			Reference Explanation: {problem["solution"]}
			Correct Final Answer: {problem["answer_number"]} {problem["unit"]}	
			Proposed Solution: {json.dumps(response)}										
		""")

		# Save correctness results, critic agent ratings, and critic agent comments for each proposed solution in the problem directory
		self.saveCriticResponseToFile(criticEvaluation, paths.evaluation_pkl)
		self.saveTokenUsageToFile(criticTokenUsage, paths.evaluation_cost_pkl)

	# Purely count which tests have already been completed
	# This might even work with just map() but there were some issues with hashing the function.
	def checkTestProgress(self, problem):
		self.setupPathsForProblem(problem)

		flexibleChatPaths = self.makePathConfig("flexibleChat")
		basicChatPaths = self.makePathConfig("basicChat")

		self.numProblems += 1
		if self.isTested(flexibleChatPaths) and self.isTested(basicChatPaths):
			self.numTested += 1


# Global definition necessary for variable access in subprocesses
tester = None

# Top-level function is required for multiprocessing with map() on the HuggingFace dataset.
# Also, each process will create a separate Tester instance to avoid issues with shared state and potential concurrency problems.
def testPassthrough(example, llmconfig=None):
	global tester
	if tester is None:
		tester = Tester(llmconfig)
		tester.setupSolvers()
	tester.run_test(example)

def checkTestProgress(example, llmconfig=None):
	global tester
	if tester is None:
		tester = Tester(llmconfig)
	return tester.checkTestProgress(example)

if __name__ == "__main__":
	# Load scibench (local or from web)
	if not os.path.exists(os.path.join(os.path.dirname(__file__), "dataset")):
		problems = load_dataset("xw27/scibench")
		problems.save_to_disk(os.path.join(os.path.dirname(__file__), "dataset"))
	else:
		problems = load_from_disk(os.path.join(os.path.dirname(__file__), "dataset"))

	# Reset (used if debugging model-specific issues)
	# configsToReset = [local_llm_config_gemma, local_llm_config_4_1_mini, local_llm_config_4_1_nano, local_llm_config_codestral, local_llm_config_mistral_small]
	# for config in configsToReset:
	# 	tester = Tester(config)
	# 	tester.hardResetResultsForCurrentModel()
	# quit()

	# Check completion and reset tester afterwards
	for llmconfig in [local_llm_config_4o_mini, local_llm_config_5_nano, commercial_llm_config_4_1_mini]:
		# Reset tester globally! This is needed for multiprocessing to work properly as each thread needs an own tester instance.
		tester = None
		problems.map(
			checkTestProgress,
			desc = "Checking test progress...",
			fn_kwargs={"llmconfig": llmconfig}
		)
		print(f"Progress: {tester.numTested}/{tester.numProblems} problems fully tested with model {tester.modelname}.")
		
		if tester.numTested == tester.numProblems:
			print("All tests are completed.")
			continue
		
		# Reset tester again before running tests to avoid any potential issues with shared state.
		tester = None
		# Process each problem in the dataset with map().
		# Optionally use multiple processes at the same time for speedup, but this will clutter the output and occasionally screw things up.
		# This while loop is just here to continually retry in the case or errors - e.g. IZ API (or mistral) rate limiting the fuck out of me cuz damn yall stingy.
		while True:
			try:
				problems.map(
					testPassthrough,
					desc = "Running tests on SciBench...",
					fn_kwargs={"llmconfig": llmconfig}
				)
				break
			except Exception as e:
				print(f"An error occurred: {e}. Retrying...")
				time.sleep(5)