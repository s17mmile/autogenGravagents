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
		
	# Define function to be executed for each problem (needed for using MAP on the HF dataset)
	def run_test(self, problem):
		print(f"Running test for problem: {problem}\n\n")

		# Create a directory (if not existent yet) to save the problem and proposed solution(s)
		problemname = f"{problem["source"]}_{problem["problemid"]}".strip().replace(" ", "_").replace("__", "_")
		problem_dir = os.path.join(os.path.dirname(__file__), "problems", problemname)
		solutions_dir = os.path.join(problem_dir, "solutions")
		evaluations_dir = os.path.join(problem_dir, "evaluations")
		results_dir = os.path.join(problem_dir, "results")
		
		os.makedirs(problem_dir, exist_ok=True)
		os.makedirs(solutions_dir, exist_ok=True)
		os.makedirs(evaluations_dir, exist_ok=True)
		os.makedirs(results_dir, exist_ok=True)

		# Filenames for problem description, saving proposed solutions, saving critic evaluations and token usage for both flexibleChat and basicChat agents
		problem_description_filepath = os.path.join(problem_dir, "problem_description.txt")

		# Save problem description and correct solution in the problem directory for reference
		if not os.path.exists(problem_description_filepath):
			with open(problem_description_filepath, "w", encoding="utf-8") as f:
				f.write(f"Problem Description:\n{problem["problem_text"]}\n\n")
				f.write("--------------------------------------------------------------------------------------\n\n")
				f.write(f"Reference Explanation:\n{problem["solution"]}\n\n")
				f.write("--------------------------------------------------------------------------------------\n\n")
				f.write(f"Correct Final Answer:\n{problem["answer_number"]} {problem["unit"]}\n")



		# Define output paths for flexible agent system
		flexibleChatPaths = PathConfig(
			solution_dir = os.path.join(solutions_dir, f"flexibleChat_{self.model}"),
			evaluation_dir = os.path.join(evaluations_dir, f"flexibleChat_{self.model}"),
			evaluation_pkl = os.path.join(results_dir, f"evaluation_flexibleChat_{self.model}.pkl"),
			solution_cost_pkl = os.path.join(results_dir, f"flexibleChat_{self.model}_cost.pkl"),
			evaluation_cost_pkl = os.path.join(results_dir, f"evaluation_flexibleChat_{self.model}_cost.pkl")
		)
		
		# Run problem through fully configured FlexibleAgents system (testing config) and evaluate
		if not self.isTested(paths=flexibleChatPaths):
			self.clearPathConfig(flexibleChatPaths)
			self.runAndEvaluateProblem(self.flexibleChat, problem, flexibleChatPaths)



		# Define output paths for basic agent (backbone LLM only, single-agent config) for comparison with flexible agent system
		basicChatPaths = PathConfig(
			solution_dir = os.path.join(solutions_dir, f"basicChat_{self.model}"),
			evaluation_dir = os.path.join(evaluations_dir, f"basicChat_{self.model}"),
			evaluation_pkl = os.path.join(results_dir, f"evaluation_basicChat_{self.model}.pkl"),
			solution_cost_pkl = os.path.join(results_dir, f"basicChat_{self.model}_cost.pkl"),
			evaluation_cost_pkl = os.path.join(results_dir, f"evaluation_basicChat_{self.model}_cost.pkl")
		)
			
		# Run problem through basic agent system (backbone LLM only) and evaluate
		if not self.isTested(paths=basicChatPaths):
			self.clearPathConfig(basicChatPaths)
			self.runAndEvaluateProblem(self.basicChat, problem, basicChatPaths)



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

if __name__ == "__main__":
	# Load scibench (local or from web)
	if not os.path.exists(os.path.join(os.path.dirname(__file__), "dataset")):
		problems = load_dataset("xw27/scibench")
		problems.save_to_disk(os.path.join(os.path.dirname(__file__), "dataset"))
	else:
		problems = load_from_disk(os.path.join(os.path.dirname(__file__), "dataset"))

	# Process each problem in the dataset with map(). Uses multiple processes at the same time for speedup - this will clutter the output.
	num_processes = 12
	print(f"Running tests on SciBench dataset with {num_processes} parallel processes...")
	problems.map(
		testPassthrough,
		num_proc=num_processes,
		desc = "Running tests on SciBench..."
	)