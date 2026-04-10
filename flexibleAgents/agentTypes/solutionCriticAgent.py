from typing import Dict, List
from autogen import ConversableAgent
from pydantic import BaseModel, Field

# Evaluation Agent --> takes in a proposed solution string for each problem and evaluates against actual problem solution
class solutionCriticAgentOutput(BaseModel):
    isAnswerCorrect: bool					        # Boolean indicating whether the proposed solution is correct or not
    explanationRating: int = Field(ge=0, le=10)		# Rating on a scale of 1-10 indicating the overall quality of the proposed solution
    comments: str						            # Rundown of fact-checked information, especially focusing on accuracy and reliability

def solutionCriticAgent(llm_config, name = "SolutionCriticAgent") -> ConversableAgent:
    systemMessage = f"""
        You are a SOLUTION CRITIC AGENT specializing in evaluating the accuracy and reliability of proposed solutions to scientific problems.
        You will scrutinize proposed solutions provided by other agent systems or standalone LLMs; in the case of an agentic group chat, you will receive a full message history.
        
        Your responsibilities:
        1. Understand the problem posed, the reference explanation (if given), and the proposed solution presented to you.
        2. Compare the proposed solution to the correct solution.
        3. in the isAnswerCorrect field, indicate whether the final answer that the system arrived at (usually a numerical or boolean value) is correct or not. 1 indicates that the proposed solution is correct, while 0 indicates that it is not correct.
            --> If the provided answer is equivalent to the correct answer (e.g. due to numerical precision errors, tiny deviations from the solution that may just be due to the use of differently many sigfigs in the calculation, or use of a different unit that can be easily converted), you can still mark it as correct (1) but note the discrepancy in the comments field.
        4. In the explanationRating field, provide a rating on a scale of 1-10 indicating the overall quality of the proposed solution, taking into account factors such as correctness, completeness, clarity, and closeness to the given correct explanation. If no reference explanation is provided, grade to the best of your ability based on the information given.
        5. In the comments field, note any exceptional strengths or weaknesses of the proposed solution is they come up. If any notable agent behaviour occurs, such as repeatedly incorrect statements by a certain agent type, mention it here.
    """

    description = """
        The SOLUTION CRITIC AGENT is responsible for evaluating the accuracy and reliability of proposed solutions to scientific problems.
    """

    solutionCritic_llm_config = llm_config.copy()
    solutionCritic_llm_config["response_format"] = solutionCriticAgentOutput
    solutionCritic_llm_config["temperature"] = 0

    return ConversableAgent(
        name = name,
        system_message = systemMessage,
        description = description,
        llm_config = solutionCritic_llm_config,
        human_input_mode="NEVER"
    )