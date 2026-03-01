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
        You will scrutinize proposed solutions provided by other agent systems or standalone LLMs.
        
        Your responsibilities:
        1. Understand the problem posed, the given correct solution, and the proposed solution presented to you.
        2. Compare the proposed solution to the correct solution.
        3. in the isAnswerCorrect field, indicate whether the final answer that the system arrived at is correct or not. 1 indicates that the proposed solution is correct, while 0 indicates that it is not correct.
        4. In the explanationRating field, provide a rating on a scale of 0-10 indicating the overall quality of the proposed solution, taking into account factors such as correctness, completeness, and clarity.
        5. In the comments field, note any exceptional strengths or weaknesses of the proposed solution is they come up. Otherwise leave it blank.
    """

    solutionCritic_llm_config = llm_config.copy()
    solutionCritic_llm_config["response_format"] = solutionCriticAgentOutput
    solutionCritic_llm_config["temperature"] = 0

    return ConversableAgent(
        name = name,
        system_message = systemMessage,
        llm_config = solutionCritic_llm_config,
        human_input_mode="NEVER"
    )