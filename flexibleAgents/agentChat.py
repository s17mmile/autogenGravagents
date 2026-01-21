import autogen
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern

from dataclasses import dataclass
from typing import List, Dict

import os

# Dynamic import of agent types from agentTypes dir
# We want to exclusively import the creation functions for each agent type
# The creation function must be named the same as the agent type itself
# I'm aware this is not clean, but it's the best I can currently think of for easy introsuction of new agent types
# Not enitrely sure why importing the agentTypes directory as a whole does not make it possible to access the individual module functions
for file in os.listdir(os.path.dirname(__file__) + "/agentTypes"):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        try:
            exec(f"from flexibleAgents.agentTypes.{module_name} import {module_name}")
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}. Skipping this module.")



# Some useful custom types for flexibility of chat instance
@dataclass
class agentSpecification:
    agentType: str
    name: str

# While this structure is not strictly necessary, it might make it easier to make a proper config editor later on
# Holds all agents and their possible transitions in a concise format
@dataclass
class chatGraph:
    agentSpecs: List[agentSpecification]
    transitions: Dict[str, List[str]]

# Main class that allows flexible agent conversations based on config files
class flexibleAgentChat:
    def __init__(self, configPath: str, llm_config, humanInTheLoop: bool = True, maxRounds: int = 10):
        self.llm_config = llm_config
        self.maxRounds = maxRounds
        self.humanInTheLoop = humanInTheLoop

        self.conversationGraph = self.parse_agent_config(configPath)

        self.instantiateAgents()

        # Create Autogen AutoPattern based on agent config
        # The restriction for allowed transitions will be enforced in the selectNextSpeaker function, which is attached to each agent at instantiation!
        self.pattern = AutoPattern(
            agents=list(self.agents.values()),
            initial_agent=self.agents[self.queryAgentName],
            user_agent = self.agents[self.humanAgentName] if self.humanInTheLoop else None,
            exclude_transit_message=False
        )



    # Parse agent chat config from text file.
    # Does not yet instantiate agents
    def parse_agent_config(self, path: str) -> chatGraph:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            self.config = "\n".join(lines)

        agentSpecs: List[agentSpecification] = []
        transitions: Dict[str, List[str]] = {}

        # Split on first line that looks like an edge definition
        split_index = None
        for i, line in enumerate(lines):
            if ":" in line:
                split_index = i
                break

        if split_index is None:
            raise ValueError("Config invalid: No transition section found")

        # Parse agents (type and name)
        # Syntax: <agentType>, <agent_name>
        for line in lines[:split_index]:
            agentType, name = map(str.strip, line.split(",", 1))
            agentSpecs.append(agentSpecification(agentType, name))

        # Parse transitions
        # Syntax: <source_agent_name>: <destination_agent_name>, <destination_agent_name>, ...
        for line in lines[split_index:]:
            source, destinations = map(str.strip, line.split(":", 1))
            destinationList = [dest.strip() for dest in destinations.split(",")]
            transitions[source] = destinationList

        # Check that at least one query agent is present
        # If so, store its index so it can be passed as initial query processor
        has_query_agent = any(agentSpec.agentType == "queryAgent" for agentSpec in agentSpecs)
        if not has_query_agent:
            raise ValueError("Config invalid: No query processing agent found in config.")
        else:
            self.queryAgentName = next(agentSpec.name for agentSpec in agentSpecs if agentSpec.agentType == "queryAgent")

        # Check that a human agent exists if humanInTheLoop is true and that none exist if false
        # If human agent exists, store its index to be passed on properly during chat instantiation
        has_human_agent = any(agentSpec.agentType == "humanAgent" for agentSpec in agentSpecs)
        if not has_human_agent and self.humanInTheLoop:
            raise ValueError("Config invalid: No human agent found but human in the loop requested.")
        elif has_human_agent and not self.humanInTheLoop:
            raise ValueError("Config invalid: Human agent found but human in the loop is disabled.")
        elif has_human_agent and self.humanInTheLoop:
            self.humanAgentName = next(agentSpec.name for agentSpec in agentSpecs if agentSpec.agentType == "humanAgent")

        # Package into chatGraph for easy use
        return chatGraph(
            agentSpecs=agentSpecs,
            transitions=transitions,
        )


    # Select the next speaker based on the last speaker and allowed transitions
    # This uses the LLM to decide which of the possible next agents should speak
    def selectNextSpeaker(self, last_speaker, agents, messages):
        possible_next_agents = self.conversationGraph.transitions[last_speaker.name]
        
        # If there is no possible next agent, return None and terminate chat
        if not possible_next_agents:
            return None

        # Extract last message's suggested next speaker
        lastMessage = messages[-1].content if messages else ""
        suggestedNextSpeaker = lastMessage.nextAgentName

        # Check if suggested next speaker is possible next agent
        if suggestedNextSpeaker in possible_next_agents:
            return agents[suggestedNextSpeaker]
        else:
            return agents[possible_next_agents[0]]  # Fallback to first possible agent



    # Instantiate agents based on the parsed config
    def instantiateAgents(self):
        self.agents = {}
        for spec in self.conversationGraph.agentSpecs:
            agent = eval(f"{spec.agentType}(llm_config=self.llm_config, name = '{spec.name}')")
            self.agents[spec.name] = agent
        return
    


    # Start the group chat using the pattern created from config file, adding in the agent chat config for evaluation
    def startConversation(self, query: str):
        extendedQuery = f"""
Query: {query}

The current agent configuration is as follows:
{self.config}

This config is be read somewhat like a graph, where each agent is a node and the transitions define which agents can pass messages to which other agents.
Each line before the transitions section defines an agent, with the agent type and the agent name separated by a comma.
The lines after that define the allowed transitions, with each line containing the source agent index and the destination agent index separated by a space.
        """
        initiate_group_chat(pattern=self.pattern, messages=extendedQuery, max_rounds=self.maxRounds)