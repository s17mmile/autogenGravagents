import autogen
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern

from dataclasses import dataclass
from typing import List, Dict, Set

from agentTypes import *



# Some useful custom types for flexibility of chat instance
@dataclass
class agentSpecification:
    agent_type: str
    name: str

# While this structure is not strictly necessary, it might make it easier to make a proper config editor later on
# Holds all agents and their possible transitions in a concise format
@dataclass
class chatGraph:
    agents: List[agentSpecification]
    transitions: Dict[int, Set[int]]



# Main class that allows flexible agent conversations based on config files
class flexibleAgentChat:
    def __init__(self, configPath: str, llm_config, humanInTheLoop: bool = True, maxTurns: int = 10):
        self.llm_config = llm_config
        self.maxTurns = maxTurns
        self.humanInTheLoop = humanInTheLoop

        self.conversationGraph = self.parse_agent_config(configPath)

        self.agents = self.instantiateAgents()

        # Create Autogen AutoPattern based on agent config
        # TODO figure out how to encode restricted transitions in the pattern
        self.pattern = AutoPattern(
            agents=self.agents,
            initial_agent=self.agents[self.queryAgentIndex],
            user_agent = self.agents[self.humanAgentIndex] if self.humanInTheLoop else None,
            max_turns=self.maxTurns
        )



    # Parse agent chat config from text file.
    # Does not yet instantiate agents
    def parse_agent_config(path: str) -> chatGraph:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        agent_specs: List[agentSpecification] = []
        transitions: Dict[int, Set[int]] = {}

        # Split on first line that looks like an edge definition
        split_index = None
        for i, line in enumerate(lines):
            if line.replace(" ", "").isdigit():
                split_index = i
                break

        if split_index is None:
            raise ValueError("Config invalid: No transition section found")

        # Parse agents (type and name)
        # Syntax: <agent_type>, <agent_name>
        for line in lines[:split_index]:
            agent_type, name = map(str.strip, line.split(",", 1))
            agent_specs.append((agent_type, name))

        # Parse transitions
        # Syntax: <source_agent_index> <destination_agent_index>
        for line in lines[split_index:]:
            src, dst = map(int, line.split())
            transitions.setdefault(src, set()).add(dst)



        # Check that at least one query agent is present
        # If so, store its index so it can be passed as initial query processor
        has_query_agent = any(agent_type == "queryAgent" for agent_type, _ in agent_specs)
        if not has_query_agent:
            raise ValueError("Config invalid: No query processing agent found in config.")
        else:
            self.queryAgentIndex = next(i for i, (agent_type, _) in enumerate(agent_specs) if agent_type == "queryAgent")

        # Check that a human agent exists if humanInTheLoop is true
        # If so, store its index to be passed on properly during chat instantiation
        has_human_agent = any(agent_type == "humanAgent" for agent_type, _ in agent_specs)
        if not has_human_agent and self.humanInTheLoop:
            raise ValueError("Config invalid: No human agent found but human in the loop requested.")
        else:
            self.humanAgentIndex = next(i for i, (agent_type, _) in enumerate(agent_specs) if agent_type == "humanAgent")



        # Package into chatGraph for easy use
        return chatGraph(
            agents=agent_specs,
            transitions=transitions,
        )

    # Instantiate agents based on the parsed config
    def instantiateAgents(self) -> List:
        agents = []
        for spec in self.conversationGraph.agents:
            exec(f"agent = {spec.agent_type}(name = '{spec.name}', llm_config=self.llm_config)")
            agents.append(agent)
        return agents
    
    # Start the group chat using the pattern created from config file
    def startConversation(self, query: str):
        initiate_group_chat(pattern=self.pattern, messages=query)