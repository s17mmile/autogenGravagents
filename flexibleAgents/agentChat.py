from dataclasses import dataclass
from typing import List, Dict, Set

# Some useful custom types for instantiating a chat
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
    def __init__(self, configPath: str, llm_config):
        self.conversationGraph = self.parse_agent_config(configPath)
        self.agents = self.instantiateAgents()
        self.llm_config = llm_config


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

        # Package into chatGraph for easy use
        return chatGraph(
            agents=agent_specs,
            transitions=transitions,
        )



    # Instantiate agents based on the parsed config
    def instantiateAgents(self) -> List:
        agents = []
        for spec in self.conversationGraph.agents:
            # TODO: Add logic to instantiate different agent types
            # Here you would instantiate the actual agent based on type
            # For now, we just store the spec
            agents.append(spec)
        return agents