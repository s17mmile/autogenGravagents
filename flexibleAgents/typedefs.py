# Dataclass and typing imports
from dataclasses import dataclass
from typing import List, Dict

from autogen import ConversableAgent

# All agent specifying information for reading in the config file
@dataclass
class agentSpecification:
    agentType: str
    name: str

# chatGraph holds all agents and their possible transitions in a concise format
# The transitions dict maps source agent names to lists of destination agent names in the exact format needed to constrain speaker transitions in autogen conversations
@dataclass
class chatGraph:
    agents: Dict[str, ConversableAgent]
    transitions: Dict[ConversableAgent, List[ConversableAgent]]