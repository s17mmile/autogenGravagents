# General imports
import os, sys, shutil, json
from pydantic import BaseModel
from typing import Dict, List
from enum import Enum

# Autogen imports
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Custom local imports
from flexibleAgents.typedefs import agentSpecification, chatGraph

def print_message(recipient: ConversableAgent, messages: List[Dict], sender: ConversableAgent, config):
    last = messages[-1]
    content = last.get("content")
    print(f"[{sender.name} → {recipient.name}] {content}")
    return None, None

# Main class that allows flexible agent conversations based on config files
class flexibleAgentChat:
    def __init__(self, configPath: str, llm_config, maxRounds: int = 10):
        # Basic setters
        self.configPath = configPath
        self.llm_config = llm_config
        self.maxRounds = maxRounds

        # Instantiate agents based on config in given path.
        # This does not yet initiate the GroupChat instance or start the conversation.
        self.buildChatGraph()

        # Verbose mode prints out extra transition messages and debug stuff
        self.verbose = False

    # Parse agent chat config from text file.
    # Does not yet instantiate agents
    def parseAgentConfig(self, path: str) -> chatGraph:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            self.config = "\n".join(lines)

        agentSpecs: List[agentSpecification] = []
        transitionSpecs: Dict[str, List[str]] = {}

        # Split on first line that looks like a transition definition
        # If none exists, the transition restrictions will be lifted.
        split_index = None
        for i, line in enumerate(lines):
            if ":" in line:
                split_index = i
                break

        # Record whether or not a transition section was found
        if split_index is None:
            self.configIncludesTransitions = False
        else:
            self.configIncludesTransitions = True

        # Parse agents (type and name)
        # Syntax: <agentType>, <agent_name>
        # Along the way, ensure a human agent is provided. Else error out.
        humanAgentCount = 0
        for line in lines[:split_index]:
            agentType, name = map(str.strip, line.split(",", 1))
            agentSpecs.append(agentSpecification(agentType, name))
            if agentType == "humanAgent":
                humanAgentCount += 1

        if humanAgentCount != 1:
            raise ValueError(f"Config invalid: {humanAgentCount} human agents found (exactly 1 required).")

        # Parse transitions if given
        if self.configIncludesTransitions:
            # Syntax: <source_agent_name>: <destination_agent_name>, <destination_agent_name>, ...
            for line in lines[split_index:]:
                source, destinations = map(str.strip, line.split(":", 1))
                destinationList = [dest.strip() for dest in destinations.split(",") if dest.strip()]
                transitionSpecs[source] = destinationList

        # Return parsed specifications for instantiation
        return agentSpecs, transitionSpecs
    
    # Instantiate agents based on the parsed config
    def buildChatGraph(self):
        # Parse agent config file to receive agent and transition specifications
        agentSpecs, transitionSpecs = self.parseAgentConfig(self.configPath)

        self.chatGraph = chatGraph(
            {},                 # Agents            Name: Agent
            {}                  # Transitions       Agent: [Agents]
        )

        # Import/Instantiate Agents and save into agents dict
        sys.path.append(os.path.dirname(__file__))      # Needed for relative local imports
        glob = globals()
        for spec in agentSpecs:
            exec(f"from agentTypes.{spec.agentType} import {spec.agentType}", glob, glob)           # The dynamic import must go into the module's globals!
            agent = eval(f"{spec.agentType}(llm_config=self.llm_config, name = '{spec.name}')")
            self.chatGraph.agents[spec.name] = agent
            if spec.agentType == "humanAgent":          # Separately keep track of the human agent name for query routing
                self.humanAgent = agent

        # Build transitions dict in required format for autogen conversations (essentially just converting keys and values from names to agent instances)
        for source_name, dest_names in transitionSpecs.items():
            source_agent = self.chatGraph.agents[source_name]
            dest_agents = [self.chatGraph.agents[dest_name] for dest_name in dest_names]
            self.chatGraph.transitions[source_agent] = dest_agents

        return

    # Start the group chat using the pattern created from config file, adding in the agent chat config for evaluation
    def startConversation(self, query: str):
        # If no transitions were given, we can simply switch the GroupChat to allow all transitions by switching them from allowed to disallowed
        if self.configIncludesTransitions:
            transitionType = "allowed"
        else:
            transitionType = "disallowed"

        # Initialize a group chat with the instantiated agents and the query
        agentList = list(self.chatGraph.agents.values())
        groupchat = GroupChat(
            agents=agentList,
            messages=[],
            send_introductions=True,
            max_round=self.maxRounds,
            allowed_or_disallowed_speaker_transitions=self.chatGraph.transitions,
            speaker_transitions_type=transitionType,
            speaker_selection_method="auto"
        )

        # Process flow within this group chat is managed by the following manager (necessary even when agents are choosing their own transitions, then it will simply not use an llm but the fixed rules)
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config,
            name = "ManagerAgent"
        )

        # Clear conversation history directory (deletion and recreation of directory, no undoing this!) before chat starts
        shutil.rmtree(f"{os.path.dirname(__file__)}/tempConversation", ignore_errors=True)
        os.makedirs(f"{os.path.dirname(__file__)}/tempConversation", exist_ok=True)

        if self.verbose:
            for agent in agentList:
                agent.register_reply(
                    [ConversableAgent, None],
                    reply_func = print_message,
                    position = 0
                )

        # Start the conversation with the prompt coming from the human and being passed to the manager.
        # We have to pass to the manager to make the GroupChat work properly - else we will just get replies from the one agent.
        # TODO figure out how exactly to expose the messages as they are generated (probably a hook method?) and expose them to the GUI
        result = self.humanAgent.initiate_chat(
            manager,
            message=query,
            clear_history=False
        )

        # Save conversation history as text file in temp directory
        # Wanted to simplify to make this a simple loadable json but kinda can't get it to work as json loads keeps failing
        path = os.path.join(os.path.dirname(__file__), "tempConversation", "conversation.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Conversation Log for query: {query}\n\n")
            for msg in groupchat.messages:
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
                