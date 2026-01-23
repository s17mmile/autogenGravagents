# General imports
import os

# Autogen imports
import autogen
from autogen import GroupChat, GroupChatManager

# Custom local imports
from flexibleAgents.typedefs import agentSpecification, chatGraph

# Dynamic import of agent types from agentTypes dir
# We want to exclusively import the creation functions for each agent type
# The creation function must be named the same as the agent type itself
# I'm aware this is not clean, but it's the best I can currently think of for easy introsuction of new agent types
# Not entirely sure why importing the agentTypes directory as a whole does not make it possible to access the individual module functions
for file in os.listdir(os.path.dirname(__file__) + "/agentTypes"):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        try:
            exec(f"from flexibleAgents.agentTypes.{module_name} import {module_name}")
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}. Skipping this module.")



# Main class that allows flexible agent conversations based on config files
class flexibleAgentChat:
    def __init__(self, configPath: str, llm_config, maxRounds: int = 10):
        # Basic setters
        self.configPath = configPath
        self.llm_config = llm_config
        self.maxRounds = maxRounds

        # Instantiate agents based on config in given path.
        # This does not yet initiate the GroupChat instance or start the conversation
        self.buildChatGraph()

        self.humanQueryRecipient = input(f"Query recipient Name ({self.chatGraph.agents}): ")

    # Parse agent chat config from text file.
    # Does not yet instantiate agents
    def parseAgentConfig(self, path: str) -> chatGraph:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            self.config = "\n".join(lines)

        agentSpecs: List[agentSpecification] = []
        transitionSpecs: Dict[str, List[str]] = {}

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
            destinationList = [dest.strip() for dest in destinations.split(",") if dest.strip()]
            transitionSpecs[source] = destinationList

        # Check that a human agent exists. Everything else is up to the user.
        has_human_agent = any(agentSpec.agentType == "humanAgent" for agentSpec in agentSpecs)
        if not has_human_agent:
            raise ValueError("Config invalid: No human agent found (required).")
        else:
            self.humanAgentName = next(agentSpec.name for agentSpec in agentSpecs if agentSpec.agentType == "humanAgent")

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

        # Instantiate Agents and save into agents dict
        for spec in agentSpecs:
            agent = eval(f"{spec.agentType}(llm_config=self.llm_config, name = '{spec.name}')")
            self.chatGraph.agents[spec.name] = agent

        # Build transitions dict in required format for autogen conversations (essentially just converting keys and values from names to agent instances)
        for source_name, dest_names in transitionSpecs.items():
            source_agent = self.chatGraph.agents[source_name]
            dest_agents = [self.chatGraph.agents[dest_name] for dest_name in dest_names]
            self.chatGraph.transitions[source_agent] = dest_agents

        return


    # Enforce chat flow given by config file.
    # Passed to group chat for next speaker selection.
    def selectNextSpeaker(self, lastSpeaker, groupchat: GroupChat) -> autogen.ConversableAgent:
        lastMessage = groupchat.messages[-1]
        
        print(lastMessage)

        # The human may speak to any agent they wish, this is stored separately.
        if lastMessage["name"] == self.humanAgentName:
            return self.chatGraph.agents[self.humanQueryRecipient]
        
        # For all other agents, cross-check their next agent suggestion with the allowed transitions.
        allowedNextSpeakerNames = self.chatGraph.transitions.get(lastSpeaker, [])
        nextSpeakerName = lastMessage.nextAgentName if hasattr(lastMessage, 'nextAgentName') else None
        
        if nextSpeakerName not in allowedNextSpeakerNames:
            nextSpeaker = None
        else:
            nextSpeaker = self.chatGraph.agents[nextSpeakerName]

        print("Found next speaker:", nextSpeaker)

        return nextSpeaker

    # Start the group chat using the pattern created from config file, adding in the agent chat config for evaluation
    def startConversation(self, query: str, startingAgentName: str = None):
        # Initialize a group chat with the instantiated agents and the query
        groupchat = GroupChat(
            agents=list(self.chatGraph.agents.values()),
            messages=[],
            send_introductions=True,
            max_round=self.maxRounds,
            allowed_or_disallowed_speaker_transitions=self.chatGraph.transitions,
            speaker_transitions_type="allowed",
            speaker_selection_method=self.selectNextSpeaker
        )

        # Process flow within this group chat is managed by the following manager (wait I don't want this do I)
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=False,
            name = "FlexibleAgentChatManager"
        )

        # Start the conversation with the prompt coming from the human and being passed to the manager.
        humanAgent = self.chatGraph.agents[self.humanAgentName]
        result = humanAgent.initiate_chat(
            manager,
            message=query,
            clear_history=False
        )

        print(result)