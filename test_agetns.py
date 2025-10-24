# ag2_scaffold.py
from ag2 import AssistantAgent, Swarm, Context
from gravagent_4agentv4 import (
    ScientificInterpreterAgent,
    CoderAgent,
    ExecutorAgent,
    DebuggerAgent,
)

# === Wrappers for your custom agents ===

class InterpreterWrapper(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="interpreter",
            system_message="You are a scientific interpreter for gravitational wave queries."
        )
        self.agent = ScientificInterpreterAgent()

    def step(self, context: Context):
        user_query = context.last_user_message()
        plan = self.agent.interpret_query(user_query)
        return {"role": "assistant", "content": str(plan)}


class CoderWrapper(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="coder",
            system_message="You are a coder agent. Generate analysis code based on tasks."
        )
        self.agent = CoderAgent()

    def step(self, context: Context):
        tasks = context.last_message().get("tasks", [])
        results = self.agent.process_task_list(tasks)
        return {"role": "assistant", "content": str(results)}


class ExecutorWrapper(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="executor",
            system_message="You are an executor agent. Integrate and run code."
        )
        self.agent = ExecutorAgent()

    def step(self, context: Context):
        code_results = context.last_message().get("code_results", [])
        query = context.get("original_query", "")
        available_packages = context.get("available_packages", {})
        result = self.agent.process_code_results(code_results, query, available_packages)
        return {"role": "assistant", "content": str(result)}


class DebuggerWrapper(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="debugger",
            system_message="You are a debugger agent. Fix failing scripts."
        )
        self.agent = DebuggerAgent()

    def step(self, context: Context):
        failure_info = context.last_message()
        # For now this is a placeholder; wire to your DebuggerAgent methods if needed
        return {"role": "assistant", "content": "Debugger would attempt fixes here."}


# === Orchestration Swarm ===

def build_swarm():
    interpreter = InterpreterWrapper()
    coder = CoderWrapper()
    executor = ExecutorWrapper()
    debugger = DebuggerWrapper()

    swarm = Swarm(
        agents=[interpreter, coder, executor, debugger],
        context_variables={"stage": "planning"},
        transitions={
            ("interpreter", "coder"): lambda ctx: ctx["stage"] == "planning",
            ("coder", "executor"): lambda ctx: ctx["stage"] == "coding",
            ("executor", "debugger"): lambda ctx: ctx["stage"] == "execution_failed",
        }
    )
    return swarm


if __name__ == "__main__":
    swarm = build_swarm()
    query = input("🔭 Enter your gravitational wave query: ")
    result = swarm.run(query)
    print("\n=== Final Result ===")
    print(result)
