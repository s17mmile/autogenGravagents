import json
import requests
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

class ScientificInterpreterAgent:
    """
    Scientific Interpreter Agent that:
    1. Receives NLP queries about gravitational waves
    2. Uses OpenAI/LLM as thinking backbone
    3. Researches information from the internet when needed
    4. Breaks down complex tasks into doable parts
    """
    
    def __init__(self):
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.api_key = "sk-28n2AKk-2GxPyO-szIZCsw"
        self.base_url = "http://131.220.150.238:8080"
    
        self.llm_config = {
        "model": "openai/gpt-4o-mini",
        "api_key": self.api_key,
        "base_url": self.base_url,
        "temperature": 0.1,
        }
        
        # Base system prompt - lets the LLM think naturally about GW physics
        self.system_prompt = """You are a scientific interpreter specializing in gravitational wave physics and data analysis. Your role is to understand natural language queries about gravitational waves and break them down into actionable computational tasks.

You have access to web search to gather current information when needed. Use your scientific knowledge and research capabilities to:

1. Understand what the user is asking about gravitational wave physics
2. Research any information you need from the internet 
3. Break down the request into specific, doable computational tasks
4. Provide scientific context for why these tasks make sense

You can search the web when you need current information, specific parameters, or technical details that would help you better understand and plan the analysis.

When you provide your final response, structure it as JSON with this general format:
{
  "understanding": "Your interpretation of what the user wants",
  "research_summary": "Key information you found from research",
  "tasks": [
    {
      "id": "task_1",
      "description": "What this task accomplishes",
      "type": "category like data_loading, analysis, visualization",
      "details": "Specific details about how to approach this task",
      "dependencies": ["list of other task ids this depends on"]
    }
  ],
  "scientific_context": "Why this approach makes sense scientifically",
  "expected_outcomes": "What we should learn from this analysis"
}"""
        
    def call_llm(self, prompt: str, include_history: bool = True) -> str:
        """Make API call to LLM"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history[-10:])  # Keep last 10 messages
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.llm_config['base_url']}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_config["model"],
                    "messages": messages,
                    "temperature": self.llm_config["temperature"],
                    "max_tokens": 4000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Update conversation history
                if include_history:
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": content})
                
                # Track token usage
                if "usage" in result:
                    self.total_tokens_used += result["usage"]["total_tokens"]
                
                return content
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def web_search(self, query: str) -> List[Dict]:
        """
        Placeholder for web search functionality
        In a real implementation, this would use a search API like Google, Bing, or DuckDuckGo
        """
        print(f"[WEB SEARCH] Searching for: {query}")
        
        # This is a placeholder - replace with actual search API
        placeholder_results = [
            {
                "title": f"Search result for: {query}",
                "url": "https://example.com",
                "snippet": f"Information about {query} would be retrieved from the web"
            }
        ]
        
        return placeholder_results
    
    def research_query(self, user_query: str) -> Dict:
        """
        Research the user query to gather information needed for task planning
        """
        research_prompt = f"""
I need to research information to properly understand and plan for this gravitational wave analysis query:

USER QUERY: "{user_query}"

Based on this query, what specific information should I search for on the internet to better understand:
1. The specific gravitational wave event(s) mentioned (if any)
2. The analysis methods being requested
3. Technical parameters or data sources needed
4. Current best practices for this type of analysis

Please suggest 3-5 specific web search queries that would help me gather the necessary information to properly break down this task.

Format your response as a JSON list of search queries:
["search query 1", "search query 2", "search query 3"]
"""
        
        search_suggestions_response = self.call_llm(research_prompt, include_history=False)
        
        try:
            # Extract search queries from response
            search_queries = json.loads(search_suggestions_response)
            if not isinstance(search_queries, list):
                search_queries = [user_query]  # Fallback
        except:
            # Fallback to basic search
            search_queries = [user_query, "gravitational wave data analysis"]
        
        # Perform web searches
        research_results = []
        for query in search_queries[:3]:  # Limit to 3 searches
            results = self.web_search(query)
            research_results.extend(results)
        
        return {
            "search_queries_used": search_queries,
            "search_results": research_results
        }
    
    def interpret_query(self, user_query: str) -> Dict:
        """
        Main method: interpret user query and break it down into tasks
        """
        print(f"Scientific Interpreter processing: {user_query}")
        print("=" * 60)
        
        # Step 1: Research the query
        print("Step 1: Researching query...")
        research_data = self.research_query(user_query)
        
        print(f"Used search queries: {research_data['search_queries_used']}")
        print(f"Found {len(research_data['search_results'])} research results")
        
        # Step 2: Build context for analysis planning
        research_context = "\n".join([
            f"- {result['title']}: {result['snippet']}" 
            for result in research_data['search_results']
        ])
        
        # Step 3: Interpret query with research context
        print("\nStep 2: Interpreting query with research context...")
        
        interpretation_prompt = f"""
USER QUERY: "{user_query}"

RESEARCH CONTEXT FROM WEB SEARCH:
{research_context}

Based on the user query and the research context above, please:

1. Interpret what the user wants to accomplish in gravitational wave analysis
2. Use your scientific knowledge and the research context to plan the approach
3. Break down the request into specific computational tasks that can be executed
4. Consider what data sources, methods, and tools would be appropriate
5. Think about the logical flow - what needs to be done first, second, etc.

Please provide your analysis in the JSON format specified in the system prompt.
Focus on creating tasks that are specific and actionable, drawing from both your knowledge of gravitational wave physics and the research context provided.
"""
        
        interpretation_response = self.call_llm(interpretation_prompt)
        
        # Step 4: Parse and validate response
        try:
            result = self._parse_interpretation_response(interpretation_response)
            result["research_data"] = research_data
            result["session_id"] = self.session_id
            result["timestamp"] = datetime.now().isoformat()
            result["original_query"] = user_query
            
            return result
            
        except Exception as e:
            # Return error info for debugging
            return {
                "error": f"Failed to parse interpretation: {str(e)}",
                "raw_response": interpretation_response,
                "research_data": research_data,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "original_query": user_query
            }
    
    def _parse_interpretation_response(self, response: str) -> Dict:
        """Parse the LLM's interpretation response"""
        print(f"\n[DEBUG] Raw LLM response length: {len(response)} characters")
        print(f"[DEBUG] Response preview: {response[:200]}...")
        
        try:
            # Try to extract JSON from the response
            response_clean = response.strip()
            
            # Handle code blocks
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                if end == -1:  # No closing ```
                    response_clean = response_clean[start:]
                else:
                    response_clean = response_clean[start:end]
                print(f"[DEBUG] Extracted JSON from code block")
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.rfind("```")
                if start != end:  # Make sure we found different positions
                    response_clean = response_clean[start:end]
                print(f"[DEBUG] Extracted content from code block")
            
            # Try to find JSON-like content if no code blocks
            if "{" in response_clean and "}" in response_clean:
                json_start = response_clean.find("{")
                json_end = response_clean.rfind("}") + 1
                potential_json = response_clean[json_start:json_end]
                print(f"[DEBUG] Found potential JSON content: {potential_json[:100]}...")
                response_clean = potential_json
            
            print(f"[DEBUG] Attempting to parse JSON...")
            # Parse JSON
            result = json.loads(response_clean)
            print(f"[DEBUG] Successfully parsed JSON with keys: {list(result.keys())}")
            
            # Validate structure
            required_keys = ["understanding", "tasks", "scientific_context"]
            for key in required_keys:
                if key not in result:
                    result[key] = f"Missing {key} in response"
            
            if "tasks" not in result or not isinstance(result["tasks"], list):
                result["tasks"] = []
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing failed: {e}")
            print(f"[DEBUG] Attempted to parse: {response_clean[:300]}...")
            
            # If JSON parsing fails, try to create a structured response
            return {
                "understanding": "Could not parse structured response",
                "research_summary": "Response parsing failed", 
                "tasks": [],
                "scientific_context": "Error in response formatting",
                "expected_outcomes": "Unable to determine",
                "parsing_error": str(e),
                "raw_response": response,
                "attempted_parse": response_clean[:500]
            }
    
    def save_interpretation(self, result: Dict, output_dir: str = "./interpretation_results") -> str:
        """Save interpretation results to file"""
        Path(output_dir).mkdir(exist_ok=True)
        
        filename = f"interpretation_{result.get('session_id', 'unknown')}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return str(filepath)
    
    def get_summary(self) -> str:
        """Get summary of agent status"""
        return f"""
Scientific Interpreter Agent
Session: {self.session_id}
Total tokens used: {self.total_tokens_used}
Conversation history: {len(self.conversation_history)} messages

Ready to interpret gravitational wave analysis queries!
"""

# Demo and testing function
def demo_scientific_interpreter():
    """Demonstrate the Scientific Interpreter Agent"""
    
    print("SCIENTIFIC INTERPRETER AGENT DEMO")
    print("=" * 50)
    
    # Initialize agent (no config needed - uses built-in API settings)
    agent = ScientificInterpreterAgent()
    print(agent.get_summary())
    
    # Test queries
    test_queries = [
        "plot the strain time series and Q-transform spectrogram of the first neutron star merger event detected",
        "analyze the gravitational wave signal from GW150914",
        "compare the noise characteristics between LIGO Hanford and Livingston detectors"
    ]
    
    print("\nTEST QUERIES:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
    
    return agent

if __name__ == "__main__":
    # Initialize the agent
    interpreter = demo_scientific_interpreter()
    
    print("\n" + "="*50)
    print("Interactive mode - Enter queries or 'quit' to exit")
    
    while True:
        print("\n" + "-"*50)
        user_input = input("Enter gravitational wave analysis query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        elif user_input:
            try:
                # Process the query
                result = interpreter.interpret_query(user_input)
                
                # Display results
                print(f"\n{'='*60}")
                print("INTERPRETATION RESULTS:")
                print(f"{'='*60}")
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    if "raw_response" in result:
                        print(f"\nRaw response: {result['raw_response'][:500]}...")
                else:
                    print(f"Understanding: {result.get('understanding', 'N/A')}")
                    print(f"\nResearch Summary: {result.get('research_summary', 'N/A')}")
                    
                    tasks = result.get('tasks', [])
                    print(f"\nTasks Generated ({len(tasks)}):")
                    for i, task in enumerate(tasks, 1):
                        print(f"{i}. {task.get('description', 'No description')}")
                        print(f"   Type: {task.get('type', 'Unknown')}")
                        if task.get('dependencies'):
                            print(f"   Dependencies: {task['dependencies']}")
                    
                    print(f"\nScientific Context: {result.get('scientific_context', 'N/A')}")
                    print(f"\nExpected Outcomes: {result.get('expected_outcomes', 'N/A')}")
                
                # Save results
                saved_path = interpreter.save_interpretation(result)
                print(f"\nResults saved to: {saved_path}")
                
                print(f"\nTokens used this session: {interpreter.total_tokens_used}")
                
            except Exception as e:
                print(f"Error processing query: {e}")
        else:
            print("Please enter a query.")