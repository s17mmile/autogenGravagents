import json
import requests
import chromadb
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
        print(f"[SCIENTIFIC INTERPRETER] Processing: {user_query}")
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
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.rfind("```")
                if start != end:  # Make sure we found different positions
                    response_clean = response_clean[start:end]
            
            # Try to find JSON-like content if no code blocks
            if "{" in response_clean and "}" in response_clean:
                json_start = response_clean.find("{")
                json_end = response_clean.rfind("}") + 1
                potential_json = response_clean[json_start:json_end]
                response_clean = potential_json
            
            # Parse JSON
            result = json.loads(response_clean)
            
            # Validate structure
            required_keys = ["understanding", "tasks", "scientific_context"]
            for key in required_keys:
                if key not in result:
                    result[key] = f"Missing {key} in response"
            
            if "tasks" not in result or not isinstance(result["tasks"], list):
                result["tasks"] = []
            
            return result
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to create a structured response
            return {
                "understanding": "Could not parse structured response",
                "research_summary": "Response parsing failed", 
                "tasks": [],
                "scientific_context": "Error in response formatting",
                "expected_outcomes": "Unable to determine",
                "parsing_error": str(e),
                "raw_response": response
            }


class CoderAgent:
    """
    CODER AGENT Agent that:
    1. Receives tasks from Scientific Interpreter
    2. Queries ChromaDB vector database for documentation
    3. Uses OpenAI API to generate code based on documentation
    4. Executes analysis tasks with proper context
    """
    
    def __init__(self, database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        # LLM Configuration
        self.api_key = "sk-28n2AKk-2GxPyO-szIZCsw"
        self.base_url = "http://131.220.150.238:8080"
        
        self.llm_config = {
            "model": "openai/gpt-4o-mini",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        # Initialize ChromaDB connection
        self.database_path = database_path
        self.client = None
        self.collection = None
        self._initialize_chromadb()
        
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # System prompt for code generation
        self.system_prompt = """You are an expert CODER AGENT specializing in gravitational wave data analysis. Your role is to write Python code based on documentation and task requirements.

You will receive:
1. A specific task to accomplish
2. Relevant documentation from gravitational wave analysis libraries
3. Context about the overall analysis workflow

Your responsibilities:
1. Understand the task requirements
2. Use the provided documentation to write accurate, working code
3. Follow best practices and handle errors appropriately
4. Generate code that accomplishes the specific task
5. Include necessary imports and setup

When writing code:
- Use the exact API calls and methods shown in the documentation
- Include proper error handling with try/except blocks  
- Add print statements for progress tracking
- Write clean, well-documented code
- Save results to variables that can be used by subsequent tasks
- Handle file paths and data loading appropriately

Always structure your response as:

ANALYSIS:
[Your understanding of the task and how the documentation helps]

CODE:
```python
# Your implementation
```

EXPLANATION:
[Brief explanation of what the code does and expected outputs]"""
        
    def _initialize_chromadb(self):
        """Initialize connection to ChromaDB database"""
        try:
            self.client = chromadb.PersistentClient(path=self.database_path)
            # Try to get existing collection
            collections = self.client.list_collections()
            if collections:
                # Use first available collection or look for specific one
                collection_names = [c.name for c in collections]
                print(f"[CHROMADB] Available collections: {collection_names}")
                
                # Look for gravitational wave documentation collection
                target_names = ['gravitational_wave_documentation', 'code_documentation', 'documentation']
                for name in target_names:
                    if name in collection_names:
                        self.collection = self.client.get_collection(name)
                        print(f"[CHROMADB] Connected to collection: {name}")
                        break
                
                if not self.collection:
                    # Use first available collection
                    self.collection = self.client.get_collection(collection_names[0])
                    print(f"[CHROMADB] Using collection: {collection_names[0]}")
            else:
                print("[CHROMADB] No collections found in database")
                
        except Exception as e:
            print(f"[CHROMADB] Warning: Could not connect to ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def query_documentation(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the ChromaDB vector database for relevant documentation"""
        if not self.collection:
            print("[WARNING] No ChromaDB collection available")
            return []
        
        try:
            print(f"[CHROMADB] Querying documentation for: {query}")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'][0] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0.5] * len(results['documents'][0])
                )):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'relevance_score': 1 - distance,
                        'source': metadata.get('source', 'Unknown') if metadata else 'Unknown',
                        'title': metadata.get('title', f'Document {i+1}') if metadata else f'Document {i+1}'
                    })
            
            print(f"[CHROMADB] Found {len(formatted_results)} relevant documents")
            return formatted_results
            
        except Exception as e:
            print(f"[CHROMADB] Error querying documentation: {e}")
            return []
    
    def call_llm(self, prompt: str, include_history: bool = True) -> str:
        """Make API call to LLM for code generation"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history[-6:])  # Keep last 6 messages
        
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
    
    def generate_code_for_task(self, task: Dict, context: Dict = None) -> Dict:
        """Generate code for a specific task using documentation"""
        print(f"\n[CODER AGENT] Processing: {task.get('description', 'Unknown task')}")
        print(f"[CODER AGENT] Type: {task.get('type', 'Unknown')}")
        
        # Step 1: Query documentation based on task
        task_type = task.get('type', '')
        task_description = task.get('description', '')
        task_details = task.get('details', '')
        
        # Create search query for documentation
        search_query = f"{task_type} {task_description} {task_details}".strip()
        
        # Query documentation database
        documentation = self.query_documentation(search_query, n_results=5)
        
        # Step 2: Build documentation context
        doc_context = ""
        if documentation:
            doc_context = "RELEVANT DOCUMENTATION:\n\n"
            for i, doc in enumerate(documentation, 1):
                doc_context += f"Document {i} [{doc['source']}] - {doc['title']}\n"
                doc_context += f"Relevance: {doc['relevance_score']:.3f}\n"
                doc_context += f"Content: {doc['content'][:500]}...\n\n"
        else:
            doc_context = "No specific documentation found for this task.\n"
        
        # Step 3: Build context from previous tasks
        context_info = ""
        if context and context.get('previous_results'):
            context_info = "\nCONTEXT FROM PREVIOUS TASKS:\n"
            for task_id, result in context['previous_results'].items():
                context_info += f"- {task_id}: {result.get('status', 'unknown')}\n"
                if result.get('outputs'):
                    context_info += f"  Available data: {list(result['outputs'].keys())}\n"
        
        # Step 4: Create prompt for code generation
        code_generation_prompt = f"""
TASK TO ACCOMPLISH:
Task ID: {task.get('id', 'unknown')}
Description: {task_description}
Type: {task_type}
Details: {task_details}
Dependencies: {task.get('dependencies', [])}

{doc_context}

{context_info}

Based on the task requirements and the documentation provided above, generate Python code that:

1. Accomplishes the specific task described
2. Uses the APIs and methods shown in the documentation
3. Handles the task dependencies appropriately
4. Includes proper error handling and progress reporting
5. Saves results in variables that can be used by subsequent tasks

For data loading tasks: Load the appropriate gravitational wave data
For processing tasks: Apply the necessary transformations or analysis
For visualization tasks: Create the requested plots and save them

Please provide your analysis and code following the format specified in the system prompt.
"""
        
        print(f"[CODER AGENT] Generating code with {len(documentation)} documentation sources...")
        
        # Step 5: Get code from LLM
        llm_response = self.call_llm(code_generation_prompt)
        
        # Step 6: Parse the response
        analysis, code, explanation = self._parse_code_response(llm_response)
        
        return {
            'task_id': task.get('id', 'unknown'),
            'task_description': task_description,
            'analysis': analysis,
            'code': code,
            'explanation': explanation,
            'documentation_used': len(documentation),
            'documentation_sources': [doc['source'] for doc in documentation],
            'raw_response': llm_response,
            'timestamp': datetime.now().isoformat()
        }
    
    def _parse_code_response(self, response: str) -> tuple:
        """Parse LLM response into analysis, code, and explanation"""
        analysis = ""
        code = ""
        explanation = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('analysis:'):
                current_section = 'analysis'
                analysis += line[9:].strip() + '\n'
            elif line_lower.startswith('code:'):
                current_section = 'code'
            elif line_lower.startswith('explanation:'):
                current_section = 'explanation'
                explanation += line[12:].strip() + '\n'
            elif line.strip().startswith('```python'):
                current_section = 'code'
            elif line.strip().startswith('```') and current_section == 'code':
                current_section = None
            elif current_section == 'analysis':
                analysis += line + '\n'
            elif current_section == 'code' and not line.strip().startswith('```'):
                code += line + '\n'
            elif current_section == 'explanation':
                explanation += line + '\n'
        
        return analysis.strip(), code.strip(), explanation.strip()
    
    def process_task_list(self, tasks: List[Dict], context: Dict = None) -> List[Dict]:
        """Process a list of tasks from the Scientific Interpreter"""
        print(f"\n[CODER AGENT] Processing {len(tasks)} tasks from Scientific Interpreter")
        print("=" * 60)
        
        results = []
        previous_results = {}
        
        for i, task in enumerate(tasks, 1):
            print(f"\n--- Processing Task {i}/{len(tasks)} ---")
            
            # Add previous results to context
            current_context = context or {}
            current_context['previous_results'] = previous_results
            
            # Generate code for this task
            result = self.generate_code_for_task(task, current_context)
            results.append(result)
            
            # Store result for future tasks
            previous_results[task.get('id', f'task_{i}')] = {
                'status': 'completed',
                'code_generated': bool(result['code']),
                'outputs': {'code': result['code']}
            }
            
            print(f"[CODER AGENT] Generated {len(result['code'])} characters of code")
            print(f"[CODER AGENT] Used {result['documentation_used']} documentation sources")
        
        return results


class IntegratedGravitationalWaveSystem:
    """
    Integrated system that combines Scientific Interpreter and CODER AGENT agents
    """
    
    def __init__(self, database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        self.scientific_interpreter = ScientificInterpreterAgent()
        self.data_analyst = CoderAgent(database_path)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("INTEGRATED GRAVITATIONAL WAVE ANALYSIS SYSTEM")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Scientific Interpreter: Ready")
        print(f"CODER AGENT: Ready (ChromaDB: {'Connected' if self.data_analyst.collection else 'Not Connected'})")
    
    def process_query(self, user_query: str) -> Dict:
        """
        Full pipeline: User query → Scientific Interpreter → CODER AGENT → Results
        """
        print(f"\n[SYSTEM] Processing query: {user_query}")
        print("=" * 80)
        
        # Step 1: Scientific Interpreter breaks down the query
        print("\nSTEP 1: SCIENTIFIC INTERPRETATION")
        print("-" * 40)
        interpretation_result = self.scientific_interpreter.interpret_query(user_query)
        
        if "error" in interpretation_result:
            return {
                "session_id": self.session_id,
                "error": "Scientific interpretation failed",
                "details": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        tasks = interpretation_result.get('tasks', [])
        if not tasks:
            return {
                "session_id": self.session_id,
                "error": "No tasks generated from query",
                "interpretation": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"[SYSTEM] Generated {len(tasks)} tasks")
        
        # Step 2: CODER AGENT generates code for tasks
        print(f"\nSTEP 2: DATA ANALYSIS & CODE GENERATION")
        print("-" * 40)
        code_results = self.data_analyst.process_task_list(tasks)
        
        # Step 3: Compile final results
        final_result = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "original_query": user_query,
            "scientific_interpretation": {
                "understanding": interpretation_result.get('understanding', ''),
                "research_summary": interpretation_result.get('research_summary', ''),
                "scientific_context": interpretation_result.get('scientific_context', ''),
                "expected_outcomes": interpretation_result.get('expected_outcomes', ''),
                "tasks_generated": len(tasks)
            },
            "data_analysis": {
                "tasks_processed": len(code_results),
                "total_documentation_sources": sum(r['documentation_used'] for r in code_results),
                "code_results": code_results
            },
            "token_usage": {
                "scientific_interpreter": self.scientific_interpreter.total_tokens_used,
                "data_analyst": self.data_analyst.total_tokens_used,
                "total": self.scientific_interpreter.total_tokens_used + self.data_analyst.total_tokens_used
            }
        }
        
        return final_result
    
    def save_session(self, result: Dict, output_dir: str = "./integrated_results") -> str:
        """Save complete session results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        filename = f"gw_analysis_session_{result['session_id']}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return str(filepath)
    
    def get_system_status(self) -> str:
        """Get status of both agents"""
        return f"""
INTEGRATED GRAVITATIONAL WAVE ANALYSIS SYSTEM STATUS
Session: {self.session_id}

Scientific Interpreter:
- Tokens used: {self.scientific_interpreter.total_tokens_used}
- Conversation history: {len(self.scientific_interpreter.conversation_history)} messages

Coder Agent:
- ChromaDB: {'Connected' if self.data_analyst.collection else 'Not Connected'}
- Collection: {self.data_analyst.collection.name if self.data_analyst.collection else 'None'}
- Tokens used: {self.data_analyst.total_tokens_used}
- Conversation history: {len(self.data_analyst.conversation_history)} messages

Total tokens used: {self.scientific_interpreter.total_tokens_used + self.data_analyst.total_tokens_used}
"""


def main():
    """Main interactive interface"""
    
    # Initialize integrated system
    system = IntegratedGravitationalWaveSystem()
    
    print(f"\n{system.get_system_status()}")
    
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print("Commands:")
    print("  - Enter a gravitational wave analysis query")
    print("  - 'status' to check system status")
    print("  - 'quit' to exit")
    print("="*60)
    
    while True:
        print(f"\n{'-'*60}")
        user_input = input("Enter gravitational wave analysis query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        elif user_input.lower() == 'status':
            print(system.get_system_status())
        elif user_input:
            try:
                # Process the complete pipeline
                result = system.process_query(user_input)
                
                # Display results
                print(f"\n{'='*80}")
                print("ANALYSIS COMPLETE")
                print(f"{'='*80}")
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    if "details" in result:
                        print(f"Details: {result['details']}")
                else:
                    # Display scientific interpretation
                    interpretation = result['scientific_interpretation']
                    print(f"Understanding: {interpretation['understanding']}")
                    print(f"Tasks generated: {interpretation['tasks_generated']}")
                    
                    # Display data analysis results
                    analysis = result['data_analysis']
                    print(f"\nCode generation:")
                    print(f"Tasks processed: {analysis['tasks_processed']}")
                    print(f"Documentation sources used: {analysis['total_documentation_sources']}")
                    
                    # Show generated code for each task
                    for i, code_result in enumerate(analysis['code_results'], 1):
                        print(f"\nTask {i}: {code_result['task_description']}")
                        print(f"Documentation sources: {code_result['documentation_sources']}")
                        if code_result['code']:
                            print("Generated code preview:")
                            print("-" * 30)
                            code_preview = code_result['code'][:300]
                            print(code_preview + "..." if len(code_result['code']) > 300 else code_preview)
                            print("-" * 30)
                        else:
                            print("No code generated for this task")
                    
                    # Display token usage
                    tokens = result['token_usage']
                    print(f"\nToken Usage:")
                    print(f"Scientific Interpreter: {tokens['scientific_interpreter']}")
                    print(f"CODER AGENT: {tokens['data_analyst']}")
                    print(f"Total: {tokens['total']}")
                
                # Save session
                saved_path = system.save_session(result)
                print(f"\nSession saved to: {saved_path}")
                
            except Exception as e:
                print(f"Error processing query: {e}")
        else:
            print("Please enter a query or command.")

if __name__ == "__main__":
    main() 