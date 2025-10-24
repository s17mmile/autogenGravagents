import json
import tempfile
import os
import requests
import chromadb
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from typing import List

class ScientificInterpreterAgent:
    """
    Scientific Interpreter Agent that:
    1. Receives NLP queries about gravitational waves
    2. Uses OpenAI/LLM as thinking backbone with its training knowledge
    3. Breaks down complex tasks into doable parts
    4. No longer depends on web search - relies on LLM knowledge
    """
    
    def __init__(self):
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.api_key = "sk-ev4v3VCbmx15mXTKC_c30w"
        self.base_url = "http://131.220.150.238:8080"
    
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        # Enhanced system prompt that relies on LLM knowledge instead of web search
        self.system_prompt = """You are a scientific interpreter specializing in gravitational wave physics and data analysis. Your role is to understand natural language queries about gravitational waves and break them down into actionable computational tasks.

You have extensive knowledge about:
- Gravitational wave theory and detection
- LIGO, Virgo, and other gravitational wave observatories
- Data analysis techniques (matched filtering, parameter estimation, etc.)
- Scientific Python packages (GWpy, PyCBC, LALSuite, etc.)
- Signal processing and statistical analysis methods
- Gravitational wave events and their characteristics

CRITICAL: You MUST respond with valid JSON only. No additional text before or after the JSON.

Your response must be a valid JSON object with this exact structure:
{
  "understanding": "Your interpretation of what the user wants",
  "knowledge_context": "Relevant gravitational wave knowledge applied to this query",
  "tasks": [
    {
      "id": "task_1",
      "description": "What this task accomplishes",
      "type": "data_loading|analysis|visualization|processing",
      "details": "Specific details about how to approach this task",
      "dependencies": []
    }
  ],
  "scientific_context": "Why this approach makes sense scientifically",
  "expected_outcomes": "What we should learn from this analysis"
}

Always ensure:
1. At least one task is generated for any valid query
2. Task IDs are unique and descriptive
3. Task types are one of: data_loading, analysis, visualization, processing
4. Dependencies reference actual task IDs from the same response
5. Use your training knowledge about gravitational wave analysis best practices
6. Respond ONLY with the JSON object - no explanatory text"""
        
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
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"[ERROR] Exception in call_llm: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def interpret_query(self, user_query: str) -> Dict:
        """Main method: interpret user query and break it down into tasks using LLM knowledge"""
        print(f"[SCIENTIFIC INTERPRETER] Processing: {user_query}")
        print("=" * 60)
        
        # Direct interpretation using LLM's training knowledge
        print("Step 1: Analyzing query with gravitational wave knowledge...")
        
        interpretation_prompt = f"""
USER QUERY: "{user_query}"

Based on your knowledge of gravitational wave physics and data analysis, analyze this query and create a structured analysis plan. Consider:

1. What gravitational wave concepts are involved?
2. What data sources might be needed (LIGO, Virgo, public datasets)?
3. What analysis techniques are appropriate (filtering, parameter estimation, etc.)?
4. What visualization or output would be most informative?

Break down the request into 2-4 specific computational tasks that follow gravitational wave analysis best practices.

For example, if asked to "analyze GW150914":
1. Load GW150914 strain data from LIGO Open Science Center
2. Apply bandpass filtering (35-350 Hz) to remove noise  
3. Perform matched filtering with binary black hole templates
4. Create time-frequency spectrogram showing the chirp

Use your knowledge of standard gravitational wave analysis workflows to create appropriate tasks.

RESPOND WITH ONLY THE JSON OBJECT - NO OTHER TEXT.
"""
        
        interpretation_response = self.call_llm(interpretation_prompt)
        
        print(f"[DEBUG] Raw interpretation response length: {len(interpretation_response)}")
        print(f"[DEBUG] Response preview: {interpretation_response[:200]}...")
        
        # Parse and validate response
        try:
            result = self._parse_interpretation_response(interpretation_response)
            
            # Ensure we have valid tasks
            if not result.get("tasks") or len(result["tasks"]) == 0:
                print("[WARNING] No tasks generated, creating default task")
                result["tasks"] = [{
                    "id": "default_analysis",
                    "description": f"Analyze gravitational wave data related to: {user_query}",
                    "type": "analysis",
                    "details": "Perform basic gravitational wave data analysis using standard techniques",
                    "dependencies": []
                }]
            
            result["session_id"] = self.session_id
            result["timestamp"] = datetime.now().isoformat()
            result["original_query"] = user_query
            result["knowledge_based"] = True  # Flag indicating this used LLM knowledge only
            
            print(f"[SUCCESS] Generated {len(result['tasks'])} tasks using LLM knowledge")
            for i, task in enumerate(result['tasks'], 1):
                print(f"  Task {i}: {task.get('id', 'unknown')} - {task.get('description', 'No description')}")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Failed to parse interpretation: {str(e)}")
            # Return error with fallback tasks
            return {
                "error": f"Failed to parse interpretation: {str(e)}",
                "understanding": f"Analysis of gravitational wave query: {user_query}",
                "knowledge_context": "LLM knowledge applied but response parsing failed",
                "tasks": [{
                    "id": "fallback_analysis",
                    "description": f"Basic gravitational wave analysis for: {user_query}",
                    "type": "analysis",
                    "details": "Perform basic analysis using standard gravitational wave techniques",
                    "dependencies": []
                }],
                "scientific_context": "Fallback analysis due to response parsing issues",
                "expected_outcomes": "Basic gravitational wave analysis results",
                "raw_response": interpretation_response,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "original_query": user_query,
                "knowledge_based": True
            }
    
    def _parse_interpretation_response(self, response: str) -> Dict:
        """Parse the LLM's interpretation response with better error handling"""
        try:
            # Clean the response
            response_clean = response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                if end != -1:
                    response_clean = response_clean[start:end]
                else:
                    response_clean = response_clean[start:]
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.rfind("```")
                if start < end:
                    response_clean = response_clean[start:end]
            
            # Find JSON boundaries if no code blocks
            if not response_clean.startswith('{'):
                json_start = response_clean.find('{')
                if json_start != -1:
                    json_end = response_clean.rfind('}') + 1
                    response_clean = response_clean[json_start:json_end]
            
            print(f"[DEBUG] Cleaned response length: {len(response_clean)}")
            print(f"[DEBUG] Cleaned response preview: {response_clean[:300]}...")
            
            # Parse JSON
            result = json.loads(response_clean)
            
            # Validate required keys (updated to match new structure)
            required_keys = ["understanding", "tasks", "scientific_context"]
            for key in required_keys:
                if key not in result:
                    if key == "understanding":
                        result[key] = f"Analysis of gravitational wave query using LLM knowledge"
                    elif key == "scientific_context":
                        result[key] = f"Applied gravitational wave analysis best practices"
                    else:
                        result[key] = f"Generated {key}"
            
            # Add knowledge_context if missing
            if "knowledge_context" not in result:
                result["knowledge_context"] = "Applied training knowledge of gravitational wave physics and analysis"
            
            # Validate tasks structure
            if "tasks" not in result or not isinstance(result["tasks"], list):
                result["tasks"] = []
            
            # Ensure each task has required fields
            for i, task in enumerate(result["tasks"]):
                if not isinstance(task, dict):
                    result["tasks"][i] = {"id": f"task_{i+1}", "description": "Invalid task", "type": "analysis", "details": "", "dependencies": []}
                else:
                    task.setdefault("id", f"task_{i+1}")
                    task.setdefault("description", "Generated task")
                    task.setdefault("type", "analysis")
                    task.setdefault("details", "")
                    task.setdefault("dependencies", [])
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error: {str(e)}")
            print(f"[ERROR] Problematic response: {response_clean[:500]}...")
            raise e
        except Exception as e:
            print(f"[ERROR] Unexpected error in parsing: {str(e)}")
            raise e



class CoderAgent:
    """
    CODER AGENT that:
    1. Receives tasks from Scientific Interpreter
    2. Checks Python environment for installed scientific packages
    3. Queries ChromaDB vector database with package-aware queries
    4. Uses OpenAI API to generate code based on documentation
    5. Executes analysis tasks with proper context
    """
    
    # Scientific packages to check for
    SCIENTIFIC_PACKAGES = [
        "gwpy", "ligo.skymap", "astropy", "pandas", "numpy", "scipy", 
        "matplotlib", "seaborn", "h5py", "healpy", "bilby", "pycbc", 
        "torch", "tensorflow", "jax"
    ]
    
    def __init__(self, database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        # LLM Configuration
        self.api_key = "sk-ev4v3VCbmx15mXTKC_c30w"
        self.base_url = "http://131.220.150.238:8080"
        
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        # Initialize ChromaDB connection
        self.database_path = database_path
        self.client = None
        self.collection = None
        self._initialize_chromadb()
        
        # Get installed scientific packages
        self.installed_packages = self._get_installed_scientific_packages()
        
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # System prompt for code generation
        self.system_prompt = """You are an expert CODER AGENT specializing in gravitational wave data analysis. Your role is to write Python code based on documentation and task requirements.

You will receive:
1. A specific task to accomplish
2. Relevant documentation from gravitational wave analysis libraries
3. Context about the overall analysis workflow
4. Information about available Python packages in the current environment

Your responsibilities:
1. Understand the task requirements
2. Use the provided documentation to write accurate, working code
3. Follow best practices and handle errors appropriately
4. Generate code that accomplishes the specific task
5. Include necessary imports and setup
6. Only use packages that are confirmed to be available in the environment

When writing code:
- Use the exact API calls and methods shown in the documentation
- Include proper error handling with try/except blocks  
- Add print statements for progress tracking
- Write clean, well-documented code
- Save results to variables that can be used by subsequent tasks
- Handle file paths and data loading appropriately
- Only import and use packages that are available in the current environment

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
                target_names = ['gw_comprehensive_docs', 'gravitational_wave_documentation', 'code_documentation', 'documentation']
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
    
    def _get_installed_scientific_packages(self) -> Dict[str, str]:
        """Check which scientific packages are installed and get their versions"""
        installed_packages = {}
        
        print(f"[PACKAGE CHECK] Checking for scientific packages...")
        
        for package_name in self.SCIENTIFIC_PACKAGES:
            try:
                pkg_version = version(package_name)
                installed_packages[package_name] = pkg_version
                print(f"[PACKAGE CHECK] ✓ {package_name} v{pkg_version}")
            except PackageNotFoundError:
                print(f"[PACKAGE CHECK] ✗ {package_name} not found")
            except Exception as e:
                print(f"[PACKAGE CHECK] ? {package_name} check failed: {e}")
        
        print(f"[PACKAGE CHECK] Found {len(installed_packages)} scientific packages")
        return installed_packages
    
    def _build_package_context(self) -> str:
        """Build context string about available packages for queries"""
        if not self.installed_packages:
            return "No specific scientific packages detected in environment"
        
        package_context = "Available scientific packages:\n"
        for package, version in self.installed_packages.items():
            package_context += f"- {package} v{version}\n"
        
        return package_context.strip()
    
    
    def query_documentation_with_rag(self, query: str, n_results: int = 3) -> str:
        """Query documentation and return RAG-style synthesized response"""
        if not self.collection:
            return "No documentation database available."
        
        try:
            # Get relevant documents
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'] or not results['documents'][0]:
                return "No relevant documentation found."
            
            # Build context from retrieved documents
            context = "RETRIEVED DOCUMENTATION:\n\n"
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                title = metadata.get('title', f'Document {i}') if metadata else f'Document {i}'
                context += f"Document {i} ({title}):\n{doc[:800]}...\n\n"
            
            # Generate RAG response using local LLM
            rag_prompt = f"""
    Based on the retrieved documentation below, answer the technical question about gravitational wave analysis.

    {context}

    QUESTION: {query}

    Provide a technical answer based on the documentation. If the documentation contains specific code examples or API usage, include them.

    ANSWER:
    """
            
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.llm_config["model"],
                        "messages": [{"role": "user", "content": rag_prompt}],
                        "temperature": 0.1,
                        "max_tokens": 1500
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"RAG query failed: {response.status_code}"
                    
            except Exception as e:
                return f"Error in RAG generation: {e}"
                
        except Exception as e:
            return f"Error querying RAG database: {e}"

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
                timeout=180
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
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"[ERROR] Exception in call_llm: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def generate_code_for_task(self, task: Dict, context: Dict = None) -> Dict:
        """Generate code for a specific task using documentation and package info"""
        print(f"\n[CODER AGENT] Processing: {task.get('description', 'Unknown task')}")
        print(f"[CODER AGENT] Type: {task.get('type', 'Unknown')}")
        print(f"[CODER AGENT] Task ID: {task.get('id', 'Unknown')}")
        
        # Validate task structure
        if not isinstance(task, dict):
            print("[ERROR] Invalid task structure - not a dictionary")
            return {
                'task_id': 'invalid',
                'error': 'Invalid task structure',
                'analysis': 'Task is not a valid dictionary',
                'code': '',
                'explanation': 'Cannot process invalid task',
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 1: Build enhanced query with package information
        task_type = task.get('type', 'analysis')
        task_description = task.get('description', 'Unknown task')
        task_details = task.get('details', '')
        
        # Create search query for documentation with package context
        base_query = f"{task_type} {task_description} {task_details}".strip()
        
        # Add package names to query for better matching
        package_names = " ".join(self.installed_packages.keys()) if self.installed_packages else ""
        search_query = f"{base_query} {package_names}".strip()
        
        print(f"[CODER AGENT] Searching with enhanced query including {len(self.installed_packages)} packages")
        
        # Query documentation database
        # Try RAG-enhanced query first, fallback to regular query
        try:
            rag_response = self.query_documentation_with_rag(search_query, n_results=3)
            documentation = [{'content': rag_response, 'source': 'RAG_synthesized', 'title': 'RAG Response', 'relevance_score': 1.0}]
            print(f"[CODER AGENT] Using RAG-synthesized response")
        except:
            documentation = self.query_documentation(search_query, n_results=5)
            print(f"[CODER AGENT] Using standard documentation query")
        
        # Step 2: Build documentation context
        doc_context = ""
        if documentation:
            doc_context = "RELEVANT DOCUMENTATION:\n\n"
            for i, doc in enumerate(documentation, 1):
                doc_context += f"Document {i} [{doc['source']}] - {doc['title']}\n"
                doc_context += f"Relevance: {doc['relevance_score']:.3f}\n"
                doc_context += f"Content: {doc['content'][:500]}...\n\n"
        else:
            doc_context = "No specific documentation found for this task. Use general gravitational wave analysis knowledge.\n"
        
        # Step 3: Build package context
        package_context = self._build_package_context()
        
        # Step 4: Build context from previous tasks
        context_info = ""
        if context and context.get('previous_results'):
            context_info = "\nCONTEXT FROM PREVIOUS TASKS:\n"
            for task_id, result in context['previous_results'].items():
                context_info += f"- {task_id}: {result.get('status', 'unknown')}\n"
                if result.get('outputs'):
                    context_info += f"  Available data: {list(result['outputs'].keys())}\n"
        
        # Step 5: Create comprehensive prompt for code generation
        code_generation_prompt = f"""
TASK TO ACCOMPLISH:
Task ID: {task.get('id', 'unknown')}
Description: {task_description}
Type: {task_type}
Details: {task_details}
Dependencies: {task.get('dependencies', [])}

PYTHON ENVIRONMENT:
{package_context}

{doc_context}

{context_info}

Based on the task requirements, available packages, and documentation provided above, generate Python code that:

1. Accomplishes the specific task described
2. ONLY uses packages that are confirmed available in the environment
3. Uses the APIs and methods shown in the documentation
4. Handles the task dependencies appropriately
5. Includes proper error handling and progress reporting
6. Saves results in variables that can be used by subsequent tasks

For data loading tasks: Load the appropriate gravitational wave data using available packages
For processing tasks: Apply the necessary transformations or analysis using available tools
For visualization tasks: Create the requested plots using available plotting libraries

IMPORTANT: Only import and use packages that are listed in the Python environment section above. If a required package is not available, mention this in your analysis and suggest alternatives.

Please provide your analysis and code following the format specified in the system prompt.
"""
        
        print(f"[CODER AGENT] Generating code with {len(documentation)} documentation sources...")
        print(f"[CODER AGENT] Available packages: {list(self.installed_packages.keys())}")
        
        # Step 6: Get code from LLM
        llm_response = self.call_llm(code_generation_prompt)
        
        # Step 7: Parse the response
        analysis, code, explanation = self._parse_code_response(llm_response)
        
        result = {
            'task_id': task.get('id', 'unknown'),
            'task_description': task_description,
            'analysis': analysis,
            'code': code,
            'explanation': explanation,
            'documentation_used': len(documentation),
            'documentation_sources': [doc['source'] for doc in documentation],
            'available_packages': dict(self.installed_packages),
            'packages_used_in_query': len(self.installed_packages),
            'raw_response': llm_response,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"[CODER AGENT] Generated {len(code)} characters of code for task: {task.get('id', 'unknown')}")
        
        return result
    
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
        print(f"[CODER AGENT] Environment has {len(self.installed_packages)} scientific packages available")
        print("=" * 60)
        
        if not tasks:
            print("[WARNING] No tasks received from Scientific Interpreter!")
            return []
        
        results = []
        previous_results = {}
        
        for i, task in enumerate(tasks, 1):
            print(f"\n--- Processing Task {i}/{len(tasks)} ---")
            
            # Validate task structure
            if not isinstance(task, dict):
                print(f"[ERROR] Task {i} is not a valid dictionary: {task}")
                continue
            
            print(f"[CODER AGENT] Task {i} details:")
            print(f"  ID: {task.get('id', 'None')}")
            print(f"  Description: {task.get('description', 'None')}")
            print(f"  Type: {task.get('type', 'None')}")
            
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
            print(f"[CODER AGENT] Query enhanced with {result['packages_used_in_query']} available packages")
        
        return results

class ExecutorAgent:
    """
    Executor Agent that:
    1. Receives individual Python code snippets from CoderAgent
    2. Uses LLM to integrate them into a cohesive executable script
    3. Executes the integrated script and captures results
    4. Handles errors and provides execution feedback
    """
    
    def __init__(self):
        # LLM Configuration
        self.api_key = "sk-ev4v3VCbmx15mXTKC_c30w"
        self.base_url = "http://131.220.150.238:8080"
        
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # System prompt for script integration
        self.system_prompt = """You are an expert EXECUTOR AGENT specializing in integrating and executing gravitational wave analysis code. Your role is to take individual Python code snippets and combine them into a cohesive, executable script.

You will receive:
1. Multiple Python code snippets from different tasks
2. Task descriptions and dependencies
3. Context about the overall analysis workflow
4. Information about available Python packages

Your responsibilities:
1. Analyze the individual code snippets and their dependencies
2. Create proper variable flow between tasks
3. Handle imports efficiently (avoid duplicates)
4. Add error handling and progress tracking
5. Create a single, executable Python script
6. Ensure proper execution order based on task dependencies

When integrating code:
- Combine all imports at the top of the script
- Remove duplicate imports and consolidate them
- Ensure variables from one task are properly passed to dependent tasks
- Add clear section headers for each task
- Include comprehensive error handling
- Add progress print statements
- Handle file paths and data persistence appropriately
- Create meaningful variable names for intermediate results

Always structure your response as:

INTEGRATION ANALYSIS:
[Your analysis of how the tasks fit together and integration approach]

INTEGRATED SCRIPT:
```python
# Your complete integrated Python script
```

EXECUTION NOTES:
[Important notes about execution, expected outputs, and potential issues]"""
    
    def call_llm(self, prompt: str, include_history: bool = True) -> str:
        """Make API call to LLM for script integration"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history[-4:])  # Keep last 4 messages
        
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
                    "max_tokens": 6000
                },
                timeout=180
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
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"[ERROR] Exception in call_llm: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def integrate_code_snippets(self, code_results: List[Dict], original_query: str, available_packages: Dict[str, str]) -> Dict:
        """Integrate multiple code snippets into a single executable script"""
        print(f"\n[EXECUTOR AGENT] Integrating {len(code_results)} code snippets")
        print("=" * 60)
        
        if not code_results:
            print("[ERROR] No code results to integrate")
            return {
                "error": "No code results provided for integration",
                "integrated_script": "",
                "analysis": "No code snippets to integrate",
                "execution_notes": "Cannot integrate empty code list",
                "timestamp": datetime.now().isoformat()
            }
        
        # Build context about the tasks and their relationships
        task_context = self._build_task_context(code_results)
        
        # Build package context
        package_context = self._build_package_context(available_packages)
        
        # Create integration prompt
        integration_prompt = f"""
ORIGINAL USER QUERY: "{original_query}"

AVAILABLE PACKAGES:
{package_context}

TASK INTEGRATION REQUIREMENTS:
{task_context}

CODE SNIPPETS TO INTEGRATE:
"""
        
        # Add each code snippet with context
        for i, code_result in enumerate(code_results, 1):
            task_id = code_result.get('task_id', f'task_{i}')
            task_description = code_result.get('task_description', 'Unknown task')
            code = code_result.get('code', '')
            
            integration_prompt += f"""

--- Task {i}: {task_id} ---
Description: {task_description}
Code:
```python
{code}
```
"""
        
        integration_prompt += """

Based on the above code snippets and task context, create a single integrated Python script that:

1. Executes the tasks in the proper dependency order
2. Passes data between tasks appropriately
3. Handles all imports at the top (no duplicates)
4. Includes comprehensive error handling
5. Provides clear progress output
6. Saves intermediate and final results
7. Is ready to execute without modification

The integrated script should accomplish the original user query by combining all the individual task codes into a cohesive workflow.

Please provide your integration analysis, the complete integrated script, and execution notes.
"""
        
        print("[EXECUTOR AGENT] Sending integration request to LLM...")
        llm_response = self.call_llm(integration_prompt)
        
        print(f"[EXECUTOR AGENT] Received integration response ({len(llm_response)} characters)")
        
        # Parse the integration response
        analysis, integrated_script, execution_notes = self._parse_integration_response(llm_response)
        
        result = {
            "integration_analysis": analysis,
            "integrated_script": integrated_script,
            "execution_notes": execution_notes,
            "original_query": original_query,
            "tasks_integrated": len(code_results),
            "task_details": [
                {
                    "task_id": cr.get('task_id', 'unknown'),
                    "description": cr.get('task_description', 'Unknown'),
                    "code_length": len(cr.get('code', ''))
                }
                for cr in code_results
            ],
            "available_packages": available_packages,
            "raw_llm_response": llm_response,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"[EXECUTOR AGENT] Integration complete:")
        print(f"  - Integrated script length: {len(integrated_script)} characters")
        print(f"  - Tasks combined: {len(code_results)}")
        print(f"  - Analysis provided: {'Yes' if analysis else 'No'}")
        
        return result
    
    def execute_integrated_script(self, integration_result: Dict, execution_dir: str = None) -> Dict:
        """Execute the integrated script and capture results with enhanced error detection"""
        print(f"\n[EXECUTOR AGENT] Executing integrated script")
        print("=" * 60)
        
        integrated_script = integration_result.get('integrated_script', '')
        
        if not integrated_script.strip():
            print("[ERROR] No integrated script to execute")
            return {
                "error": "No integrated script provided for execution",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 0,
                "has_runtime_errors": True,
                "error_indicators": ["No script provided"],
                "timestamp": datetime.now().isoformat()
            }
        
        # Create execution directory if not provided
        if execution_dir is None:
            execution_dir = "/home/sr/Desktop/code/gravagents/garvagents_logs/executor_script"
            print(f"[EXECUTOR AGENT] Using default script directory: {execution_dir}")

        # Ensure the directory exists
        Path(execution_dir).mkdir(parents=True, exist_ok=True)
        print(f"[EXECUTOR AGENT] Using execution directory: {execution_dir}")
        
        # Write the script to a file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = Path(execution_dir) / f"integrated_analysis_{timestamp}.py"
        
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(integrated_script)
            print(f"[EXECUTOR AGENT] Script written to: {script_path}")
            
            # Execute the script
            print("[EXECUTOR AGENT] Starting script execution...")
            start_time = datetime.now()
            
            # Change to execution directory for proper relative paths
            original_cwd = os.getcwd()
            os.chdir(execution_dir)
            
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                print(f"[EXECUTOR AGENT] Execution completed in {execution_time:.2f} seconds")
                print(f"[EXECUTOR AGENT] Return code: {result.returncode}")
                
                if result.stdout:
                    print(f"[EXECUTOR AGENT] Full stdout output:")
                    print("="*60)
                    print(result.stdout)
                    print("="*60)
                if result.stderr:
                    print(f"[EXECUTOR AGENT] Full stderr output:")
                    print("="*60)
                    print(result.stderr)
                    print("="*60)
                
                # ENHANCED ERROR DETECTION
                has_runtime_errors, error_indicators = self._detect_runtime_errors(
                    result.stdout, result.stderr, result.returncode
                )
                
                # Override success if runtime errors detected
                script_success = result.returncode == 0 and not has_runtime_errors
                
                if has_runtime_errors:
                    print(f"[EXECUTOR AGENT] Runtime errors detected: {error_indicators}")
                
                execution_result = {
                    "success": script_success,  # Modified to consider runtime errors
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time,
                    "script_path": str(script_path),
                    "execution_directory": execution_dir,
                    "has_runtime_errors": has_runtime_errors,  # NEW
                    "error_indicators": error_indicators,      # NEW
                    "timestamp": datetime.now().isoformat()
                }
                
                # Check for output files in execution directory
                output_files = list(Path(execution_dir).glob("*"))
                execution_result["output_files"] = [str(f) for f in output_files if not f.name.startswith("integrated_analysis_")]
                
                return execution_result
                
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            print("[ERROR] Script execution timed out after 5 minutes")
            return {
                "error": "Script execution timed out",
                "timeout": True,
                "stdout": "",
                "stderr": "Execution timed out after 5 minutes",
                "return_code": -1,
                "execution_time": 300,
                "script_path": str(script_path),
                "has_runtime_errors": True,
                "error_indicators": ["Script timeout"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[ERROR] Exception during script execution: {str(e)}")
            return {
                "error": f"Exception during execution: {str(e)}",
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "execution_time": 0,
                "script_path": str(script_path) if 'script_path' in locals() else "",
                "has_runtime_errors": True,
                "error_indicators": [f"Execution exception: {str(e)}"],
                "timestamp": datetime.now().isoformat()
            }
        
    def _detect_runtime_errors(self, stdout: str, stderr: str, return_code: int) -> tuple:
        """
        Detect actual errors, not normal successful output
        Returns: (has_errors: bool, error_indicators: List[str])
        """
        error_indicators = []
        
        # Check for actual error keywords in stdout
        if stdout and stdout.strip():
            stdout_lower = stdout.lower()
            
            # Success indicators that mean everything is fine
            success_keywords = [
                'all tasks completed successfully',
                'completed successfully',
                'analysis complete',
                'success',
                'template bank search complete',
                'results saved successfully'
            ]
            
            # Error indicators
            error_keywords = [
                'error:',
                'exception:',
                'traceback',
                'failed',
                'critical error',
                'could not',
                'unable to'
            ]
            
            has_success = any(keyword in stdout_lower for keyword in success_keywords)
            has_error = any(keyword in stdout_lower for keyword in error_keywords)
            
            # Only flag as error if we see error keywords AND no success message
            if has_error and not has_success:
                error_indicators.append(f"stdout_output: {stdout}")
        
        # Filter stderr for actual errors (not just warnings)
        if stderr and stderr.strip():
            harmless_patterns = [
                "pkg_resources is deprecated",
                "userwarning",
                "deprecationwarning",
                "futurewarning"
            ]
            
            stderr_lower = stderr.lower()
            is_harmless = any(pattern in stderr_lower for pattern in harmless_patterns)
            
            if not is_harmless:
                error_indicators.append(f"stderr_output: {stderr}")
        
        # Non-zero return code is always an error
        if return_code != 0:
            error_indicators.append(f"exit_code: {return_code}")
        
        has_errors = len(error_indicators) > 0
        
        return has_errors, error_indicators

    def process_code_results(self, code_results: List[Dict], original_query: str, 
                           available_packages: Dict[str, str], execute: bool = True,
                           execution_dir: str = None) -> Dict:
        """Complete pipeline: integrate code snippets and optionally execute"""
        print(f"\n[EXECUTOR AGENT] Processing {len(code_results)} code results")
        print(f"[EXECUTOR AGENT] Original query: {original_query}")
        print(f"[EXECUTOR AGENT] Execute after integration: {execute}")
        
        # Step 1: Integrate code snippets
        integration_result = self.integrate_code_snippets(code_results, original_query, available_packages)
        
        if "error" in integration_result:
            print(f"[ERROR] Integration failed: {integration_result['error']}")
            return {
                "session_id": self.session_id,
                "status": "integration_failed",
                "integration_result": integration_result,
                "execution_result": None,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 2: Execute if requested
 # Step 2: Execute if requested
        execution_result = None
        debug_result = None

        if execute:
            execution_result = self.execute_integrated_script(integration_result, execution_dir)
            
            if execution_result.get("success"):
                print("[EXECUTOR AGENT] Script executed successfully!")
            else:
                print("[EXECUTOR AGENT] Script execution had issues")
                if execution_result.get("stderr"):
                    print(f"[EXECUTOR AGENT] Error output: {execution_result['stderr'][:200]}...")
                
                # Return results to allow debugger to handle errors
                # The debugging will be handled at the system level
        # Compile final result
        final_result = {
            "session_id": self.session_id,
            "status": "success" if not execution_result or execution_result.get("success") else "execution_failed",
            "original_query": original_query,
            "integration_result": integration_result,
            "execution_result": execution_result,
            "execution_requested": execute,
            "token_usage": self.total_tokens_used,
            "timestamp": datetime.now().isoformat()
        }
        
        return final_result
    
    def _build_task_context(self, code_results: List[Dict]) -> str:
        """Build context about tasks and their relationships"""
        context = f"Total tasks to integrate: {len(code_results)}\n\n"
        
        for i, code_result in enumerate(code_results, 1):
            task_id = code_result.get('task_id', f'task_{i}')
            description = code_result.get('task_description', 'Unknown task')
            analysis = code_result.get('analysis', 'No analysis provided')
            explanation = code_result.get('explanation', 'No explanation provided')
            
            context += f"Task {i} ({task_id}):\n"
            context += f"  Description: {description}\n"
            context += f"  Analysis: {analysis[:100]}...\n" if len(analysis) > 100 else f"  Analysis: {analysis}\n"
            context += f"  Expected output: {explanation[:100]}...\n" if len(explanation) > 100 else f"  Expected output: {explanation}\n"
            context += "\n"
        
        return context
    
    def _build_package_context(self, available_packages: Dict[str, str]) -> str:
        """Build context about available packages"""
        if not available_packages:
            return "No specific packages detected in environment"
        
        context = "Available packages for integration:\n"
        for package, version in available_packages.items():
            context += f"- {package} v{version}\n"
        
        return context.strip()
    
    def _parse_integration_response(self, response: str) -> tuple:
        """Parse LLM integration response into analysis, script, and notes"""
        analysis = ""
        integrated_script = ""
        execution_notes = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('integration analysis:'):
                current_section = 'analysis'
                analysis += line[22:].strip() + '\n'
            elif line_lower.startswith('integrated script:'):
                current_section = 'script'
            elif line_lower.startswith('execution notes:'):
                current_section = 'notes'
                execution_notes += line[16:].strip() + '\n'
            elif line.strip().startswith('```python'):
                current_section = 'script'
            elif line.strip().startswith('```') and current_section == 'script':
                current_section = None
            elif current_section == 'analysis':
                analysis += line + '\n'
            elif current_section == 'script' and not line.strip().startswith('```'):
                integrated_script += line + '\n'
            elif current_section == 'notes':
                execution_notes += line + '\n'
        
        return analysis.strip(), integrated_script.strip(), execution_notes.strip()

class DebuggerAgent:
    """
    Debugger Agent that:
    1. Catches execution errors from ExecutorAgent
    2. Analyzes error messages and failed code
    3. Uses LLM to generate fixes
    4. Asks user permission before each retry attempt
    5. Loops until code executes successfully or user terminates
    """
    
    def __init__(self,database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        # LLM Configuration
        self.api_key = "sk-ev4v3VCbmx15mXTKC_c30w"
        self.base_url = "http://131.220.150.238:8080"

        self.database_path = database_path
        self.client = None
        self.collection = None
        self._initialize_chromadb()
        
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_attempt_count = 0
        self.max_debug_attempts = 5
        
        # System prompt for code debugging
        self.system_prompt = """You are an expert DEBUGGER AGENT specializing in fixing gravitational wave analysis code. Your role is to analyze Python execution errors and provide corrected code.

You will receive:
1. Failed Python code that caused an error
2. Complete error message and traceback
3. Context about what the code was trying to accomplish
4. Information about available Python packages

Your responsibilities:
1. Analyze the error message and identify the root cause
2. Understand what the code was intended to do
3. Fix the specific issues causing the failure
4. Ensure the fix maintains the original functionality
5. Only use packages that are confirmed to be available
6. Provide robust error handling to prevent similar failures

When fixing code:
- Focus on the specific error reported
- Maintain the original code structure and logic where possible
- Add appropriate error handling (try/except blocks)
- Use proper imports and package versions
- Test for common edge cases (network timeouts, missing data, etc.)
- Add fallback mechanisms when appropriate
- Include progress reporting and status messages

Always structure your response as:

ERROR ANALYSIS:
[Your analysis of what went wrong and why]

FIXED CODE:
```python
# Your corrected implementation
```

EXPLANATION:
[Brief explanation of the fixes applied and why they should work]"""
    
    def _initialize_chromadb(self):
        """Initialize connection to ChromaDB database"""
        try:
            self.client = chromadb.PersistentClient(path=self.database_path)
            collections = self.client.list_collections()
            if collections:
                collection_names = [c.name for c in collections]
                print(f"[DEBUGGER CHROMADB] Available collections: {collection_names}")
                
                # Look for documentation collection
                target_names = ['gw_comprehensive_docs', 'gravitational_wave_documentation', 'code_documentation']
                for name in target_names:
                    if name in collection_names:
                        self.collection = self.client.get_collection(name)
                        print(f"[DEBUGGER CHROMADB] Connected to collection: {name}")
                        break
                
                if not self.collection and collection_names:
                    self.collection = self.client.get_collection(collection_names[0])
                    print(f"[DEBUGGER CHROMADB] Using collection: {collection_names[0]}")
            else:
                print("[DEBUGGER CHROMADB] No collections found")
        except Exception as e:
            print(f"[DEBUGGER CHROMADB] Could not connect: {e}")
            self.client = None
            self.collection = None
    
    def query_documentation(self, query: str, n_results: int = 3) -> List[Dict]:
        """Query ChromaDB for relevant documentation"""
        if not self.collection:
            print("[DEBUGGER] No ChromaDB collection available")
            return []
        
        try:
            print(f"[DEBUGGER] Querying documentation for: {query}")
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0] if results['metadatas'][0] else [{}] * len(results['documents'][0])):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'title': metadata.get('title', 'Unknown') if metadata else 'Unknown'
                    })
            
            print(f"[DEBUGGER] Found {len(formatted_results)} relevant documents")
            return formatted_results
        except Exception as e:
            print(f"[DEBUGGER] Error querying documentation: {e}")
            return []

    def call_llm(self, prompt: str, include_history: bool = True) -> str:
        """Make API call to LLM for code debugging"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history[-4:])  # Keep last 4 messages
        
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
                    "max_tokens": 6000
                },
                timeout=90
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
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"[ERROR] Exception in call_llm: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def get_user_permission(self, attempt_count: int, error_summary: str, debug_history: List[Dict] = None) -> bool:
        """Ask user for permission to continue debugging with learning context"""
        print(f"\n{'='*60}")
        print(f"DEBUGGER AGENT - ATTEMPT {attempt_count}")
        print(f"{'='*60}")
        print(f"Error encountered: {error_summary}")
        print(f"Debug attempts so far: {attempt_count}")
        print(f"Maximum attempts allowed: {self.max_debug_attempts}")
        
        # Show learning context
        if debug_history and len(debug_history) > 0:
            print(f"\nLEARNING CONTEXT:")
            print(f"Previous attempts: {len(debug_history)}")
            
            # Show what approaches were tried
            for i, attempt in enumerate(debug_history[-2:], max(1, len(debug_history)-1)):  # Show last 2 attempts
                print(f"  Attempt {i}: {attempt.get('explanation', 'Unknown approach')[:60]}...")
        
        print(f"{'='*60}")
        
        while True:
            user_choice = input("Do you want the Debugger Agent to attempt a fix? (y/n/details): ").strip().lower()
            
            if user_choice in ['y', 'yes']:
                return True
            elif user_choice in ['n', 'no']:
                print("User chose to terminate debugging session.")
                return False
            elif user_choice in ['d', 'details', 'detail']:
                print(f"\nDETAILED ERROR INFORMATION:")
                print(f"Attempt: {attempt_count}/{self.max_debug_attempts}")
                print(f"Error type: {error_summary}")
                
                if debug_history:
                    print(f"\nPrevious attempts summary:")
                    for i, attempt in enumerate(debug_history, 1):
                        print(f"{i}. {attempt.get('explanation', 'No explanation')}")
                
                print("The Debugger Agent will analyze the error and attempt to fix the code.")
                print("You can choose to continue, stop, or see these details again.\n")
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'details' for more information.")
    
    def analyze_and_fix_error(self, failed_code: str, error_message: str, 
                         stderr_output: str, stdout_output: str, original_query: str, 
                         available_packages: Dict[str, str], debug_history: List[Dict] = None) -> Dict:
        """Two-phase error analysis: first understand, then fix"""
        
        print(f"\n[DEBUGGER AGENT] Starting two-phase error analysis...")
        
        # PHASE 1: Error Understanding
        combined_output = self._combine_execution_output(stdout_output, stderr_output, error_message)
        error_summary = self._analyze_error_context(combined_output, debug_history)
        
        # PHASE 2: Code Fix Generation  
        fix_result = self._generate_targeted_fix_v2(failed_code, error_summary, 
                                                original_query, available_packages)
        
        return {
            "debug_attempt": self.debug_attempt_count,
            "original_error": error_message,
            "error_analysis": error_summary.get("analysis_response", ""),
            "fixed_code": fix_result.get("fixed_code", ""),
            "explanation": fix_result.get("explanation", ""),
            "original_query": original_query,
            "available_packages": available_packages,
            "previous_attempts_analyzed": len(debug_history) if debug_history else 0,
            "timestamp": datetime.now().isoformat()
        }

    def _combine_execution_output(self, stdout: str, stderr: str, error_msg: str) -> str:
        """Combine all execution information chronologically"""
        combined = "=== COMPLETE EXECUTION ANALYSIS ===\n\n"
        
        if stdout and stdout.strip():
            combined += "STDOUT OUTPUT:\n"
            combined += stdout
            combined += "\n\n"
        
        if stderr and stderr.strip():
            combined += "STDERR OUTPUT:\n" 
            combined += stderr
            combined += "\n\n"
        
        if error_msg:
            combined += f"ERROR MESSAGE: {error_msg}\n\n"
        
        return combined

    def _analyze_error_context(self, combined_output: str, debug_history: List[Dict]) -> Dict:
            """Phase 1: Pure error analysis - no code generation"""
            
            previous_context = ""
            if debug_history:
                previous_context = f"\n=== PREVIOUS FAILED ATTEMPTS ===\n"
                for i, attempt in enumerate(debug_history, 1):
                    previous_context += f"Attempt {i}: {attempt.get('explanation', 'Unknown')}\n"
                previous_context += "=== END PREVIOUS ATTEMPTS ===\n"
            
            analysis_prompt = f"""
        You are analyzing a Python execution failure. Your ONLY job is to understand what went wrong.
        DO NOT generate any code fixes - just analyze the problem.

        {combined_output}

        {previous_context}

        Analyze this execution failure and provide:

        ERROR CLASSIFICATION:
        [What type of error is this: API misuse, logic error, data issue, environment problem, etc.]

        ROOT CAUSE:
        [The fundamental reason this error occurred - be specific about the exact line/operation that failed]

        ERROR PROGRESSION:
        [How the execution progressed and where exactly it failed - trace the sequence]

        KEY INDICATORS:
        [The most important clues from the output that point to the solution]

        FAILED OPERATION:
        [The specific function/method call that caused the failure]

        PATTERN ANALYSIS:
        [If this is a repeat failure, what pattern do you see?]

        Remember: NO CODE FIXES - just detailed understanding of what went wrong.
        """
            
            print("[DEBUGGER] Phase 1: Analyzing error context...")
            analysis_response = self.call_llm(analysis_prompt, include_history=False)
            
            return {
                "analysis_response": analysis_response,
                "combined_output_length": len(combined_output),
                "phase": "error_analysis"
            }

    def _generate_targeted_fix_v2(self, failed_code: str, error_summary: Dict, 
                                original_query: str, available_packages: Dict[str, str]) -> Dict:
            """Phase 2: Generate fix based on clear error understanding"""
            
            error_analysis = error_summary.get("analysis_response", "")
            
            fix_prompt = f"""
        Based on the detailed error analysis below, generate a corrected version of the code.

        DETAILED ERROR ANALYSIS:
        {error_analysis}

        FAILED CODE:
        ```python
        {failed_code}
        AVAILABLE PACKAGES:
        {self._build_package_context(available_packages)}
        Based on the error analysis above, generate a corrected version that:

        Addresses the ROOT CAUSE identified in the analysis
        Fixes the FAILED OPERATION mentioned
        Uses the correct APIs for available packages
        Includes robust error handling

        FIXED CODE:
        python# Your corrected implementation
        EXPLANATION:
        [Brief explanation of the specific changes made to address the root cause]
        """
            print("[DEBUGGER] Phase 2: Generating targeted fix...")
            fix_response = self.call_llm(fix_prompt, include_history=False)

        # Parse response
            _, fixed_code, explanation = self._parse_debug_response(fix_response)

            return {
            "fixed_code": fixed_code,
            "explanation": explanation,
            "based_on_analysis": error_analysis[:200] + "...",
            "phase": "code_generation"
            }
    

    def debug_execution_loop(self, integration_result: Dict, execution_result: Dict,
                        original_query: str, available_packages: Dict[str, str],
                        executor_agent) -> Dict:
            """Main debugging loop that continues until success or user termination"""
            print(f"\n[DEBUGGER AGENT] Starting debug loop for failed execution")
            print("=" * 60)
            
            current_code = integration_result.get('integrated_script', '')
            debug_history = []
            
            while self.debug_attempt_count < self.max_debug_attempts:
                self.debug_attempt_count += 1
                
                # ENHANCED ERROR EXTRACTION
                error_message = execution_result.get('error', 'Unknown error')
                stderr_output = execution_result.get('stderr', '')
                stdout_output = execution_result.get('stdout', '')
                error_indicators = execution_result.get('error_indicators', [])
                
                # Combine all error information
                full_error_context = f"""
        Return Code: {execution_result.get('return_code', 'Unknown')}
        Error Message: {error_message}
        Error Indicators: {', '.join(error_indicators)}
        STDERR Output: {stderr_output}
        STDOUT Output (last 500 chars): {stdout_output[-500:] if stdout_output else 'No output'}
                """.strip()
                
                # Create error summary for user
                error_summary = self._create_enhanced_error_summary(
                    error_message, stderr_output, stdout_output, error_indicators
                )
                
                # Ask user permission to continue with learning context
                if not self.get_user_permission(self.debug_attempt_count, error_summary, debug_history):
                    print("[DEBUGGER AGENT] User terminated debugging session")
                    return {
                        "status": "user_terminated",
                        "debug_attempts": self.debug_attempt_count,
                        "debug_history": debug_history,
                        "final_result": execution_result,
                        "timestamp": datetime.now().isoformat()
                    }
                
                
                # Analyze error and generate fix WITH LEARNING CONTEXT
                debug_result = self.analyze_and_fix_error(
                    current_code, error_message, stderr_output, stdout_output,
                    original_query, available_packages, debug_history
                )
                
                debug_history.append(debug_result)
                
                # Update integration result with fixed code
                updated_integration = dict(integration_result)
                updated_integration['integrated_script'] = debug_result['fixed_code']
                updated_integration['debug_info'] = {
                    'attempt': self.debug_attempt_count,
                    'previous_error': error_summary,
                    'fix_applied': debug_result['explanation']
                }
                
                print(f"\n[DEBUGGER AGENT] Attempting execution with fix #{self.debug_attempt_count}")
                
                # Re-execute with fixed code
                try:
                    execution_result = executor_agent.execute_integrated_script(updated_integration)
                    
                    # Check for success (no return code errors AND no runtime errors)
                    if (execution_result.get('success') and 
                        not execution_result.get('has_runtime_errors', False)):
                        print(f"[DEBUGGER AGENT] ✓ Execution successful after {self.debug_attempt_count} debug attempts!")
                        return {
                            "status": "debug_success",
                            "debug_attempts": self.debug_attempt_count,
                            "debug_history": debug_history,
                            "final_integration_result": updated_integration,
                            "final_execution_result": execution_result,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        print(f"[DEBUGGER AGENT] ✗ Attempt {self.debug_attempt_count} failed, analyzing next error...")
                        current_code = debug_result['fixed_code']
                        
                except Exception as e:
                    print(f"[DEBUGGER AGENT] Exception during re-execution: {str(e)}")
                    execution_result = {
                        "success": False,
                        "error": f"Exception during debug execution: {str(e)}",
                        "stderr": str(e),
                        "return_code": -1,
                        "has_runtime_errors": True,
                        "error_indicators": [f"Debug execution exception: {str(e)}"]
                    }
                    current_code = debug_result['fixed_code']
            
            # Maximum attempts reached
            print(f"[DEBUGGER AGENT] Maximum debug attempts ({self.max_debug_attempts}) reached")
            return {
                "status": "max_attempts_reached",
                "debug_attempts": self.debug_attempt_count,
                "debug_history": debug_history,
                "final_result": execution_result,
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_failure_patterns(self, debug_history: List[Dict]) -> str:
        """Analyze patterns in previous failures to suggest different approaches"""
        if not debug_history or len(debug_history) < 2:
            return ""
        
        # Look for repeating patterns
        error_types = [attempt.get('original_error', '') for attempt in debug_history]
        explanations = [attempt.get('explanation', '') for attempt in debug_history]
        
        pattern_analysis = "\n=== FAILURE PATTERN ANALYSIS ===\n"
        
        # Check if same error keeps recurring
        if len(set(error_types)) == 1:
            pattern_analysis += f"PATTERN: Same error recurring {len(error_types)} times: '{error_types[0]}'\n"
            pattern_analysis += "RECOMMENDATION: Try completely different libraries or approaches\n"
        
        # Check for similar solution approaches
        common_keywords = ['fix', 'handle', 'try', 'catch', 'import', 'install']
        approach_patterns = []
        
        for exp in explanations:
            exp_lower = exp.lower()
            keywords_found = [kw for kw in common_keywords if kw in exp_lower]
            approach_patterns.append(keywords_found)
        
        if len(debug_history) >= 2:
            pattern_analysis += f"APPROACHES TRIED: {len(debug_history)} different fixes attempted\n"
            pattern_analysis += "RECOMMENDATION: Consider simplifying the entire approach\n"
        
        pattern_analysis += "=== END PATTERN ANALYSIS ===\n"
        return pattern_analysis
    
    def _create_error_summary(self, error_message: str, stderr_output: str) -> str:
        """Create a brief error summary for user display"""
        if "TimeoutError" in error_message or "timeout" in stderr_output.lower():
            return "Network timeout during data download"
        elif "ImportError" in error_message or "ModuleNotFoundError" in error_message:
            return "Missing Python package or import error"
        elif "FileNotFoundError" in error_message:
            return "File or directory not found"
        elif "KeyError" in error_message:
            return "Data structure or dictionary key error"
        elif "AttributeError" in error_message:
            return "Object attribute or method error"
        elif "ConnectionError" in error_message:
            return "Network connection error"
        else:
            # Extract the main error type
            error_lines = stderr_output.split('\n') if stderr_output else [error_message]
            for line in reversed(error_lines):
                if line.strip() and ':' in line:
                    return line.strip()[:100] + "..." if len(line) > 100 else line.strip()
            return error_message[:100] + "..." if len(error_message) > 100 else error_message
    
    def _build_package_context(self, available_packages: Dict[str, str]) -> str:
        """Build context about available packages"""
        if not available_packages:
            return "No specific packages detected in environment"
        
        context = "Available packages:\n"
        for package, version in available_packages.items():
            context += f"- {package} v{version}\n"
        
        return context.strip()
    

    def _create_enhanced_error_summary(self, error_message: str, stderr_output: str, 
                                 stdout_output: str, error_indicators: List[str]) -> str:
        """Create error summary from any available information"""
        
        # Just return the most relevant available information
        if stderr_output and stderr_output.strip():
            return f"Script errors detected in output"
        elif stdout_output and "error" in stdout_output.lower():
            return f"Errors found in execution output" 
        elif error_indicators:
            return f"Script execution issues detected"
        else:
            return f"Unknown execution problem"
    
    def _parse_debug_response(self, response: str) -> tuple:
        """Parse LLM debug response into analysis, fixed code, and explanation"""
        error_analysis = ""
        fixed_code = ""
        explanation = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('error analysis:'):
                current_section = 'analysis'
                error_analysis += line[15:].strip() + '\n'
            elif line_lower.startswith('fixed code:'):
                current_section = 'code'
            elif line_lower.startswith('explanation:'):
                current_section = 'explanation'
                explanation += line[12:].strip() + '\n'
            elif line.strip().startswith('```python'):
                current_section = 'code'
            elif line.strip().startswith('```') and current_section == 'code':
                current_section = None
            elif current_section == 'analysis':
                error_analysis += line + '\n'
            elif current_section == 'code' and not line.strip().startswith('```'):
                fixed_code += line + '\n'
            elif current_section == 'explanation':
                explanation += line + '\n'
        
        return error_analysis.strip(), fixed_code.strip(), explanation.strip()
        
class IntegratedGravitationalWaveSystem:
    """
    Integrated system that combines Scientific Interpreter, CODER AGENT, and Executor agents
    Complete pipeline: Query → Task Planning → Code Generation → Script Integration → Execution
    """
    
    def __init__(self, database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        self.scientific_interpreter = ScientificInterpreterAgent()
        self.data_analyst = CoderAgent(database_path)
        self.executor = ExecutorAgent()
        self.debugger = DebuggerAgent(database_path)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("INTEGRATED GRAVITATIONAL WAVE ANALYSIS SYSTEM (3-Agent Pipeline)")
        print("=" * 70)
        print(f"Session ID: {self.session_id}")
        print(f"Scientific Interpreter: Ready (LLM Knowledge-Based)")
        print(f"CODER AGENT: Ready (ChromaDB: {'Connected' if hasattr(self.data_analyst, 'collection') and self.data_analyst.collection else 'Not Connected'})")
        print(f"Executor Agent: Ready")
        print(f"Debugger Agent: Ready")
        if hasattr(self.data_analyst, 'installed_packages'):
            print(f"Available Scientific Packages: {len(self.data_analyst.installed_packages)}")
            if self.data_analyst.installed_packages:
                for pkg, version in self.data_analyst.installed_packages.items():
                    print(f"  - {pkg} v{version}")
    
    def process_query_with_execution(self, user_query: str, execute_script: bool = True, 
                                   execution_dir: str = None) -> Dict:
        """
        Complete 3-agent pipeline: 
        Query → Scientific Interpreter → CODER AGENT → Executor Agent → Results
        """
        print(f"\n[SYSTEM] Processing query with 3-agent pipeline: {user_query}")
        print("=" * 90)
        
        # Step 1: Scientific Interpreter (same as before)
        print("\nSTEP 1: SCIENTIFIC INTERPRETATION")
        print("-" * 40)
        interpretation_result = self.scientific_interpreter.interpret_query(user_query)
        
        if "error" in interpretation_result and not interpretation_result.get('tasks'):
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
        
        # Step 2: CODER AGENT (same as before)
        print(f"\nSTEP 2: CODE GENERATION")
        print("-" * 40)
        
        try:
            code_results = self.data_analyst.process_task_list(tasks)
            if not code_results:
                return {
                    "session_id": self.session_id,
                    "error": "No code results generated",
                    "interpretation": interpretation_result,
                    "timestamp": datetime.now().isoformat()
                }
            print(f"[SYSTEM] Generated code for {len(code_results)} tasks")
        except Exception as e:
            return {
                "session_id": self.session_id,
                "error": f"CODER AGENT processing failed: {str(e)}",
                "interpretation": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 3: Executor Agent (NEW)
        print(f"\nSTEP 3: SCRIPT INTEGRATION & EXECUTION")
        print("-" * 40)
        
        try:
            execution_result = self.executor.process_code_results(
                code_results, 
                user_query, 
                getattr(self.data_analyst, 'installed_packages', {}),
                execute=execute_script,
                execution_dir=execution_dir
            )
            
            print(f"[SYSTEM] Executor completed with status: {execution_result.get('status', 'unknown')}")

            # Step 4: Debugger Agent (NEW) - Handle execution failures
            # Step 4: Debugger Agent - Always check execution output
            debug_result = None
            exec_result = execution_result.get('execution_result')

            # Simple trigger - if we have any output, let debugger analyze it
            if execute_script and exec_result:
                stdout = exec_result.get('stdout', '')
                stderr = exec_result.get('stderr', '')
                return_code = exec_result.get('return_code', 0)
                
                # Trigger debugger if:
                # 1. Non-zero exit code, OR
                # 2. Any stderr output (except harmless warnings), OR  
                # 3. Return code 0 but we want LLM to verify if stdout indicates success
                should_debug = (
                    return_code != 0 or 
                    (stderr and not all(pattern in stderr.lower() for pattern in ["pkg_resources", "warning"])) or
                    (return_code == 0 and stdout)  # Let LLM check if stdout indicates real success
                )
                
                if should_debug:
                    print(f"\nSTEP 4: ANALYZING EXECUTION OUTPUT")
                    print("-" * 40)
                    
                    try:
                        # Always pass ALL available information to debugger
                        debug_result = self.debugger.debug_execution_loop(
                            execution_result['integration_result'],
                            execution_result['execution_result'], 
                            user_query,
                            getattr(self.data_analyst, 'installed_packages', {}),
                            self.executor
                        )
                        
                        print(f"[SYSTEM] Debugger completed with status: {debug_result.get('status', 'unknown')}")
                        
                        if debug_result.get('status') == 'debug_success':
                            execution_result['integration_result'] = debug_result['final_integration_result']
                            execution_result['execution_result'] = debug_result['final_execution_result']
                            execution_result['status'] = 'success'
                        
                    except Exception as e:
                        print(f"[ERROR] Exception in Debugger Agent: {str(e)}")
                        debug_result = {
                            "status": "debugger_error", 
                            "error": str(e),
                            "debug_attempts": 0
                        }
            
        except Exception as e:
            print(f"[ERROR] Exception in Executor Agent: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "session_id": self.session_id,
                "error": f"Executor Agent processing failed: {str(e)}",
                "interpretation": interpretation_result,
                "code_results": code_results,
                "timestamp": datetime.now().isoformat()
            }
        
        # Compile final results
        final_result = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "original_query": user_query,
            "status": execution_result.get('status', 'unknown'),
            "pipeline_complete": True,
            "scientific_interpretation": {
                "understanding": interpretation_result.get('understanding', ''),
                "knowledge_context": interpretation_result.get('knowledge_context', ''),
                "scientific_context": interpretation_result.get('scientific_context', ''),
                "expected_outcomes": interpretation_result.get('expected_outcomes', ''),
                "tasks_generated": len(tasks)
            },
            "code_generation": {
                "tasks_processed": len(code_results),
                "total_documentation_sources": sum(r.get('documentation_used', 0) for r in code_results),
                "code_results": code_results
            },
            "script_execution": execution_result,
            "debug_session": debug_result,
            "token_usage": {
                "scientific_interpreter": self.scientific_interpreter.total_tokens_used,
                "data_analyst": getattr(self.data_analyst, 'total_tokens_used', 0),
                "executor": self.executor.total_tokens_used,
                "debugger": self.debugger.total_tokens_used,
                "total": (self.scientific_interpreter.total_tokens_used + 
                        getattr(self.data_analyst, 'total_tokens_used', 0) + 
                        self.executor.total_tokens_used + 
                        self.debugger.total_tokens_used)
            }
                    }
        
        return final_result
    
    # Keep existing methods for backward compatibility
    def process_query(self, user_query: str) -> Dict:
        """Original 2-agent pipeline for backward compatibility"""
        return self.process_query_with_execution(user_query, execute_script=False)
    
    def save_session(self, result: Dict, output_dir: str = "/home/sr/Desktop/code/gravagents/garvagents_logs/integrated_results") -> str:
        """Save complete session results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filename = f"gw_analysis_session_{result['session_id']}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return str(filepath)
    
    def get_system_status(self) -> str:
        """Get status of all three agents"""
        return f"""
INTEGRATED GRAVITATIONAL WAVE ANALYSIS SYSTEM STATUS (3-Agent Pipeline)
Session: {self.session_id}

Scientific Interpreter:
- Mode: LLM Knowledge-Based (No Web Search)
- Tokens used: {self.scientific_interpreter.total_tokens_used}

Coder Agent:
- ChromaDB: {'Connected' if hasattr(self.data_analyst, 'collection') and self.data_analyst.collection else 'Not Connected'}
- Available packages: {len(getattr(self.data_analyst, 'installed_packages', {}))}
- Tokens used: {getattr(self.data_analyst, 'total_tokens_used', 0)}

Executor Agent:
- Status: Ready
- Tokens used: {self.executor.total_tokens_used}

Debugger Agent:
- Status: Ready
- Tokens used: {self.debugger.total_tokens_used}

Available Scientific Packages:
{chr(10).join(f'- {pkg} v{ver}' for pkg, ver in getattr(self.data_analyst, 'installed_packages', {}).items()) if hasattr(self.data_analyst, 'installed_packages') and self.data_analyst.installed_packages else '- None detected'}

Total tokens used: {self.scientific_interpreter.total_tokens_used + getattr(self.data_analyst, 'total_tokens_used', 0) + self.executor.total_tokens_used + self.debugger.total_tokens_used}
"""
def main():
    """Main interactive interface with a simplified 3-agent pipeline."""
    
    # Initialize integrated system
    system = IntegratedGravitationalWaveSystem()
    
    print(f"\n{'='*60}")
    print("GRAVAGENT: 3-AGENT EXECUTION MODE")
    print("All queries will be processed by the 3-agent pipeline.")
    print("Type 'quit' to exit.")
    print("="*60)
    
    while True:
        print(f"\n{'-'*60}")
        user_input = input("Enter gravitational wave analysis query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a query or command.")
            continue
            
        try:
            # Always process with the 3-agent pipeline including execution
            result = system.process_query_with_execution(user_input, execute_script=True)
            
            # Display results
            print(f"\n{'='*80}")
            print("3-AGENT PIPELINE ANALYSIS COMPLETE")
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
                
                # Display code generation results
                code_gen = result['code_generation']
                print(f"\nCode generation:")
                print(f"Tasks processed: {code_gen['tasks_processed']}")
                print(f"Documentation sources used: {code_gen['total_documentation_sources']}")
                
                # Display ExecutorAgent results
                if 'script_execution' in result:
                    execution = result['script_execution']
                    print(f"\nScript Integration & Execution:")
                    print(f"Status: {execution.get('status', 'unknown')}")
                    
                    if execution.get('integration_result'):
                        integration = execution['integration_result']
                        print(f"Integrated script length: {len(integration.get('integrated_script', ''))} characters")
                        print(f"Tasks integrated: {integration.get('tasks_integrated', 0)}")
                    
                    if execution.get('execution_result'):
                        exec_result = execution['execution_result']
                        if exec_result.get('success'):
                            print(f"✓ Script executed successfully in {exec_result.get('execution_time', 0):.2f} seconds")
                            print(f"Script saved to: {exec_result.get('script_path', 'unknown')}")
                            if exec_result.get('output_files'):
                                print(f"Generated files: {len(exec_result['output_files'])}")
                                for file_path in exec_result['output_files'][:5]:  # Show first 5 files
                                    print(f"  - {file_path}")
                            if exec_result.get('stdout'):
                                print(f"\nExecution Output Preview:")
                                print("-" * 40)
                                stdout_preview = exec_result['stdout'][:500]
                                print(stdout_preview + "..." if len(exec_result['stdout']) > 500 else stdout_preview)
                                print("-" * 40)
                        else:
                            print(f"✗ Script execution failed")
                            if exec_result.get('stderr'):
                                print(f"Error: {exec_result['stderr'][:300]}...")
                
                # Display token usage
                if 'debug_session' in result and result['debug_session']:
                    debug_info = result['debug_session']
                    print(f"\nDebugging Session:")
                    print(f"Status: {debug_info.get('status', 'unknown')}")
                    print(f"Debug attempts: {debug_info.get('debug_attempts', 0)}")
                    
                    if debug_info.get('status') == 'debug_success':
                        print("✓ Code execution successful after debugging")
                    elif debug_info.get('status') == 'user_terminated':
                        print("✗ User terminated debugging session")
                    elif debug_info.get('status') == 'max_attempts_reached':
                        print("✗ Maximum debug attempts reached")

                # Display token usage  
                tokens = result['token_usage']
                print(f"\nToken Usage:")
                print(f"Scientific Interpreter: {tokens.get('scientific_interpreter', 0)}")
                print(f"CODER AGENT: {tokens.get('data_analyst', 0)}")
                if 'executor' in tokens:
                    print(f"Executor Agent: {tokens['executor']}")
                if 'debugger' in tokens:
                    print(f"Debugger Agent: {tokens['debugger']}")
                print(f"Total: {tokens.get('total', 0)}")
                            
            # Save session
            saved_path = system.save_session(result)
            print(f"\nSession saved to: {saved_path}")
            
        except Exception as e:
            print(f"Error processing query with execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 