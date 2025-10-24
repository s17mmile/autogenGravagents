import json
import requests
import chromadb
import subprocess
import sys
import pkg_resources
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

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

        self.api_key = "sk-28n2AKk-2GxPyO-szIZCsw"
        self.base_url = "http://131.220.150.238:8080"
    
        self.llm_config = {
            "model": "openai/gpt-4o-mini",
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
    
    def _get_installed_scientific_packages(self) -> Dict[str, str]:
        """Check which scientific packages are installed and get their versions"""
        installed_packages = {}
        
        print(f"[PACKAGE CHECK] Checking for scientific packages...")
        
        for package_name in self.SCIENTIFIC_PACKAGES:
            try:
                # Try using pkg_resources first (more reliable for version info)
                try:
                    distribution = pkg_resources.get_distribution(package_name)
                    installed_packages[package_name] = distribution.version
                    print(f"[PACKAGE CHECK] ✓ {package_name} v{distribution.version}")
                except pkg_resources.DistributionNotFound:
                    # Try direct import as fallback
                    try:
                        module = __import__(package_name)
                        version = getattr(module, '__version__', 'unknown')
                        installed_packages[package_name] = version
                        print(f"[PACKAGE CHECK] ✓ {package_name} v{version}")
                    except ImportError:
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
    
    def query_documentation(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the ChromaDB vector database for relevant documentation with package context"""
        if not self.collection:
            print("[WARNING] No ChromaDB collection available, returning empty results")
            return []
        
        try:
            # Enhance query with package information
            package_context = self._build_package_context()
            enhanced_query = f"{query}\n\nContext: {package_context}"
            
            print(f"[CHROMADB] Querying documentation for: {query}")
            print(f"[CHROMADB] With package context: {len(self.installed_packages)} packages available")
            
            results = self.collection.query(
                query_texts=[enhanced_query],
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


class IntegratedGravitationalWaveSystem:
    """
    Integrated system that combines Scientific Interpreter and CODER AGENT agents
    Now relies on LLM knowledge instead of web search
    """
    
    def __init__(self, database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        self.scientific_interpreter = ScientificInterpreterAgent()
        self.data_analyst = CoderAgent(database_path)  # Assuming CoderAgent class exists
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("INTEGRATED GRAVITATIONAL WAVE ANALYSIS SYSTEM (Knowledge-Based)")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Scientific Interpreter: Ready (LLM Knowledge-Based)")
        print(f"CODER AGENT: Ready (ChromaDB: {'Connected' if hasattr(self.data_analyst, 'collection') and self.data_analyst.collection else 'Not Connected'})")
        if hasattr(self.data_analyst, 'installed_packages'):
            print(f"Available Scientific Packages: {len(self.data_analyst.installed_packages)}")
            if self.data_analyst.installed_packages:
                for pkg, version in self.data_analyst.installed_packages.items():
                    print(f"  - {pkg} v{version}")
    
    def process_query(self, user_query: str) -> Dict:
        """
        Full pipeline: User query → Scientific Interpreter (LLM knowledge) → CODER AGENT → Results
        """
        print(f"\n[SYSTEM] Processing query: {user_query}")
        print("=" * 80)
        
        # Step 1: Scientific Interpreter breaks down the query using LLM knowledge
        print("\nSTEP 1: SCIENTIFIC INTERPRETATION (LLM Knowledge-Based)")
        print("-" * 40)
        interpretation_result = self.scientific_interpreter.interpret_query(user_query)
        
        # Check for interpretation errors
        if "error" in interpretation_result and not interpretation_result.get('tasks'):
            print(f"[ERROR] Scientific interpretation failed completely: {interpretation_result['error']}")
            return {
                "session_id": self.session_id,
                "error": "Scientific interpretation failed",
                "details": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        # Extract tasks
        tasks = interpretation_result.get('tasks', [])
        if not tasks:
            print("[ERROR] No tasks generated from query")
            return {
                "session_id": self.session_id,
                "error": "No tasks generated from query",
                "interpretation": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"[SYSTEM] Successfully generated {len(tasks)} tasks using LLM knowledge")
        for i, task in enumerate(tasks, 1):
            print(f"  Task {i}: {task.get('id', 'unknown')} - {task.get('description', 'No description')[:50]}...")
        
        # Step 2: CODER AGENT generates code for tasks
        print(f"\nSTEP 2: DATA ANALYSIS & CODE GENERATION")
        print("-" * 40)
        
        try:
            if hasattr(self.data_analyst, 'process_task_list'):
                code_results = self.data_analyst.process_task_list(tasks)
            else:
                print("[ERROR] CoderAgent not properly initialized")
                code_results = []
            
            if not code_results:
                print("[ERROR] No code results generated")
                return {
                    "session_id": self.session_id,
                    "error": "No code results generated by CODER AGENT",
                    "interpretation": interpretation_result,
                    "timestamp": datetime.now().isoformat()
                }
            
            print(f"[SYSTEM] Successfully generated code for {len(code_results)} tasks")
            
        except Exception as e:
            print(f"[ERROR] Exception in CODER AGENT processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "session_id": self.session_id,
                "error": f"CODER AGENT processing failed: {str(e)}",
                "interpretation": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 3: Compile final results
        final_result = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "original_query": user_query,
            "status": "success",
            "knowledge_based": True,  # Flag indicating LLM knowledge was used
            "scientific_interpretation": {
                "understanding": interpretation_result.get('understanding', ''),
                "knowledge_context": interpretation_result.get('knowledge_context', ''),
                "scientific_context": interpretation_result.get('scientific_context', ''),
                "expected_outcomes": interpretation_result.get('expected_outcomes', ''),
                "tasks_generated": len(tasks),
                "has_error": "error" in interpretation_result
            },
            "data_analysis": {
                "tasks_processed": len(code_results),
                "total_documentation_sources": sum(r.get('documentation_used', 0) for r in code_results) if code_results else 0,
                "available_packages": getattr(self.data_analyst, 'installed_packages', {}),
                "code_results": code_results
            },
            "token_usage": {
                "scientific_interpreter": self.scientific_interpreter.total_tokens_used,
                "data_analyst": getattr(self.data_analyst, 'total_tokens_used', 0),
                "total": self.scientific_interpreter.total_tokens_used + getattr(self.data_analyst, 'total_tokens_used', 0)
            }
        }
        
        # Add any interpretation errors to final result
        if "error" in interpretation_result:
            final_result["interpretation_warning"] = interpretation_result["error"]
            final_result["raw_interpretation_response"] = interpretation_result.get("raw_response", "")
        
        return final_result
    
    def save_session(self, result: Dict, output_dir: str = "/home/sr/Desktop/code/gravagents/garvagents_logs/integrated_results") -> str:
        """Save complete session results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filename = f"gw_analysis_session_{result['session_id']}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return str(filepath)
    
    def debug_task_flow(self, user_query: str) -> Dict:
        """Debug version that shows detailed task flow"""
        print(f"\n[DEBUG MODE] Processing query: {user_query}")
        print("=" * 80)
        
        # Step 1: Scientific Interpreter
        print("\n[DEBUG] STEP 1: Scientific Interpreter")
        print("-" * 50)
        interpretation_result = self.scientific_interpreter.interpret_query(user_query)
        
        print(f"[DEBUG] Interpretation result keys: {list(interpretation_result.keys())}")
        print(f"[DEBUG] Has error: {'error' in interpretation_result}")
        print(f"[DEBUG] Tasks: {interpretation_result.get('tasks', 'None')}")
        
        if interpretation_result.get('tasks'):
            print(f"[DEBUG] Number of tasks: {len(interpretation_result['tasks'])}")
            for i, task in enumerate(interpretation_result['tasks']):
                print(f"[DEBUG] Task {i+1}: {task}")
        
        # Step 2: CODER AGENT
        print(f"\n[DEBUG] STEP 2: CODER AGENT Processing")
        print("-" * 50)
        
        tasks = interpretation_result.get('tasks', [])
        if tasks:
            print(f"[DEBUG] Passing {len(tasks)} tasks to CODER AGENT")
            if hasattr(self.data_analyst, 'process_task_list'):
                code_results = self.data_analyst.process_task_list(tasks)
            else:
                code_results = []
            print(f"[DEBUG] CODER AGENT returned {len(code_results)} results")
            
            for i, result in enumerate(code_results):
                print(f"[DEBUG] Code result {i+1}:")
                print(f"  Task ID: {result.get('task_id', 'None')}")
                print(f"  Code length: {len(result.get('code', ''))}")
                print(f"  Has analysis: {bool(result.get('analysis', ''))}")
        else:
            print("[DEBUG] No tasks to pass to CODER AGENT")
            code_results = []
        
        return {
            "debug_mode": True,
            "interpretation_result": interpretation_result,
            "code_results": code_results,
            "tasks_passed": len(tasks),
            "results_returned": len(code_results)
        }
    
    def get_system_status(self) -> str:
        """Get status of both agents"""
        return f"""
INTEGRATED GRAVITATIONAL WAVE ANALYSIS SYSTEM STATUS (Knowledge-Based)
Session: {self.session_id}

Scientific Interpreter:
- Mode: LLM Knowledge-Based (No Web Search)
- Tokens used: {self.scientific_interpreter.total_tokens_used}
- Conversation history: {len(self.scientific_interpreter.conversation_history)} messages

Coder Agent:
- ChromaDB: {'Connected' if hasattr(self.data_analyst, 'collection') and self.data_analyst.collection else 'Not Connected'}
- Available packages: {len(getattr(self.data_analyst, 'installed_packages', {}))}
- Tokens used: {getattr(self.data_analyst, 'total_tokens_used', 0)}

Available Scientific Packages:
{chr(10).join(f'- {pkg} v{ver}' for pkg, ver in getattr(self.data_analyst, 'installed_packages', {}).items()) if hasattr(self.data_analyst, 'installed_packages') and self.data_analyst.installed_packages else '- None detected'}

Total tokens used: {self.scientific_interpreter.total_tokens_used + getattr(self.data_analyst, 'total_tokens_used', 0)}
"""

def main():
    """Main interactive interface with enhanced debugging"""
    
    # Initialize integrated system
    system = IntegratedGravitationalWaveSystem()
    
    print(f"\n{system.get_system_status()}")
    
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print("Commands:")
    print("  - Enter a gravitational wave analysis query")
    print("  - 'status' to check system status")
    print("  - 'packages' to see available packages")
    print("  - 'debug <query>' to run in debug mode")
    print("  - 'test' to run a simple test query")
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
        elif user_input.lower() == 'packages':
            print("\nAvailable Scientific Packages:")
            if system.data_analyst.installed_packages:
                for pkg, version in system.data_analyst.installed_packages.items():
                    print(f"  ✓ {pkg} v{version}")
            else:
                print("  No scientific packages detected")
        elif user_input.lower().startswith('debug '):
            query = user_input[6:].strip()
            if query:
                try:
                    debug_result = system.debug_task_flow(query)
                    print(f"\n[DEBUG SUMMARY]")
                    print(f"Tasks generated: {debug_result['tasks_passed']}")
                    print(f"Code results: {debug_result['results_returned']}")
                    print(f"Success: {debug_result['tasks_passed'] > 0 and debug_result['results_returned'] > 0}")
                except Exception as e:
                    print(f"Debug failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Please provide a query after 'debug'")
        elif user_input.lower() == 'test':
            try:
                print("\n[RUNNING TEST QUERY]")
                test_query = "analyze gravitational wave strain data"
                result = system.process_query(test_query)
                
                print(f"\n[TEST RESULTS]")
                print(f"Status: {result.get('status', 'unknown')}")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Tasks generated: {result['scientific_interpretation']['tasks_generated']}")
                    print(f"Tasks processed: {result['data_analysis']['tasks_processed']}")
                    print(f"Documentation sources: {result['data_analysis']['total_documentation_sources']}")
                
            except Exception as e:
                print(f"Test failed: {e}")
                import traceback
                traceback.print_exc()
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
                    
                    if interpretation.get('has_error'):
                        print(f"Warning: {result.get('interpretation_warning', 'Unknown interpretation issue')}")
                    
                    # Display data analysis results
                    analysis = result['data_analysis']
                    print(f"\nCode generation:")
                    print(f"Tasks processed: {analysis['tasks_processed']}")
                    print(f"Documentation sources used: {analysis['total_documentation_sources']}")
                    print(f"Available packages: {len(analysis['available_packages'])}")
                    
                    # Show generated code for each task
                    for i, code_result in enumerate(analysis['code_results'], 1):
                        print(f"\nTask {i}: {code_result['task_description']}")
                        print(f"Documentation sources: {code_result.get('documentation_sources', [])}")
                        print(f"Packages available: {len(code_result.get('available_packages', {}))}")
                        if code_result.get('code'):
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
                import traceback
                traceback.print_exc()
        else:
            print("Please enter a query or command.")

if __name__ == "__main__":
    main()