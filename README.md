FlexibleAgents: A framework for easy and customizable Multi-Agent Orchestration

FlexibbleAgents is the Bachelor's Thesis Project of Maximilian Miles, aimed at building and benchmarking a multi-agent conversation framework. Originally built as an autogen-/ag2-based reimplementation of the GravAgents project (https://github.com/ultor1996/gravagents), the project morphed to expand upon the original idea to bridge the gap between highly performant, specialized LLM Agent systems and the more general applicability of regular LLM chatbots/completions.

Noteable required package versions:
    autogen/ag2: 0.11.0
    langchain: 1.2.15
    beautifulsoup4: 4.14.3
    docling: 2.90.0
    llama-index: 0.13.6
    browser-use: 0.1.37
    pydantic: 2.13.3

Most of these dependencies are auto-versioned and properly installed by "pip install ag2[openai, rag, docling]" as of 04/2026.

Further libraries (used for implementation of more task-specific chats):
    pycbc: 2.11.0
    lalsuite: 7.26.7
    gwpy: 4.0.1
    bilby: 2.8.0