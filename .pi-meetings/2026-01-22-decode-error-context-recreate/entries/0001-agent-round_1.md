# agent (Round 1): Diagnose decode errors and context recreation every 10 loop in LlamaServer

# Investigation: Decode Errors & Context Recreation

## Task
- Read the LlamaServer project code
- Diagnose decode errors
- Understand why context is recreated every 10 loop
- Focus on fixing one problem at a time

## Context from AGENTS.md
- Context is recreated every 10 generations to flush Metal/allocator state
- The recreate runs off the response path
- Key files: Sources/App/LlamaInference.swift, Sources/App/LlamaHTTPServer.swift