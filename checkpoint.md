# Checkpoint

- **Current task**: Verify non-MTP path performance bottleneck
- **Context**: LlamaServer iOS app project
- **User instructions**: See AGENTS.md
- **Next step**: Read code to verify performance findings before taking action
- **Key files to inspect**: Sources/App/LlamaInference.swift (decode + sample path), Sources/App/LlamaHTTPServer.swift (serial queue, context recreate)
- **Key questions**: Is plain path O(N²) CPU work per N tokens? Does it reuse prompt prefix? What's the ceiling?