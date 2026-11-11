---
trigger: always_on
description: MANDATORY Graph-First Policy: Always query graphify before inspecting or modifying any files in the project.
---

## Graphify Mandatory Graph-First Investigation Rule

The primary purpose of Graphify in this project is to eliminate blind, unguided, and expensive raw file analysis. The knowledge graph at `graphify-out/` provides the structural truth, symbol dependencies, call flows, and architectural relationships of the codebase.

### Mandatory Rules (Zero Exceptions):

1. **NEVER ANALYZE OR READ FILES DIRECTLY FIRST:**
   - Under no circumstances should the agent start a task by directly opening, reading, listing, or grepping source code files to "explore" or "understand" the system.
   - Direct, unguided file analysis before consulting graphify is strictly forbidden.

2. **ALWAYS QUERY GRAPHIFY FIRST:**
   - For ANY task (feature implementation, bug diagnosis, mobile adaptation, refactoring, code review, or architecture questions), the agent MUST first execute:
     - `graphify query "<question or intent>"` to retrieve relevant nodes, community context, and structural relationships.
     - `graphify explain "<concept or symbol>"` for focused concepts or components.
     - `graphify path "<node_A>" "<node_B>"` to trace paths and interactions between components.

3. **ANALYZE AND MODIFY ONLY BASED ON GRAPHIFY KNOWLEDGE:**
   - Review the scoped subgraph and node coordinates (`src`, `loc`, `community`) returned by graphify.
   - Use the knowledge, dependencies, and call-paths surfaced by graphify to pinpoint the exact files and lines that require inspection or edits.
   - Ground all architectural decisions and code modifications in the graph's validated structure.

4. **KEEP THE GRAPH SYNCHRONIZED:**
   - Immediately after modifying, creating, or deleting code files in any session, run:
     `graphify update .`
     (AST-only, ultra-fast, zero API cost) to ensure subsequent queries reflect the latest code state.
