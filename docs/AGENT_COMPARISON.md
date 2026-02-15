# Q&A Agent Comparison Guide

## Quick Answer: Which Agent Should I Use?

### 🧠 LangGraph Agent (Advanced)
**Use when you need:**
- Complex, multi-part questions
- Queries that require breaking down into steps
- Research and exploration
- Handling of unclear or ambiguous questions

**Example questions:**
- "Compare sales trends between Electronics and Home categories over the last 3 months, and identify any unusual patterns"
- "What's the correlation between discount rates and sales volume, segmented by region?"
- "Show me products with declining sales but increasing returns"

### 🤖 Multi-Agent Orchestrator (Fast)
**Use when you need:**
- Quick, direct answers
- Simple aggregations (SUM, COUNT, AVG)
- Dashboard queries
- Real-time metrics

**Example questions:**
- "Which category has the highest total sales?"
- "What's the average order value?"
- "Show me top 10 products by revenue"
- "How many orders were placed in May?"

---

## Detailed Comparison

### Architecture

| Aspect | LangGraph | Multi-Agent |
|--------|-----------|-------------|
| **Nodes/Stages** | 7 processing nodes | 4 sequential agents |
| **Flow Type** | State machine with loops | Linear pipeline |
| **Branching** | Conditional (simple vs complex) | Sequential only |
| **Retry Logic** | Built-in with query refinement | Basic error handling |

### Performance

| Metric | LangGraph | Multi-Agent |
|--------|-----------|-------------|
| **Simple Query** | 3-4 seconds | 1-2 seconds |
| **Complex Query** | 5-8 seconds | 2-3 seconds (if possible) |
| **Memory Usage** | Higher (state tracking) | Lower (stateless) |
| **Throughput** | 8-10 queries/min | 20-30 queries/min |

### Capabilities

| Feature | LangGraph | Multi-Agent |
|---------|-----------|-------------|
| **Query Decomposition** | ✅ Automatic | ❌ No |
| **Sub-query Execution** | ✅ Yes | ❌ No |
| **Multi-table Joins** | ✅ Complex joins | ⚠️ Limited |
| **Self-healing** | ✅ Retry & refine | ⚠️ Basic |
| **Validation Loops** | ✅ Multi-stage | ✅ Single-stage |
| **Direct Computation** | ⚠️ Always uses LLM | ✅ Pattern matching |

### Processing Flow

**LangGraph (7 Nodes):**
```
1. Analyze Query → Classify complexity
2. Decompose (if complex) → Break into sub-queries
3. Extract Data → Execute SQL
4. LLM Analysis → Generate insights
5. Validate Results → Check quality
6. Refine Query (if needed) → Loop back
7. Format Response → Final output
```

**Multi-Agent (4 Stages):**
```
1. Query Resolution → Parse intent, map columns
2. Data Extraction → Execute SQL with aggregations
3. Validation → Compute answers, generate insights
4. Response Formatting → Add visualizations
```

---

## When to Switch Agents

### Start with Multi-Agent if:
- ✅ You're asking a straightforward question
- ✅ You need a quick dashboard metric
- ✅ The query involves a single table
- ✅ You're asking for totals, averages, or counts

### Switch to LangGraph if:
- ⚠️ Multi-Agent didn't understand your question
- ⚠️ You need to compare across multiple dimensions
- ⚠️ The question has multiple parts
- ⚠️ You need drill-down analysis

---

## Technical Implementation Details

### LangGraph Agent
**File**: `src/graph/langgraph_agent.py`

**Key Features:**
- Uses LangGraph's StateGraph for workflow management
- Tool-based architecture with `query_database()` and `analyze_data()` tools
- State persistence across nodes
- Conditional edges based on query complexity
- Automatic retry with refinement

**Typical Flow:**
```python
workflow = StateGraph(...)
workflow.add_node("analyze_query", self._node_analyze_query)
workflow.add_node("decompose_query", self._node_decompose_query)
workflow.add_node("extract_data", self._node_extract_data)
workflow.add_node("llm_analysis", self._node_llm_analysis)
workflow.add_node("validate_results", self._node_validate_results)
workflow.add_node("refine_query", self._node_refine_query)
workflow.add_node("format_response", self._node_format_response)
```

### Multi-Agent Orchestrator
**File**: `src/agents/multi_agent.py`

**Key Features:**
- Linear 4-agent pipeline
- Direct computation for common aggregations
- Pattern matching for question types
- Optimized SQL generation
- Fast path for simple queries

**Typical Flow:**
```python
# 1. Query Resolution
query_resolution = self.query_agent.resolve_query(user_query)

# 2. Data Extraction (with GROUP BY, aggregations)
extracted_data = self.data_agent.extract_data(query_resolution)

# 3. Validation & Insights
validation_result = self.validation_agent.validate_and_interpret(
    extracted_data, user_query
)

# 4. Format Response
formatted_response = self.formatter.format_response(
    validation_result.get("insights"),
    extracted_data.get("data"),
    query_resolution
)
```

---

## Common Questions

**Q: Can I use both agents for the same question?**
A: Yes! Try Multi-Agent first for speed. If the answer isn't satisfactory, use LangGraph for deeper analysis.

**Q: Which agent is more accurate?**
A: Both use the same data source and LLM. LangGraph may be more accurate for complex queries due to validation loops, while Multi-Agent is more reliable for simple aggregations due to direct computation.

**Q: Does conversation memory work with both?**
A: Yes! Both agents support conversation memory with FAISS + RAG for context-aware responses.

**Q: Can I see which agent was used?**
A: Yes, you select the agent before asking your question in the Q&A tab. The UI shows which agent is active.

**Q: Are the results different between agents?**
A: For simple queries, results should be identical (both query the same database). For complex queries, LangGraph may provide more detailed insights due to its multi-stage analysis.

---

## Performance Tips

### For Faster Responses:
1. Use Multi-Agent for dashboard queries
2. Ask direct questions ("What is the total sales?")
3. Query specific tables using the source selector
4. Avoid overly complex compound questions

### For Better Accuracy:
1. Use LangGraph for ambiguous questions
2. Let the system decompose complex queries
3. Review confidence scores
4. Use follow-up questions for refinement

### For Best Results:
1. Start specific → Use Multi-Agent
2. If unclear → Use LangGraph
3. Check confidence score (>80% = reliable)
4. Use conversation memory for context

---

## Summary

| Your Need | Recommended Agent | Reason |
|-----------|------------------|--------|
| Quick metrics | Multi-Agent | 2x faster |
| Dashboard KPIs | Multi-Agent | Optimized for aggregations |
| Exploration | LangGraph | Better at understanding intent |
| Complex analysis | LangGraph | Query decomposition |
| Research questions | LangGraph | Self-healing with retries |
| Real-time queries | Multi-Agent | Lower latency |
| Ambiguous questions | LangGraph | Better clarification |

**Default recommendation**: Start with **Multi-Agent** for most queries. Use **LangGraph** when you need more sophisticated analysis or when initial results don't fully answer your question.
