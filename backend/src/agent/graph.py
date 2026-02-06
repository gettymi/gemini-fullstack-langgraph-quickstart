import os
from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.utils import get_research_topic, search_local_directory

load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Groq Llama to create optimized search queries for web research.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Initialize Groq LLM
    llm = ChatGroq(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: OverallState):
    """LangGraph node that sends the search queries to the web research node."""
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx), "search_dir": state.get("search_dir", ".")})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs local directory search.

    Searches the local directory for files matching the search query and summarizes results with Groq.

    Args:
        state: Current graph state containing the search query
        config: Configuration for the runnable

    Returns:
        Dictionary with state update, including sources and research results
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Perform local directory search
    search_results = search_local_directory(
        query=state["search_query"],
        directory=state.get("search_dir", "."),
        max_results=5,
    )
    
    # Extract and format results
    sources_gathered = []
    context_parts = []
    
    for idx, result in enumerate(search_results.get("results", [])):
        source = {
            "label": result.get("title", "Source"),
            "value": result.get("url", ""),
        }
        sources_gathered.append(source)
        
        # Build context for LLM with file paths - LLM will use these in its response
        context_parts.append(
            f"**{source['label']}** (from: {source['value']})\n{result.get('content', '')}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Use Groq to summarize the findings
    llm = ChatGroq(
        model=configurable.query_generator_model,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    
    full_prompt = f"{formatted_prompt}\n\nLocal Search Results:\n{context}"
    response = llm.invoke(full_prompt)
    
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [response.content],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates follow-up queries.

    Args:
        state: Current graph state containing research results
        config: Configuration for the runnable

    Returns:
        Dictionary with reflection results
    """
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    
    # Initialize Groq LLM
    llm = ChatGroq(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: OverallState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                    "search_dir": state.get("search_dir", "."),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Args:
        state: Current graph state with all research results
        config: Configuration for the runnable

    Returns:
        Dictionary with final answer
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # Initialize Groq LLM
    llm = ChatGroq(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Return all sources - LLM includes them in the response naturally
    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": state["sources_gathered"],
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint
builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")