import os
from typing import List, Dict, Any
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def search_local_directory(query: str, directory: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for files in a local directory that match the query.
    Uses keyword-based matching to find relevant documents.
    
    Args:
        query: The search query
        directory: The directory to search in
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results in Tavily-compatible format
    """
    # Check if directory exists
    if not os.path.isdir(directory):
        return {"results": []}
    
    # Split query into keywords (remove common words)
    keywords = [word.lower() for word in query.split() if len(word) > 2]
    
    file_scores = {}  # {file_path: (score, content, filename)}
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # Skip hidden files and non-text files
            if file.startswith('.') or not _is_text_file(file):
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                # Try to read the file as text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Score the file based on keyword matches
                score = _score_file(file, content, keywords)
                
                if score > 0:
                    file_scores[file_path] = (score, content, file)
                        
            except (IsADirectoryError, PermissionError, UnicodeDecodeError):
                continue
    
    # Sort by score and take top results
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1][0], reverse=True)
    
    results = []
    for file_path, (score, content, filename) in sorted_files[:max_results]:
        # Extract a relevant snippet
        snippet = _extract_snippet(content, keywords)
        
        results.append({
            "title": filename,
            "url": file_path,
            "content": snippet,
        })
    
    return {"results": results}


def _is_text_file(filename: str) -> bool:
    """Check if file is likely a text file."""
    text_extensions = {
        '.md', '.txt', '.py', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.js'
    }
    _, ext = os.path.splitext(filename)
    return ext.lower() in text_extensions


def _score_file(filename: str, content: str, keywords: List[str]) -> int:
    """Score a file based on how many keywords it contains."""
    score = 0
    content_lower = content.lower()
    filename_lower = filename.lower()
    
    for keyword in keywords:
        # Count occurrences in filename (higher weight)
        filename_count = filename_lower.count(keyword)
        score += filename_count * 10
        
        # Count occurrences in content
        content_count = content_lower.count(keyword)
        score += content_count
    
    return score


def _extract_snippet(content: str, keywords: List[str]) -> str:
    """Extract a relevant snippet from content based on keywords."""
    content_lower = content.lower()
    
    # Find the first occurrence of any keyword
    earliest_pos = len(content)
    found_keyword = None
    
    for keyword in keywords:
        pos = content_lower.find(keyword)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            found_keyword = keyword
    
    if found_keyword:
        # Extract 300 chars before and 500 chars after the keyword
        start = max(0, earliest_pos - 300)
        end = min(len(content), earliest_pos + 500)
        snippet = content[start:end]
        
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
    else:
        # No keyword found, return first 500 chars
        snippet = content[:500]
        if len(content) > 500:
            snippet = snippet + "..."
    
    return snippet