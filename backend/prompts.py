import json
from typing import List, Dict
from datetime import datetime
from backend.knowledge_base import knowledge_base


def _create_base_system_prompt(file_summary: str) -> str:
    """Shared base system prompt"""
    current_time = datetime.now().strftime("%A, %B %d, %Y, at %I:%M:%S %p")
    
    return f"""
- Answers must strictly come from the knowledge base
- To answer completely, you may call the `retrieve_files` tool multiple times
- If you're 100% certain of the answer, you may skip calling the `retrieve_files` tool
- If after diligent multi-round retrieval you still haven't found relevant knowledge, please answer "I don't know"
- Current time: {current_time}

## Knowledge Base File Summary
```
{file_summary}
```

## retrieve_files
- NEVER answer "I don't know" without calling the `retrieve_files` tool
- Input format: {{"file_paths": ["path1", "path2"]}}

### Examples
- Retrieve specific files: ["Product-Line-A-Smartwatch-Series/SW-1500-Sport.md", "Product-Line-B-Smart-Earbuds-Series/AE-Sport-Athletic.md"]
- Retrieve multiple directories: ["2024-Market-Layout/", "2023-Market-Layout/"]
- Retrieve directories and files together: ["2024-Market-Layout/", "2023-Market-Layout/South-China-Region.md"]
- Full retrieval with ["/"] may be disabled by server policy

""".strip()


def create_system_prompt(file_summary: str) -> str:
    """System prompt for function calling mode"""
    return _create_base_system_prompt(file_summary)


def create_react_system_prompt(file_summary: str) -> str:
    """System prompt for ReAct mode with format instructions"""
    base_prompt = _create_base_system_prompt(file_summary)
    
    return f"""
{base_prompt}

## Direct Answer
- `Knowledge Base File Summary` has the answer

### Example
- Question: Besides AMOLED and OLED screens, what other display types do we have?
- Answer: LCD, TFT

## Tool Call
- `Knowledge Base File Summary` doesn't have enough details

### Pattern
- <|Thought|> Think about what information you need to answer the question
- <|Action|> {"tool":"retrieve_files","input":{"file_paths":["path1","path2"]}}
- <|Observation|> [The system will provide file contents here]
- ... (repeat Thought/Action/Observation as needed)
- <|Final Answer|> [Your final answer based on the retrieved information]

### Example
- Question: What are all the technical specifications of SW-2100?
- <|Thought|> The File Summary only mentions basic info. I need the complete product file for all specifications
- <|Action|> {{"tool":"retrieve_files","input":{{"file_paths":["Product-Line-A-Smartwatch-Series/SW-2100-Flagship.md"]}}}}
- <|Observation|> [System provides file content]
- <|Final Answer|> [Complete specifications based on retrieved file]

""".strip()
    

def create_file_retrieval_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "retrieve_files",
            "description": "Retrieve markdown file contents from the knowledge base. You can retrieve specific files or entire directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths or directory paths to retrieve."
                    }
                },
                "required": ["file_paths"]
            }
        }
    }


async def process_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
    results = []

    for tool_call in tool_calls:
        if tool_call.get("function", {}).get("name") == "retrieve_files":
            try:
                args = json.loads(tool_call["function"]["arguments"])
                file_paths = args.get("file_paths", [])

                content = await knowledge_base.retrieve_files(file_paths)

                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": content
                })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": f"Error retrieving files: {str(e)}"
                })

    return results


def _extract_json_after_marker(text: str, marker: str) -> dict | None:
    marker_index = text.rfind(marker)
    if marker_index == -1:
        return None

    start = marker_index + len(marker)
    while start < len(text) and text[start].isspace():
        start += 1

    if start >= len(text):
        return None

    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def parse_react_response(text: str) -> tuple:
    """Parse ReAct response with structured JSON payload and legacy fallback."""
    action_payload = _extract_json_after_marker(text, "<|Action|>")
    if action_payload:
        action = action_payload.get("tool")
        action_input = action_payload.get("input")
        if isinstance(action, str) and isinstance(action_input, dict):
            return action, action_input, True

    # Backward compatibility for older two-line format:
    # <|Action|> retrieve_files
    # <|Action Input|> {"file_paths":[...]}
    import re

    action_pattern = r"<\|Action\|>\s*(\w+)"
    action_match = re.search(action_pattern, text)
    legacy_action_input = _extract_json_after_marker(text, "<|Action Input|>")

    if action_match and isinstance(legacy_action_input, dict):
        action = action_match.group(1)
        return action, legacy_action_input, True

    return None, None, False


async def process_react_response(text: str) -> tuple:
    """Process ReAct response and execute actions"""
    action, action_input, has_action = parse_react_response(text)

    if has_action and action == "retrieve_files":
        file_paths = action_input.get("file_paths", [])
        content = await knowledge_base.retrieve_files(file_paths)

        return {
            "action": action,
            "input": action_input,
            "observation": content
        }, True

    return None, False
