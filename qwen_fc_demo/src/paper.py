import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from utils import search_papers, extract_info

# 加载.env文件中的环境变量
load_dotenv()

# 使用load_dotenv()后，可以通过os.getenv()获取环境变量
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for papers on arXiv based on a topic and store their information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for"
                    }, 
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve",
                        "default": 5
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": "Search for information about a specific paper across all topic directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The ID of the paper to look for"
                    }
                },
                "required": ["paper_id"]
            }
        }
    }
]

mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

def execute_tool(tool_name, tool_args):
    
    result = mapping_tool_function[tool_name](**tool_args)

    if result is None:
        result = "The operation completed but didn't return any results."
        
    elif isinstance(result, list):
        result = ', '.join(result)
        
    elif isinstance(result, dict):
        # Convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)
    
    else:
        # For any other type, convert using str()
        result = str(result)
    return result


def process_query(query):
    
    messages = [{'role': 'user', 'content': query}]
    
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools
    )
    
    process_query = True
    while process_query:

        for choice in response.choices:
            if choice.finish_reason == "stop":
                # 直接回复，无需工具调用
                print(choice.message.content)
                
                # 保存回复到历史记录
                messages.append({
                    'role': 'assistant',
                    'content': choice.message.content
                })
                
                if len(response.choices) == 1:
                    process_query = False
            
            elif choice.finish_reason == "tool_calls":
                # 需要调用工具
                if choice.message.content:
                    print(f"Assistant: {choice.message.content}")

                # 添加带有工具调用的消息
                messages.append({
                    'role': 'assistant', 
                    'content': choice.message.content,
                    'tool_calls': choice.message.tool_calls
                })

                # 处理每个工具调用
                for tool_call in choice.message.tool_calls:
                    tool_id = tool_call.id
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_name = tool_call.function.name
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    result = execute_tool(tool_name, tool_args)
                    
                    # 添加工具响应消息，使用特定格式
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result
                    })
                response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=messages,
                    tools=tools
                )
                print(response.choices)

                if len(response.choices) == 1 and response.choices[0].finish_reason == "stop":
                    print(response.choices[0].message.content)
                    process_query = False

def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
    
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")

chat_loop()