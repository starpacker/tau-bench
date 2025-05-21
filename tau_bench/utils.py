from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import uuid
import re
import json

def build_prompt(messages: List[Dict[str, Any]], tools_info: List[Dict[str, Any]]) -> str:
    """
    构造包含工具描述的 prompt,供本地模型使用。
    
    参数:
        messages: 对话历史，如 [{"role": "user", "content": "..."}, ...]
        tools_info: 工具列表，每个工具为一个字典

    返回:
        构造好的 prompt 字符串
    """
    # Step 1: 构造工具描述部分
    if tools_info is None or len(tools_info) == 0:
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            full_prompt += f"{role}: {content}\n"

        full_prompt += "Assistant: "

        return full_prompt
    tool_descriptions = []
    for i, tool in enumerate(tools_info):
        func = tool["function"]
        name = func["name"]
        description = func["description"]
        parameters = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        # 构造参数描述
        param_strs = []
        for param_name, param_info in parameters.items():
            param_type = param_info.get("type", "unknown")
            param_desc = param_info.get("description", "")
            is_required = " (required)" if param_name in required else ""
            param_strs.append(f"  - {param_name}: {param_type}{is_required} — {param_desc}")

        # 拼接单个工具描述
        tool_desc = f"{i + 1}. **{name}**\n   {description}\n   Parameters:\n" + "\n".join(param_strs)
        tool_descriptions.append(tool_desc)

    tools_section = "\n\n".join(tool_descriptions)

    # Step 2: 构造完整的 prompt
    system_prompt = (
        "你是一个助手。当用户的问题需要调用外部工具时，请使用 <tool_call> 标记调用工具。\n"
        "可用工具如下：\n"
        f"{tools_section}\n"
        "调用格式为：\n"
        "<tool_call>\n"
        "```json\n"
        "{\n"
        '  "name": "tool_name",\n'
        '  "arguments": {"key": "value", ...}\n'
        "}\n"
        "```\n"
        "否则，请直接回答用户问题。"
    )

    # Step 3: 拼接对话历史
    full_prompt = system_prompt + "\n\n"

    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        full_prompt += f"{role}: {content}\n"

    full_prompt += "Assistant: "

    return full_prompt

def parse_tool_call(text: str, tools_info: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    从模型生成的文本中提取并验证 tool_call 内容。

    参数:
        text: 模型生成的完整响应文本。
        tools_info: 工具列表，用于验证调用的合法性。

    返回:
        若成功解析并验证通过，返回 tool_call 字典；否则返回 None。
    """
    # Step 1: 提取 tool_call 部分
    tool_call_match = re.search(r"<tool_call>\n```json\n(.*?)\n```\n", text, re.DOTALL)
    if not tool_call_match:
        print("⚠️ 未找到有效的 tool_call 标记")
        return None

    # Step 2: 解析 JSON
    try:
        tool_call_json = json.loads(tool_call_match.group(1))
    except json.JSONDecodeError as e:
        print(f"⚠️ 无法解析 JSON: {e}")
        return None

    # Step 3: 检查 JSON 结构
    if not isinstance(tool_call_json, dict):
        print("⚠️ tool_call 内容不是字典")
        return None

    if "name" not in tool_call_json or "arguments" not in tool_call_json:
        print("⚠️ tool_call 缺少 name 或 arguments 字段")
        return None

    tool_name = tool_call_json["name"]
    tool_args = tool_call_json["arguments"]

    # Step 4: 检查工具是否存在
    tool_names = [tool["function"]["name"] for tool in tools_info]
    if tool_name not in tool_names:
        print(f"⚠️ 调用的工具 '{tool_name}' 不存在于 tools_info 中")
        return None

    # Step 5: 获取工具定义
    tool_def = next(tool for tool in tools_info if tool["function"]["name"] == tool_name)
    param_schema = tool_def["function"].get("parameters", {}).get("properties", {})
    required_params = tool_def["function"].get("parameters", {}).get("required", [])

    # Step 6: 检查参数是否完整
    missing_required = [param for param in required_params if param not in tool_args]
    if missing_required:
        print(f"⚠️ 缺少必填参数: {missing_required}")
        return None

    # Step 7: 检查参数类型（仅支持 string 类型）
    for param_name, param_value in tool_args.items():
        expected_type = param_schema.get(param_name, {}).get("type")
        if expected_type == "string" and not isinstance(param_value, str):
            print(f"⚠️ 参数 {param_name} 类型应为 string，但实际为 {type(param_value)}")
            return None
        elif expected_type == "number" and not isinstance(param_value, (int, float)):
            print(f"⚠️ 参数 {param_name} 类型应为 number，但实际为 {type(param_value)}")
            return None

    # Step 8: 生成唯一调用 ID
    call_id = f"call_{uuid.uuid4().hex[:8]}"

    # Step 9: 构造返回值
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": tool_args
        }
    }

def completion(
    messages: List[Dict[str, Any]],
    model,
    tokenizer,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.1,
    max_new_tokens: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> Dict[str, Any]:
    """
    替代 litellm.completion 的本地模型推理函数，支持工具调用。
    """
    
    # 2. 构造 Prompt（包含 tools_info）
    prompt = build_prompt(messages, tools)

    # 3. 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 4. 设置生成参数
    generation_config = GenerationConfig(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        **kwargs
    )

    # 5. 生成响应
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    # 6. 解码生成结果
    generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    # 7. 解析 tool_call
    tool_call = parse_tool_call(generated_text,tools_info=tools)

    # 8. 构造返回值
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                    "tool_calls": [tool_call] if tool_call else []
                }
            }
        ],
        "_hidden_params": {
            "response_cost": float(len(generated_text))
        }
    }
