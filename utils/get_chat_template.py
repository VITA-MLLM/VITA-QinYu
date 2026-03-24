qwen2_chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""

hunyuan_chat_template = """
{% set context = {'has_head': true} %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['content'] %}{% if loop.index0 == 0 %}{% if content == '' %}{% set _ = context.update({'has_head': false}) %}{% else %}{% set content = '<|startoftext|>' + content + '<|extra_4|>' %}{% endif %}{% endif %}{% if message['role'] == 'user' %}{% if loop.index0 == 1 and not context.has_head %}{% set content = '<|startoftext|>' + content %}{% endif %}{% if loop.index0 == 1 and context.has_head %}{% set content = content + '<|extra_0|>' %}{% else %}{% set content = '<|startoftext|>' + content + '<|extra_0|>' %}{% endif %}{% elif message['role'] == 'assistant' %}{% set content = content + '<|eos|>' %}{% endif %}{{ content  }}{% endfor %}
"""

glm4_chat_template = """
{%- for message in messages %} {%- if (message.role == "system") %} {{- '<|system|>' + '\n' + message.content }} {%- elif (message.role == "user") %} {{- '<|user|>' + '\n' + message.content }} {%- elif message.role == "assistant" %} {{- '<|assistant|>' }} {%- if message.content %} {{- 'streaming_transcription\n' + message.content }} {%- endif %} {%- endif %} {%- endfor %} {%- if add_generation_prompt %} {{- '<|assistant|>streaming_transcription\n' }} {%- endif %}
"""

if __name__ == "__main__":
    from transformers import AutoTokenizer

    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]

    # print("=" * 100)
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
    # print(tokenizer.get_chat_template())
    # message = tokenizer.apply_chat_template(chat, tokenize=False)
    # print(message)

    # print("=" * 100)
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    # print(tokenizer.get_chat_template())
    # message = tokenizer.apply_chat_template(chat, tokenize=False)
    # print(message)

    # print("=" * 100)
    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-70B-Instruct")
    # print(tokenizer.get_chat_template())
    # message = tokenizer.apply_chat_template(chat, tokenize=False)
    # print(message)

    # print("=" * 100)
    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
    # print(tokenizer.get_chat_template())
    # message = tokenizer.apply_chat_template(chat, tokenize=False)
    # print(message)
    # message = tokenizer.apply_chat_template(chat, tokenize=True)
    # print(message)

    print("=" * 100)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    print(tokenizer.get_chat_template())
    message = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    print(message)
    message = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(message)
