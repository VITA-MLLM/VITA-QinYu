# VITA-QinYu Web Demo Refactored Version

## 概述

`web_demo_ractor.py` 是 `web_demo_stream.py` 的重构版本，主要改进是将硬编码的参数提取到 YAML 配置文件中，使配置更加灵活和易于维护。

## 文件说明

- `web_demo_ractor.py` - 重构后的主程序文件
- `config.yaml` - 配置文件（包含所有可配置参数）
- `web_demo_stream.py` - 原始版本（保持不变作为参考）

## 主要改进

### 1. 配置文件化
所有重要参数都移到了 `config.yaml` 文件中，包括：
- 服务器配置（IP、端口、最大用户数、超时等）
- 模型路径和参数
- 音频处理参数
- 流式生成参数
- ASR和Turn Detection配置
- 特殊Token配置
- 系统路径配置

### 2. 保持代码逻辑不变
核心函数逻辑完全保持不变：
- `decode_stream()` - 音频解码流处理
- `fade_in_out()` - 音频淡入淡出
- `__run_infer_stream()` - 核心推理流处理
- `run_infer_stream()` - 推理流包装函数
- `send_pcm()` - PCM音频流处理
- `turn_detect()` - Turn Detection检测
- 所有SocketIO handlers保持不变

### 3. 灵活的配置覆盖
支持通过命令行参数覆盖配置文件中的设置：

```bash
# 使用默认配置
python web_demo_ractor.py

# 使用自定义配置文件
python web_demo_ractor.py --config my_config.yaml

# 覆盖特定参数
python web_demo_ractor.py --port 8080 --mode roleplay

# 完整示例
python web_demo_ractor.py --config config.yaml --port 8080 --mode roleplay \
  --role_description "该角色是一个中年男性，身份是修真界的前辈高人"
```

## 使用方法

### 基本使用

```bash
# 默认模式（使用config.yaml中的配置）
CUDA_VISIBLE_DEVICES=0 python web_demo_ractor.py

# 指定端口
CUDA_VISIBLE_DEVICES=0 python web_demo_ractor.py --port 8080

# Roleplay模式
CUDA_VISIBLE_DEVICES=0 python web_demo_ractor.py --mode roleplay
```

### 配置文件修改

编辑 `config.yaml` 文件来修改默认配置：

```yaml
# 修改服务器配置
server:
  ip: '0.0.0.0'
  port: 8081  # 修改默认端口
  max_users: 20
  timeout: 600

# 修改模型路径
model:
  model_name_or_path: "../vita-qinyu-models/checkpoint-8000-sing"

# 修改流式参数
streaming:
  max_code_length: 50
  first_code_length: 16
  fade_code_length: 4
```

## 配置文件结构

### 服务器配置
```yaml
server:
  ip: '0.0.0.0'
  port: 8081
  max_users: 20
  timeout: 600
  ssl_enabled: false
```

### 模型配置
```yaml
model:
  model_name_or_path: "../vita-qinyu-models/checkpoint-8000-sing"
  audio_tokenizer_path: [...]
  audio_tokenizer_type: "sensevoice_xytokenizer_speaker"
  device_map: "cuda:0"
  torch_dtype: "bfloat16"
```

### 生成配置
```yaml
generation:
  max_new_tokens: 8192
  temperature: 1.0
  top_k: 50
  top_p: 1.0
  do_sample: false
```

### 流式参数
```yaml
streaming:
  max_code_length: 50
  first_code_length: 16
  fade_code_length: 4
  cache_wav_len: 6400
  num_history: 4
```

### ASR配置
```yaml
asr:
  model_id: 'openai/whisper-large-v3'
  torch_dtype: "float32"
  language: "zh"
  max_new_tokens: 128
```

## 命令行参数

```bash
usage: web_demo_ractor.py [-h] [--config CONFIG] [--ip IP] [--port PORT]
                          [--max_users MAX_USERS] [--timeout TIMEOUT]
                          [--mode MODE] [--role_description ROLE_DESCRIPTION]

optional arguments:
  --config CONFIG              配置文件路径 (默认: config.yaml)
  --ip IP                      服务器IP (覆盖配置文件)
  --port PORT                  服务器端口 (覆盖配置文件)
  --max_users MAX_USERS        最大用户数 (覆盖配置文件)
  --timeout TIMEOUT            超时时间 (覆盖配置文件)
  --mode MODE                  运行模式: default/roleplay (覆盖配置文件)
  --role_description DESC      角色描述 (roleplay模式使用)
```

## 与原版的差异

1. **参数来源**：原版硬编码 → 新版从YAML读取
2. **灵活性**：原版需修改代码 → 新版只需修改配置文件
3. **可维护性**：原版参数分散 → 新版集中管理
4. **核心逻辑**：完全相同，保持兼容性

## 注意事项

1. 首次运行前请确保 `config.yaml` 中的所有路径都正确
2. 模型路径需要根据实际部署位置调整
3. 如果使用 roleplay 模式，确保相关模型文件存在
4. SSL配置默认关闭，如需启用请修改 `config.yaml` 中的 `ssl_enabled`

## 故障排查

### 配置文件未找到
```
错误: FileNotFoundError: config.yaml
解决: 确保config.yaml在当前目录，或使用--config指定路径
```

### 模型路径错误
```
错误: OSError: model not found
解决: 检查config.yaml中的model_name_or_path是否正确
```

### 端口被占用
```
错误: Address already in use
解决: 修改config.yaml中的port或使用--port参数指定其他端口
```

## 示例配置场景

### 场景1：开发环境
```yaml
server:
  port: 8080
  max_users: 5
model:
  torch_dtype: "float32"  # 兼容性更好
logging:
  level: "DEBUG"
```

### 场景2：生产环境
```yaml
server:
  port: 443
  max_users: 100
  ssl_enabled: true
model:
  torch_dtype: "bfloat16"  # 性能更好
logging:
  level: "INFO"
```

### 场景3：测试环境
```yaml
server:
  port: 9000
  max_users: 2
  timeout: 300
streaming:
  max_code_length: 30  # 降低资源占用
```

## 技术支持

如遇到问题，请检查：
1. YAML语法是否正确（注意缩进）
2. 所有路径是否存在且可访问
3. CUDA设备是否可用
4. 依赖包是否已安装

## 更新日志

### v1.0 (2026-02-05)
- 初始版本
- 将所有硬编码参数提取到config.yaml
- 保持原有代码逻辑完全不变
- 添加命令行参数覆盖支持
