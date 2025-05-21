# Function Calling Agent (Local Version)

**Paper**: [https://arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045)

基于本地大模型的智能代理系统，通过Prompt工程引导模型调用工具完成指定任务。

## 功能特性
- 本地化部署，支持私有模型加载
- 动态工具调用能力
- 支持多任务并发处理
- 可扩展的任务配置系统

## 快速启动

### 1. 安装依赖
```bash
pip install -e .
```

### 2. 配置模型路径
修改 config 中的 model 参数：
```python
model = "/path/to/your/local/model"  # ← 修改为实际模型路径
```

### Run
```bash
python run.py 
```

## 配置说明
- `max-concurrency`: 控制最大并发任务数（建议根据GPU显存调整）
- `task-ids`: 指定要执行的任务编号（不指定则运行所有任务）
- 模型配置：支持常见的本地模型格式（GGUF/PyTorch等）

## 扩展开发
1. 添加新工具：在`tools/`目录下实现工具类并注册
2. 新增任务：在任务配置文件中定义任务逻辑
3. 自定义Prompt：修改`wiki.md`
