# config.py

# Agent模型配置
MODEL = "D:/qwen_0.6b"
MODEL_PROVIDER = "qwen"

# 用户模拟器配置  此处无用，不需调整
USER_MODEL = "gpt-4o"
USER_MODEL_PROVIDER = "openai"

# 任务参数
NUM_TRIALS = 1
ENV = "retail"  # 可选: "retail", "airline"
AGENT_STRATEGY = "tool-calling"  # 可选: "tool-calling", "act", "react", "few-shot"
TASK_SPLIT = "test"  # 可选: "train", "test", "dev"
TEMPERATURE = 0.1

# 任务范围控制
START_INDEX = 0
END_INDEX = -1  # -1 表示运行所有任务
TASK_IDS = [2]

# 日志与并发
LOG_DIR = "results"
MAX_CONCURRENCY = 1

# 随机种子
SEED = 10
SHUFFLE = 0  # 0 表示不打乱顺序

# 用户行为策略
USER_STRATEGY = "human"  # 参考 UserStrategy 枚举值
FEW_SHOT_DISPLAYS_PATH = None  # 可选: 少样本路径
