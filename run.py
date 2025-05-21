# runner.py
import argparse
from config import *  # 导入所有配置变量
from tau_bench.types import RunConfig
from tau_bench.run import run
from tau_bench.envs.user import UserStrategy

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--env", type=str, choices=["retail", "airline"], default=ENV)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--model-provider", type=str, default=MODEL_PROVIDER)
    parser.add_argument("--user-model", type=str, default=USER_MODEL)
    parser.add_argument("--user-model-provider", type=str, default=USER_MODEL_PROVIDER)
    parser.add_argument("--agent-strategy", type=str, default=AGENT_STRATEGY, 
                        choices=["tool-calling", "act", "react", "few-shot"])
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--task-split", type=str, default=TASK_SPLIT, 
                        choices=["train", "test", "dev"])
    parser.add_argument("--start-index", type=int, default=START_INDEX)
    parser.add_argument("--end-index", type=int, default=END_INDEX)
    parser.add_argument("--task-ids", type=int, nargs="+", default=TASK_IDS)
    parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shuffle", type=int, default=SHUFFLE)
    parser.add_argument("--user-strategy", type=str, default=USER_STRATEGY, 
                        choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, default=FEW_SHOT_DISPLAYS_PATH)
    args = parser.parse_args()

    return RunConfig(**vars(args))

def main():
    config = parse_args()
    print(config)
    run(config)

if __name__ == "__main__":
    main()
