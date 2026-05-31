# Cat Behavior RAG

基于 DashScope 的猫行为咨询系统，使用多轮问诊、工具调用和最终深度分析来模拟猫行为顾问与猫主人的咨询流程。

当前代码分为两层：

1. 正式运行层：只负责完整对话 flow，以及 round 级 checkpoint 保存与恢复。
2. 测试层：放在 tests/flow 中，按功能切片验证 tool phase、state update、consult response 等局部行为。

---

## 项目结构

```text
CatBehaviorRag/
├── main.py                         # CLI 入口
├── consultation_runtime.py         # 正式运行 flow（benchmark / free-chat 共用主循环）
├── checkpoint_store.py             # round checkpoint save / load / list
├── Config.py                       # 全局配置
├── requirements.txt
├── agents/
│   ├── consultant_agent.py         # 核心咨询 Agent：rewrite / hypotheses / tool / response / think
│   ├── user_agent.py               # benchmark 模式下模拟猫主人
│   └── memory_store.py             # 记忆占位模块
├── utils/
│   ├── llm_client.py               # DashScope LLM 封装
│   ├── benchmark_loader.py         # benchmark 加载
│   └── tools.py                    # 工具注册与 weather 工具
├── evaluation/
│   ├── batch_eval.py               # 批量评测入口
│   └── metrics/                    # AQT / UFS / DCS / AS / UHA
├── tests/
│   └── flow/
│       ├── test_checkpoint_store.py
│       ├── test_state_helpers.py
│       ├── test_tool_phase.py
│       ├── test_state_update_phase.py
│       └── test_consult_response_phase.py
├── data/
│   └── enquiries_benchmark_v2.json
└── run/                            # 每次运行生成的日志与 checkpoint
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

OCR 功能还需要本地安装 Tesseract。

### 2. 配置 API Key

```bash
# Linux / macOS
export DASHSCOPE_API_KEY=your_api_key_here

# Windows CMD
set DASHSCOPE_API_KEY=your_api_key_here

# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"
```

---

## 运行模式

### 1. Benchmark 对话

默认运行 benchmark case，由 UserAgent 模拟猫主人。

```bash
python main.py
python main.py --case-id 1
python main.py --case-id 2
python main.py --case-id 3
python main.py --list-cases
```

### 2. 自由对话

真实用户直接和顾问对话。

```bash
python main.py --free-chat
python main.py -f
```

输入 q、quit、exit 或 退出 即可结束。

### 3. 批量评测

```bash
python main.py --batch-eval
python main.py -b
```

输出指标包括：

1. AQT：平均提问轮数
2. UFS：用户友好度
3. DCS：关键方向覆盖率
4. AS：建议可操作性
5. UHA：不确定性处理准确率

---

## Checkpoint 机制

正式运行只保留 round 级 checkpoint，不再保存 tool/state_update/consult_response 这类细粒度恢复点。

每个 run 目录下会生成：

```text
run/<timestamp>_case1/
├── manifest.json
├── state.jsonl
├── history.jsonl
├── trajectory.jsonl
└── checkpoints/
    ├── round_000.json
    ├── round_001.json
    └── final_state.json
```

### 列出 checkpoint

```bash
python main.py --resume-run 20260530_215324_case1 --list-checkpoints --no-tracing
```

### 从某一轮恢复继续执行

```bash
python main.py --resume-run 20260530_215324_case1 --checkpoint-id round_001 --no-tracing
```

恢复执行会创建一个新的 run 目录，不会覆盖原始 run。

---

## CLI 参数

```text
python main.py [选项]

--case-id ID            指定 benchmark case ID
--list-cases            列出所有 case ID
--free-chat, -f         自由对话模式
--batch-eval, -b        批量评测
--resume-run RUN_ID     从指定 run 目录恢复
--checkpoint-id ID      恢复指定 checkpoint，默认最新
--list-checkpoints      列出指定 run 的 checkpoint 摘要
--no-routing            关闭模型路由
--no-tracing            关闭 LangSmith tracing
```

---

## Prompt 调试

tests/flow 现在只保留 prompt 调试脚本和对应场景，不再维护 unittest 形式的 verify 测试。

其中：

1. tests/flow/fixtures/prompt_phase_cases.json 用来恢复调 prompt 时的 context
2. tests/flow/prompt_phase_debug.py 用来执行 tool phase / state update phase / consult response phase
3. 调试脚本和正式运行共用 consultation_runtime.py 里的 phase core

### 运行 prompt phase 调试脚本

PowerShell:

```bash
python tests/flow/prompt_phase_debug.py --phase tool
python tests/flow/prompt_phase_debug.py --phase state_update
python tests/flow/prompt_phase_debug.py --phase consult
python tests/flow/prompt_phase_debug.py --phase all
```

也可以只看某个 case：

```bash
python tests/flow/prompt_phase_debug.py --phase consult --case end
```

要求：

1. 已设置 DASHSCOPE_API_KEY
2. 调试脚本和正式运行共用 consultation_runtime.py 里的 phase core，不再单独维护 preview 版本
3. 脚本不会做任何 assert，只会输出恢复后的 context、phase 输出和更新后的 context
4. 调 prompt 时，直接修改 tests/flow/fixtures/prompt_phase_cases.json 里的场景即可

---

## 模型路由

默认模型路由：

1. fast：qwen-turbo
2. strong：qwen-plus
3. think：qwen3-32b

关闭路由：

```bash
python main.py --no-routing
```

---

## 配置参考

可通过环境变量覆盖：

1. DASHSCOPE_API_KEY：DashScope API 密钥
2. CONSULTATION_MODEL：路由关闭时的备用模型
3. MAX_CONVERSATION_ROUNDS：最大对话轮数
4. ROUTING_ENABLED：模型路由开关
5. MODEL_FAST：快速模型
6. MODEL_STRONG：主对话模型
7. MODEL_THINK：深度思考模型
8. MODEL_THINK_ENABLE_THINKING：是否启用 thinking 模式
9. MEMORY_ENABLED：是否注入记忆
10. CHECKPOINT_ENABLED：是否保存 round checkpoint
11. CHECKPOINT_DIRNAME：checkpoint 目录名

---

## Benchmark 数据格式

data/enquiries_benchmark_v2.json 中每条记录包含：

```json
{
  "id": 1,
  "latitude": "38.9104",
  "longitude": "121.5964",
  "date": "2023-01-19",
  "initial_user_message": "我的猫咪乱尿怎么办？",
  "user_setting": "用户能明确观察到的内容、模糊感觉到的内容、无法回答的问题、主观倾向、不会主动提供的隐藏线索",
  "reference_answer": "参考标准答案",
  "uncertainty": true,
  "required_directions": ["需要识别的关键诊断方向列表"]
}
```

---

## 说明

1. 正式运行能力只提供 round 级 checkpoint 恢复。
2. 细粒度的 tool/state update/consult response 验证放在 tests/flow 中完成。
3. free chat 与 benchmark 现在共用同一条主循环，差别只在用户输入来源。