# Cat Behavior RAG

基于 DashScope 的猫行为咨询系统，结合多轮对话与 Planner/State 机制，模拟专业猫行为学顾问与猫主人之间的咨询流程。

---

## 项目结构

```
CatBehaviorRag/
├── main.py                              # CLI 入口（所有运行模式的统一入口）
├── Config.py                            # 全局配置（模型、路由、路径等）
├── requirements.txt
├── agents/
│   ├── behaviorist_agent.py             # 行为专家 Agent（传统模式：提问 → 深度建议）
│   ├── consultation_agent.py            # 咨询 Agent（Planner 模式：将指令转为自然语言提问）
│   ├── planner_agent.py                 # Planner（决定 ask / call_tool / end）
│   ├── state_manager.py                 # State Manager（更新诊断假设与方向覆盖情况）
│   ├── user_agent.py                    # 猫主人 Agent（Benchmark 模式下模拟用户）
│   └── memory_store.py                  # 记忆存储（当前为占位符）
├── utils/
│   ├── llm_client.py                    # DashScope LLM 封装（流式 + 思考模型）
│   ├── benchmark_loader.py              # Benchmark JSON 加载与解析
│   └── weather_tool.py                  # 天气工具（Open-Meteo Archive API）
├── evaluation/
│   ├── batch_eval.py                    # 批量评测运行器
│   └── metrics/                         # 评估指标（AQT / UFS / DCS / AS / UHA）
├── data/
│   ├── enquiries_benchmark_v2.json      # Benchmark 测试用例
│   └── session_state_history.jsonl      # Planner 模式运行日志（自动生成）
├── rag_document_uploader/               # 文档上传、解析、向量化模块
└── app_update_rag_doc.py                # 文档上传 Streamlit UI
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

OCR 功能还需要本地安装 [Tesseract](https://github.com/tesseract-ocr/tesseract)。

### 2. 配置 API Key

系统使用 DashScope（阿里云百炼）API。

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

系统提供四种运行模式，均通过 `main.py` 启动。

### 模式一：Benchmark 对话（默认模式）

使用预设测试用例，由 `UserAgent` 模拟猫主人，与 `BehavioristAgent`（行为专家）进行多轮对话。专家收集到足够信息后，自动切换思考模型给出深度建议。

```bash
# 运行第一个 case（默认）
python main.py

# 指定 case ID
python main.py --case-id 1
python main.py --case-id 2
python main.py --case-id 3

# 查看所有可用 case
python main.py --list-cases
```

### 模式二：自由对话

由真实用户扮演猫主人，直接与行为专家对话。输入 `q`、`quit` 或 `退出` 结束。

```bash
python main.py --free-chat
# 或简写
python main.py -f
```

### 模式三：Planner 模式

使用 Planner + State Manager + 工具调用的结构化对话循环。每轮由 Planner 决策（提问/调用工具/结束），State Manager 根据回答动态更新诊断假设，运行过程实时写入 `data/session_state_history.jsonl`。

```bash
# 默认运行第一个 case
python main.py --planner

# 指定 case
python main.py --planner --case-id 3
```

**Planner 循环流程：**

```
初始用户消息
    ↓
StateManager 初始化假设（hypothesis）
    ↓
循环：
  Planner 决策 → ask / call_tool / end
      ├── ask      → ConsultationAgent 提问（流式）→ UserAgent 回答 → StateManager 更新 state
      ├── call_tool → 调用工具（如 weather）→ 将结果传回 Planner 重新决策
      └── end      → 输出最终 state，对话结束
```

**天气工具**：当诊断涉及温度、季节等环境因素时，Planner 会自动调用 Open-Meteo Archive API，使用 benchmark 中的经纬度和日期查询真实历史天气数据。

**运行日志**：每个事件（planner 决策、state 更新、工具调用结果、用户回答）均以 JSON 格式追加写入 `data/session_state_history.jsonl`，便于调试和分析。

### 模式四：批量评测

对所有 benchmark case 运行对话并计算评估指标。

```bash
python main.py --batch-eval
# 或简写
python main.py -b
```

输出指标说明：

| 指标 | 全称 | 含义 |
|---|---|---|
| AQT | Average Question Turns | 平均提问轮数 |
| UFS | User-Friendliness Score | 用户友好度（归一化） |
| DCS | Direction Coverage Score | 关键方向覆盖率 |
| AS | Actionability Score | 建议可操作性（归一化） |
| UHA | Uncertainty Handling Accuracy | 不确定性处理准确率 |

---

## CLI 参数速查

```
python main.py [选项]

选项：
  --case-id ID        指定 benchmark case ID（默认使用第一个）
  --list-cases        列出所有可用 case ID 后退出
  --free-chat, -f     自由对话模式（直接与行为专家对话）
  --planner           Planner 模式（使用 planner/tool/state 结构化循环）
  --batch-eval, -b    批量评测所有 case 并输出评估报告
  --no-routing        关闭模型路由，全部使用 consultation_model（消融实验用）
  --no-tracing        强制关闭 LangSmith tracing
```

---

## 模型路由

系统根据任务复杂度自动选用不同模型：

| 角色 | 默认模型 | 用途 |
|---|---|---|
| `fast` | `qwen-turbo` | 低成本分类（UserAgent 事实判断） |
| `strong` | `qwen-plus` | 问答对话、Planner、StateManager |
| `think` | `qwen3-32b` | 最终深度建议（启用 `enable_thinking`） |

关闭路由后，所有调用均使用 `consultation_model`（默认 `qwen-plus`），不使用思考模型：

```bash
python main.py --no-routing
```

---

## 配置参考

所有配置均可通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | *(必填)* | DashScope API 密钥 |
| `CONSULTATION_MODEL` | `qwen-plus` | 路由关闭时的备用模型 |
| `MAX_CONVERSATION_ROUNDS` | `10` | 最大对话轮数 |
| `ROUTING_ENABLED` | `true` | 模型路由总开关 |
| `MODEL_FAST` | `qwen-turbo` | 快速分类模型 |
| `MODEL_STRONG` | `qwen-plus` | 问答对话模型 |
| `MODEL_THINK` | `qwen3-32b` | 深度思考模型 |
| `MODEL_THINK_ENABLE_THINKING` | `true` | 是否启用 qwen3 thinking 模式 |
| `MEMORY_ENABLED` | `true` | 记忆注入总开关 |
| `EMBEDDING_PROVIDER` | `dashscope` | 向量化服务：`dashscope` / `local` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地向量模型 |
| `LOCAL_EMBEDDING_DEVICE` | `cuda` | 本地向量模型设备 |
| `STATE_HISTORY_OUTPUT_PATH` | `data/session_state_history.jsonl` | Planner 模式运行日志路径 |

---

## Benchmark 数据格式

`data/enquiries_benchmark_v2.json` 中每条用例包含：

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

- `latitude` / `longitude`：用于 Planner 模式中的天气工具调用
- `date`：问题发生的日期，用于查询对应日期的历史天气
- `user_setting`：定义 `UserAgent` 的知识边界，包含已知信息、模糊感知和不会主动披露的隐藏线索
- `uncertainty`：标记该 case 是否存在诊断不确定性（用于 UHA 评估）
- `required_directions`：评估 DCS 指标时使用的关键方向列表

---

## 记忆功能

`agents/memory_store.py` 中的 `get_memory()` 当前为占位符，返回空字符串。接入真实来源（向量检索、用户档案等）时，只需修改该函数。

```bash
# 关闭记忆注入
MEMORY_ENABLED=false python main.py
```

---

## 文档上传（RAG）

通过 Streamlit UI 上传 PDF / TXT / EPUB 文档，自动解析并存入向量库：

```bash
streamlit run app_update_rag_doc.py
```

---

## Todo

- [x] CLI 多轮咨询对话（Benchmark 模式）
- [x] 自由对话模式
- [x] 模型路由（fast / strong / think）
- [x] 流式输出（含思考过程展示）
- [x] 批量评测（AQT / UFS / DCS / AS / UHA）
- [x] Planner + State Manager 结构化对话循环
- [x] 工具调用（天气，Open-Meteo Archive API）
- [x] 运行日志持久化（JSONL）
- [ ] 向量库构建与检索
- [ ] 记忆功能接入真实来源
- [ ] 意图识别与查询改写
