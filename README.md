# Cat Behavior RAG

基于 DashScope 的猫行为咨询系统，结合 RAG 检索增强生成与多轮对话，模拟专业的猫行为学顾问与猫主人之间的咨询流程。

---

## 项目结构

```
CatBehaviorRag/
├── main.py                          # CLI 入口
├── Config.py                        # 全局配置（模型、路由、记忆、路径等）
├── agents/
│   ├── behaviorist_agent.py         # 行为专家 Agent（提问 → 深度建议）
│   ├── user_agent.py                # 猫主人 Agent（用于 benchmark 测试）
│   └── memory_store.py              # 记忆存储（当前为占位符）
├── utils/
│   ├── llm_client.py                # DashScope LLM 封装（流式 + 思考模型）
│   └── benchmark_loader.py          # benchmark JSON 加载与解析
├── Data/
│   └── enquiries_benchmark.json     # 2 条 benchmark 测试用例
├── rag_document_uploader/
│   └── markdown_cleaner.py          # 文档清洗模块
└── app_clean_rag_doc.py             # 文档清洗 CLI
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

系统使用 DashScope（阿里云百炼）API。通过环境变量设置：

```bash
# Linux / macOS
export DASHSCOPE_API_KEY=your_api_key_here

# Windows CMD
set DASHSCOPE_API_KEY=your_api_key_here

# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"
```

---

## 咨询系统使用说明

### 运行 Benchmark 测试

使用预设的测试用例，由 AI 模拟猫主人与专家进行对话：

```bash
# 运行第一个 case（默认）
python main.py

# 指定 case
python main.py --case-id cat_single_case_01
python main.py --case-id cat_single_case_02

# 查看所有可用 case
python main.py --list-cases
```

### 自由对话模式

直接与行为专家对话，由真实用户扮演猫主人：

```bash
python main.py --free-chat
# 或简写
python main.py -f
```

对话过程中输入 `q`、`quit` 或 `退出` 结束。

专家收集到足够信息后，会自动切换到深度思考模型给出最终建议。

---

## 对话流程说明

```
用户（或 UserAgent）发送初始描述
         ↓
行为专家提问（强模型，流式输出）
         ↓
用户回答（自由对话：真人输入；benchmark：UserAgent 生成）
         ↓
       ... 多轮 ...
         ↓
行为专家判断信息足够 → 切换思考模型，给出深度分析与建议（流式，含思考过程）
         ↓
对话结束（或达到 10 轮上限后强制结束）
```

**思考过程展示**：最终建议使用深度思考模型，思考内容会以灰色 `[思考过程] ... [思考结束]` 的形式显示在终端。

---

## 模型路由

系统根据任务复杂度自动选用不同模型，在性能与成本之间取得平衡：

| 角色 | 默认模型 | 用途 |
|---|---|---|
| `fast` | `qwen-turbo` | 低成本分类（判断是否披露事实） |
| `strong` | `qwen-plus` | 正常问答轮次 |
| `think` | `qwen3-32b` | 最终深度建议（启用 `enable_thinking`） |

### 关闭路由（消融实验）

路由关闭后，所有调用均使用 `consultation_model`（默认 `qwen-plus`），不使用思考模型：

```bash
# CLI 参数
python main.py --no-routing --case-id cat_single_case_01

# 环境变量
ROUTING_ENABLED=false python main.py --free-chat
```

### 覆盖模型配置

```bash
MODEL_FAST=qwen-turbo \
MODEL_STRONG=qwen-max \
MODEL_THINK=qwen3-32b \
python main.py --free-chat
```

---

## 记忆功能

记忆功能用于将历史会话信息注入行为专家的 system prompt，使专家能结合历史背景进行更个性化的建议。

**当前状态**：`agents/memory_store.py` 中的 `get_memory()` 为占位符，始终返回空字符串。接入真实记忆来源（向量检索、用户档案等）时，只需修改该函数。

### 记忆开关

```bash
# 关闭记忆注入（env）
MEMORY_ENABLED=false python main.py --free-chat

# Config.py 中默认值
memory_enabled = True
```

---

## 全部配置项（Config.py）

所有配置均可通过环境变量在运行时覆盖：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | *(必填)* | DashScope API 密钥 |
| `CONSULTATION_MODEL` | `qwen-plus` | 路由关闭时的默认模型 |
| `MAX_CONVERSATION_ROUNDS` | `10` | 最大对话轮数（超出强制结束） |
| `ROUTING_ENABLED` | `true` | 模型路由总开关 |
| `MODEL_FAST` | `qwen-turbo` | 快速分类模型 |
| `MODEL_STRONG` | `qwen-plus` | 问答对话模型 |
| `MODEL_THINK` | `qwen3-32b` | 深度思考模型 |
| `MODEL_THINK_ENABLE_THINKING` | `true` | 是否启用 qwen3 的 thinking 模式 |
| `MEMORY_ENABLED` | `true` | 记忆注入总开关 |
| `EMBEDDING_PROVIDER` | `dashscope` | 向量化服务：`dashscope` / `local` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地向量模型（provider=local 时生效） |
| `LOCAL_EMBEDDING_DEVICE` | `cuda` | 本地向量模型设备（`cuda` / `cpu`） |

---

## CLI 参数速查

```
python main.py [选项]

选项：
  --case-id ID        指定 benchmark case（默认使用第一个）
  --list-cases        列出所有可用 case ID 后退出
  --free-chat, -f     自由对话模式（不加载 benchmark）
  --no-routing        关闭模型路由，全部使用 consultation_model（消融实验用）
```

---

## Benchmark 数据格式

`Data/enquiries_benchmark.json` 中每条用例包含：

```json
{
  "case_id": "cat_single_case_01",
  "initial_user_message": "新来的猫特别怕人，老猫会一直盯着它。",
  "user_state": {
    "initially_known": ["..."],
    "discoverable_facts": [{"fact": "...", "revealed_when_asked_about": ["..."]}],
    "user_beliefs": ["...（可能不准确的固有认知）"]
  },
  "ground_truth": {
    "primary_issue": "...",
    "critical_facts": ["..."],
    "accepted_conclusions": ["..."],
    "rejected_conclusions": ["..."]
  },
  "reference_solution": "完整参考解决方案..."
}
```

`UserAgent` 只能根据 `user_state` 中的信息回答，`discoverable_facts` 仅在专家问到相关话题时才会披露（通过 LLM 语义判断）。

---

## 数据与 RAG 部分

### Embedding Provider 配置

```bash
# 使用 DashScope（默认）
EMBEDDING_PROVIDER=dashscope DASHSCOPE_API_KEY=your_key python ...

# 使用本地模型（推荐有显卡时）
EMBEDDING_PROVIDER=local LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 LOCAL_EMBEDDING_DEVICE=cuda python ...
```

### 文档清洗

```bash
# 默认运行（需要 DASHSCOPE_API_KEY）
python app_clean_rag_doc.py

# 禁用 LLM，仅规则清洗
python app_clean_rag_doc.py --no-llm
```

### 语料统计

| 指标 | 数值 |
|---|---|
| 文档数 | 541 |
| 有效总字数 | 350,891 |
| 平均每 doc 字数 | 648.6 |
| 预计入库片段数 | 2,551 |

---

## 评估（Ragas）

系统集成 [Ragas](https://github.com/explodinggradients/ragas) 对 RAG 质量进行自动化评估，无需人工标注：

- **忠实度**（Faithfulness）：答案是否基于检索内容，避免幻觉
- **答案相关性**（Answer Relevance）：答案是否切题
- **上下文相关性**（Context Relevance）：检索内容是否精炼

---

## Todo

- [x] 文档格式转换（PDF / TXT → 纯文本）
- [x] 文档清洗与 Markdown 化
- [x] CLI 多轮咨询对话（Benchmark 模式）
- [x] 自由对话模式
- [x] 模型路由（fast / strong / think）
- [x] 流式输出（含思考过程展示）
- [x] 记忆模块占位符
- [ ] 向量库构建与检索
- [ ] 记忆功能接入真实来源
- [ ] 意图识别与查询改写
- [ ] 完整 Ragas 评估流水线
