# Cat Behavior RAG

## Embedding Provider 配置

项目现在支持两种 embedding provider：

- `dashscope`：云端向量服务（默认）
- `local`：本地 HuggingFace/sentence-transformers 模型（可使用 GPU）

通过环境变量切换：

```bash
# 1) 使用 DashScope（默认）
EMBEDDING_PROVIDER=dashscope
DASHSCOPE_API_KEY=your_api_key

# 2) 使用本地模型（推荐有显卡时）
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
LOCAL_EMBEDDING_DEVICE=cuda
```

说明：

- `LOCAL_EMBEDDING_DEVICE` 可设为 `cuda`、`cpu` 或 `cuda:0`
- 若使用 `local`，请确保已安装 `langchain-huggingface` 与 `sentence-transformers`

## Todo List

- [x] Convert file to pure text (support pdf and txt)
- [ ] Chunk and retrieve data from database.
- [ ] Form a small benchmark, with various queries.
- [ ] Intent Recognition and Query Rewrite

## Statistic

- 文档数: 541
- 有效总字数（不含空白）: 350891
- 平均每个 doc 的有效字数: 648.6
- doc 字数中位数: 632.0
- doc 字数 P95: 1019
- 最长 doc 字数: 2260
- 按当前切分配置预计入库片段数: 2551
- 因切分新增的片段数: 2010
- 触发切分的 doc 数: 488
- 最长 doc: 猫应用行为学.txt#L127 (2260)

### 每个文件的摘要
- 猫应用行为学.txt | doc 数 272 | 有效字数 202824 | 平均 doc 字数 745.7 | 最长 doc 2260 | 触发切分 doc 257 | 新增片段 1205
- 家有恶猫.txt | doc 数 269 | 有效字数 148067 | 平均 doc 字数 550.4 | 最长 doc 841 | 触发切分 doc 231 | 新增片段 805

### 最长的 10 个 doc
1. 猫应用行为学.txt#L127 | 字数 2260 | 预计入库片段 16 | 新增片段 15
2. 猫应用行为学.txt#L175 | 字数 2208 | 预计入库片段 15 | 新增片段 14
3. 猫应用行为学.txt#L211 | 字数 2166 | 预计入库片段 16 | 新增片段 15
4. 猫应用行为学.txt#L209 | 字数 2148 | 预计入库片段 16 | 新增片段 15
5. 猫应用行为学.txt#L95 | 字数 2134 | 预计入库片段 15 | 新增片段 14
6. 猫应用行为学.txt#L213 | 字数 2111 | 预计入库片段 16 | 新增片段 15
7. 猫应用行为学.txt#L91 | 字数 2092 | 预计入库片段 15 | 新增片段 14
8. 猫应用行为学.txt#L173 | 字数 2074 | 预计入库片段 15 | 新增片段 14
9. 猫应用行为学.txt#L129 | 字数 2070 | 预计入库片段 16 | 新增片段 15
10. 猫应用行为学.txt#L277 | 字数 2035 | 预计入库片段 15 | 新增片段 14

# Ragas

Ragas（Retrieval Augmented Generation Assessment）是一个旨在对检索增强生成（RAG）自动化、无参考（reference-free）评估的框架
。由于 RAG 系统由检索模块和基于大语言模型（LLM）的生成模块组成，评估这些架构非常具有挑战性
。Ragas 的核心优势在于它不需要依赖人工标注的标准答案（Ground Truth），即可对 RAG 系统的不同维度进行评估
。
Ragas 主要通过以下三个关键维度来评估 RAG 系统：
1. 忠实度 (Faithfulness)
忠实度评估生成的答案是否完全基于所检索到的上下文，以避免模型产生幻觉
。
评估流程：首先利用 LLM 将生成的答案分解为若干个简短且集中的陈述（statements）
。接着，再次使用 LLM 验证每一个陈述是否可以从检索到的上下文中推导出来
。
计算方式：忠实度得分等于被支持的陈述数量与陈述总数的比值
。
2. 答案相关性 (Answer Relevance)
该指标评估生成的答案是否直接且恰当地回答了用户提出的问题，而不考虑其事实准确性（事实性由忠实度衡量）
。
评估流程：Ragas 会提示 LLM 根据生成的答案反向生成 n 个潜在的问题
。然后，使用嵌入模型（如 text-embedding-ada-002）获取这些生成问题与原始问题之间的向量，并计算它们之间的余弦相似度
。
计算方式：最终得分为所有生成的潜在问题与原问题相似度的平均值
。
3. 上下文相关性 (Context Relevance)
评估检索到的上下文是否精炼，是否只包含回答问题所需的关键信息，旨在惩罚包含冗余信息的情况
。
评估流程：给定问题和上下文，LLM 会从上下文中提取出对回答该问题至关重要的句子子集
。
计算方式：得分由提取出的关键句子数量与上下文中句子总数的比值决定
。

## Retrieval Evaluation

- 对于Retrieval的部分，使用上下文相关性来评估，无需依赖LLM生成的答案。 
- 完成LLM的问答后，再将另外两个metrics加入评估。