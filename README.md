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
