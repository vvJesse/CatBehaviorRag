# Cat Behavior RAG

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

## Data Cleaning

新增了一个面向 OCR/抽取长文本的 Markdown 清洗脚本：`app_clean_rag_doc.py`。

### 清洗策略

1. 先做规则预清洗：统一换行与空格，移除页码、页眉/页脚这类明显版面噪声。
2. 再按字符数把全文切成多个片段，避免把整本书直接送给 LLM。
3. 每个片段单独送给 DashScope 做“只整理、不创作”的 Markdown 化：保留原文顺序，恢复标题层级和自然段，删除重复页码页眉。
4. 最后把各片段合并，并做一次 Markdown 后处理，去掉相邻重复标题和多余空行。

### 关键约束

- 输出必须与原文对应，不允许补充、总结或改写事实。
- 标题层级只使用 `#` 和 `##`，不确定就宁可保守，不强造标题。
- 如果本地没有 DashScope API Key，可以使用 `--no-llm` 先跑规则版清洗；有 Key 时建议默认启用 LLM 精修。

### 运行方式

```bash
source .venv/bin/activate
python app_clean_rag_doc.py
```

默认会读取 `RagDocumentUploader/data-cleaning-example/家有恶猫-片段.txt`，并将结果写入 `RagDocumentUploader/cleaned_data/家有恶猫-片段.cleaned.md`。

如需禁用 DashScope：

```bash
source .venv/bin/activate
python app_clean_rag_doc.py --no-llm
```
