from __future__ import annotations

import copy
import json
import logging
from typing import Iterator, Literal

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils.llm_client import LLMClient
from utils.tools import TOOL_MAP, TOOLS

logger = logging.getLogger(__name__)

HypothesisConfidence = Literal["Unexplored", "Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class HypothesisItem(BaseModel):
    description: str = Field(description="假设内容描述")
    confidence: HypothesisConfidence = Field(description="当前置信度")


class ConsultState(BaseModel):
    user_initial_query: str
    consultation_date: str | None = None
    consultation_latitude: str | None = None
    consultation_longitude: str | None = None
    rewritten_initial_query: str | None = None
    user_response_this_round: str | None = None
    hypothesis: list[HypothesisItem] = Field(default_factory=list)
    collected_evidence: list[str] = Field(default_factory=list)
    resolved: bool = False

    def to_prompt_str(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Agent response schemas
# ---------------------------------------------------------------------------

class IntermediateResponse(BaseModel):
    type: Literal["tool_call", "end_tool"] = Field(
        description="tool_call=需要调用工具；end_tool=无需继续调用工具"
    )
    thought: str = Field(description="推理过程")
    tool_name: str | None = Field(
        default=None,
        description="type=tool_call 时填写工具名称，否则为 null",
    )
    tool_args: dict | None = Field(
        default=None,
        description="type=tool_call 时填写工具参数，否则为 null",
    )

    def is_tool_call(self) -> bool:
        return self.type == "tool_call"

    def end_tool_call(self) -> bool:
        return self.type == "end_tool"


class ConsultResponse(BaseModel):
    text: str = Field(description="向用户说的话（问题或回应）")
    end: bool = Field(
        default=False,
        description="是否认为已收集到足够信息，准备给出最终建议",
    )


class InitialHypotheses(BaseModel):
    hypothesis: list[HypothesisItem] = Field(
        description="基于初始问题生成的3-5个诊断方向，每条只是简短的大方向名称",
    )


# --- 状态更新操作函数（由 LLM 以 tool_call 形式调用）---

class SetHypothesisConfidence(BaseModel):
    """调整已有假设方向的置信度"""
    description: str = Field(description="假设方向名称，须与现有假设列表中的名称完全一致")
    confidence: HypothesisConfidence = Field(description="调整后的置信度")


class AddHypothesis(BaseModel):
    """添加新的假设方向（初始置信度自动设为 Unexplored）"""
    description: str = Field(description="新假设方向的简短名称，2-5字，如医疗原因、环境应激")


class RemoveHypothesis(BaseModel):
    """移除已被明确排除的假设方向"""
    description: str = Field(description="要移除的假设方向名称，须与现有假设列表中的名称完全一致")


class AddEvidence(BaseModel):
    """将用户本轮回答中的关键信息记录为证据"""
    evidence: str = Field(description="从用户本轮回答中提取的关键事实，一句话描述")


class RewriteQuery(BaseModel):
    """发现用户原始问题存在误解时，重写结构化问题描述"""
    rewritten_query: str = Field(
        description="修正后的结构化问题描述，严格基于已知事实，去除误解性表述"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = """\
你是一位猫行为学专家。用户刚刚发来了一条初始咨询消息。
请将这条消息改写为一个清晰、结构化的问题描述，便于后续诊断分析。
为了避免被用户的认知带偏，你会严格区分用户观察到的现象和ta的主观解释，并且不轻易接受ta的假设和解释。
例子：
“我的猫为了报复我，一直在抓沙发。”
改写后应该是：
“我的猫抓沙发，我认为是因为他要报复我。这个说法对吗？”

要求：
- 这样修改后，方便后续去验证假设和解释，并探索猫行为背后的真正原因。
只输出改写后的文本，不要加任何解释。"""

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _REWRITE_SYSTEM),
    ("human", "{query}"),
])

# ---
_INIT_HYPOTHESIS_SYSTEM = """\
你是一位专业的猫行为学专家。根据用户的初始咨询，生成诊断方向列表。

要求：
- 生成3-5个假设方向，覆盖不同维度，如：医疗/生理、行为习得、环境应激、主人互动、社交/领地等
- 每个方向只需一个简短标签（2-5字），不要写详细解释，例如：医疗原因、环境应激、行为习得、主人互动
- 方向之间须有明显差异，严禁同质化（如焦虑和应激不能作为两个独立方向）
- 主动考虑用户可能没有意识到的因素（如无意间的强化行为、就医指征等）
- 初始置信度一律设为 Unexplored，表示这些只是待探索方向，不能据此直接下结论"""

_INIT_HYPOTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _INIT_HYPOTHESIS_SYSTEM),
    ("human", "用户原始问题：{query}\n\n改写后的结构化问题：{rewritten_query}\n\n请生成初始假设列表。"),
])

# ---

_UPDATE_STATE_SYSTEM = """\
你是一位专业的猫行为学专家，正在根据本轮对话更新诊断状态。

当前诊断状态（State）：
{state}

用户本轮回答：
{user_response}

你有以下五个函数可以调用，可同时调用多个：
- SetHypothesisConfidence(description, confidence)：调整某个方向的置信度（Unexplored/Low/Medium/High），description 须与列表中名称完全一致
- AddHypothesis(description)：添加新的假设方向，description 为简短标签（2-5字），初始置信度设为Unexplored，表示未探索，必须有确切证据才能调整置信度。
- RemoveHypothesis(description)：移除已被明确排除的方向，description 须与列表中名称完全一致
- AddEvidence(evidence)：将用户本轮回答中的关键事实提取为一句话证据并记录
- RewriteQuery(rewritten_query)：当发现用户原始问题存在误解（如主人混淆了概念，提出的问题不符合意图）时，重写结构化问题描述；通常同时配合 RemoveHypothesis 删除基于误解生成的旧方向，并用 AddHypothesis 补充正确方向

决策原则：
- 用户回答中的关键事实（时间线、频率、触发条件、环境变化等）都应用 AddEvidence 记录
- 发现原始问题存在误解时才调用 RewriteQuery，并同步清理受影响的假设方向
- Unexplored 表示尚未探索，只有拿到明确支持或反证后才调整为 Low/Medium/High
- 本轮信息支持某方向 → 提升置信度；信息否定某方向 → 降低或移除；信息揭示新方向 → 添加
- 未受本轮信息影响的方向无需操作
- 保持方向多样性，不要过早收敛到单一结论"""

_UPDATE_STATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _UPDATE_STATE_SYSTEM),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "请根据以上对话调用相应函数更新诊断状态。"),
])

# ---
_TOOL_SYSTEM = """\
你是一位猫行为学专家，正在使用工具收集诊断所需信息。

当前诊断状态（State）：
{state}

你的局部工具操作记录（Trajectory）：
{trajectory}

可用工具：
{tool_descriptions}

决策规则：
- 如果需要调用工具，输出 type=tool_call，填写 thought、tool_name、tool_args
- 如果已无需继续调用工具，输出 type=end_tool，填写 thought
- 每次只调用一个工具"""

_TOOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _TOOL_SYSTEM),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "请决定是否需要调用工具，以及调用哪个工具。"),
])

# ---

_CONSULT_SYSTEM = """\
你是一位专业的猫行为学顾问，正在与猫主人进行多轮咨询对话。

当前诊断状态（State）：
{state}

职责：
- 根据当前 state 和对话历史，向用户提1~3个关键问题，逐步收集诊断信息。语气温和专业。
- 当还有假设方向未被验证时，必须继续提出问题，直到所有方向都已验证。
- state 的假设存在不够完善的可能，这时必须提出新的假设方向，并继续向用户提问。
- 当你认为已经收集够了证据，并且不需要增加新的假设时，将 end 设为 true
- end=true 时，text 必须是一句说明性陈述，明确告知用户已收集到足够信息、将进行深度分析，不得再提问
- 用中文回复"""

_CONSULT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CONSULT_SYSTEM),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "请根据当前状态向用户提问，或决定结束问诊。"),
])

# ---

_THINK_SYSTEM = """\
你是一位专业的猫行为学顾问。问诊已结束，请根据完整的诊断状态和对话历史，给出详细的分析和建议。

当前诊断状态（State）：
{state}

输出要求：
- 根据收集到的信息，判断猫咪是否真的存在异常。
- 如果存在，则给出具体可操作的建议；如果不存在，则明确告知用户可能的替代性解释，避免误导用户。
- 如果有假设存在不确定性，明确说明。
- 给出具体可操作的建议。
- 你的回复在保持专业性的同时也要用户友好，避免使用太生硬、冗长的文字。
- 用中文回复"""

_THINK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _THINK_SYSTEM),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "请给出完整的诊断分析和建议。"),
])


# ---------------------------------------------------------------------------
# State update helpers (pure functions)
# ---------------------------------------------------------------------------

def _update_state_with_tool_result(state: ConsultState, tool_name: str, result: str) -> ConsultState:
    """将工具调用结果加入 collected_evidence，返回新 state。"""
    new_state = state.model_copy(deep=True)
    new_state.collected_evidence.append(f"[{tool_name}] {result}")
    return new_state


def _format_trajectory(trajectory: list[dict]) -> str:
    if not trajectory:
        return "（本轮尚未调用任何工具）"
    lines = []
    for i, step in enumerate(trajectory, 1):
        lines.append(
            f"步骤 {i}：thought={step['thought']} | "
            f"tool={step['tool_name']}({step['tool_args']}) → {step['observation']}"
        )
    return "\n".join(lines)


def _format_tool_descriptions() -> str:
    return "\n".join(
        f"- {t.name}: {t.description}" for t in TOOLS
    )


# ---------------------------------------------------------------------------
# ConsultantAgent
# ---------------------------------------------------------------------------

class ConsultantAgent:
    """统一的咨询 Agent：负责 query rewrite、tool 调用、正式提问和最终 think。"""

    def __init__(self, llm_strong: LLMClient, llm_think: LLMClient) -> None:
        self._llm_strong = llm_strong
        self._llm_think = llm_think

        self._rewrite_chain = (
            _REWRITE_PROMPT | llm_strong.chat_model
        )
        self._init_hypothesis_chain = (
            _INIT_HYPOTHESIS_PROMPT
            | llm_strong.chat_model.bind_tools(
                [InitialHypotheses],
                tool_choice={"type": "function", "function": {"name": "InitialHypotheses"}},
            )
            | PydanticToolsParser(tools=[InitialHypotheses], first_tool_only=True)
        )
        _update_ops = [SetHypothesisConfidence, AddHypothesis, RemoveHypothesis, AddEvidence, RewriteQuery]
        self._update_state_chain = (
            _UPDATE_STATE_PROMPT
            | llm_strong.chat_model.bind_tools(_update_ops)
            | PydanticToolsParser(tools=_update_ops, first_tool_only=False)
        )
        self._tool_chain = (
            _TOOL_PROMPT
            | llm_strong.chat_model.bind_tools(
                [IntermediateResponse],
                tool_choice={"type": "function", "function": {"name": "IntermediateResponse"}},
            )
            | PydanticToolsParser(tools=[IntermediateResponse], first_tool_only=True)
        )
        self._consult_chain = (
            _CONSULT_PROMPT
            | llm_strong.chat_model.bind_tools(
                [ConsultResponse],
                tool_choice={"type": "function", "function": {"name": "ConsultResponse"}},
            )
            | PydanticToolsParser(tools=[ConsultResponse], first_tool_only=True)
        )
        self._think_chain = (
            _THINK_PROMPT | llm_think.chat_model
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(name="rewrite_initial_query", run_type="chain")
    def rewrite_initial_query(self, query: str) -> str:
        result = self._rewrite_chain.invoke({"query": query})
        return result.content

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    @traceable(name="initialize_hypotheses", run_type="chain")
    def initialize_hypotheses(self, query: str, rewritten_query: str) -> list[HypothesisItem]:
        """根据初始问题生成多维度假设列表。"""
        result: InitialHypotheses = self._init_hypothesis_chain.invoke({
            "query": query,
            "rewritten_query": rewritten_query,
        })
        if result is None:
            raise ValueError("initialize_hypotheses 返回 None，触发重试")
        return result.hypothesis

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    @traceable(name="update_state_hypotheses", run_type="chain")
    def update_state(self, state: ConsultState, history: list[BaseMessage], user_response: str) -> ConsultState:
        """根据本轮对话以函数调用方式更新假设方向，返回新 state（不修改原对象）。"""
        operations: list = self._update_state_chain.invoke({
            "state": state.to_prompt_str(),
            "user_response": user_response,
            "history": history,
        })
        if not operations:
            # LLM 认为本轮无需更新，直接返回原 state 副本
            return state.model_copy(deep=True)
        new_state = state.model_copy(deep=True)
        for op in operations:
            if isinstance(op, SetHypothesisConfidence):
                for h in new_state.hypothesis:
                    if h.description == op.description:
                        h.confidence = op.confidence
                        break
            elif isinstance(op, AddHypothesis):
                if not any(h.description == op.description for h in new_state.hypothesis):
                    new_state.hypothesis.append(
                        HypothesisItem(description=op.description, confidence="Unexplored")
                    )
            elif isinstance(op, RemoveHypothesis):
                new_state.hypothesis = [
                    h for h in new_state.hypothesis if h.description != op.description
                ]
            elif isinstance(op, AddEvidence):
                new_state.collected_evidence.append(op.evidence)
            elif isinstance(op, RewriteQuery):
                new_state.rewritten_initial_query = op.rewritten_query
        return new_state

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    @traceable(name="generate_intermediate_response", run_type="chain")
    def generate_intermediate_response(
        self,
        state: ConsultState,
        history: list[BaseMessage],
        trajectory: list[dict],
    ) -> IntermediateResponse:
        """内循环决策：决定是否调用工具，或结束工具调用阶段。"""
        result = self._tool_chain.invoke({
            "state": state.to_prompt_str(),
            "trajectory": _format_trajectory(trajectory),
            "tool_descriptions": _format_tool_descriptions(),
            "history": history,
        })
        return result

    def execute_tool(self, response: IntermediateResponse) -> str:
        """执行工具调用，返回结果字符串。"""
        tool_fn = TOOL_MAP.get(response.tool_name or "")
        if not tool_fn:
            return f"未知工具: {response.tool_name}"
        try:
            return tool_fn.invoke(response.tool_args or {})
        except Exception as e:
            logger.warning("工具 %s 调用失败: %s", response.tool_name, e)
            return f"工具调用失败：{e}"

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    @traceable(name="generate_consult_response", run_type="chain")
    def generate_response(
        self,
        state: ConsultState,
        history: list[BaseMessage],
        trajectory: list[dict] | None,
    ) -> ConsultResponse:
        """外循环决策：根据当前 state 和历史生成正式回复。"""
        result = self._consult_chain.invoke({
            "state": state.to_prompt_str(),
            "history": history,
        })
        return result

    @traceable(name="think_final", run_type="llm")
    def think(self, state: ConsultState, history: list[BaseMessage]) -> Iterator[str]:
        """流式输出最终诊断建议（使用思考模型）。"""
        chain = _THINK_PROMPT | self._llm_think.chat_model
        return self._llm_think.stream_chat_lc(chain, {
            "state": state.to_prompt_str(),
            "history": history,
        })
