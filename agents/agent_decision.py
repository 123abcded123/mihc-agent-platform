"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
【中文注释】本文件是整个多智能体医疗助手系统的调度核心。
用 LangGraph 搭建状态机（StateGraph），负责：
  1. 入口安检 + 图片类型识别（analyze_input）
  2. LLM 决策路由（route_to_agent）——决定交给哪个专职 Agent
  3. 六个执行节点：对话(CONVERSATION) / 知识库检索(RAG) / 联网搜索(WEB_SEARCH) / 脑肿瘤(BRAIN_TUMOR) / 胸片(CHEST_XRAY) / 皮肤病变(SKIN_LESION)
  4. 人工审核检查（check_validation）→ 人工确认（human_validation）
  5. 出口安检（apply_guardrails）→ 结束（END）
状态包 AgentState 贯穿所有节点，实现节点间信息传递。
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config

load_dotenv()

# Load configuration
# 【中文注释】加载全局配置对象（模型、各 Agent 参数、阈值等均从这里读取）
config = Config()

# Initialize memory
# 【中文注释】内存版检查点存储：保存每次对话后的图状态，实现多轮记忆（进程重启后丢失）
memory = MemorySaver()

# Specify a thread
# 【中文注释】固定线程 ID=1，所有请求共享同一条会话线程，从而共享同一段对话历史
thread_config = {"configurable": {"thread_id": "1"}}


# Agent that takes the decision of routing the request further to correct task specific agent
# 【中文注释】决策系统的静态配置类：决策模型、阈值、决策提示词、图片分析器都挂在这里
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # 【中文注释】配置类：集中存放调度所需的模型名称、信心阈值、系统提示词和共享的图片分析器
    # Decision model
    # 【中文注释】路由决策所用的 LLM 模型名
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    
    # Vision model for image analysis
    # 【中文注释】图片分析所用的视觉模型名
    VISION_MODEL = "gpt-4o"
    
    # Confidence threshold for responses
    # 【中文注释】决策信心阈值：低于 0.85 的决策视为不可靠，会被改道
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    # 【中文注释】路由决策 LLM 的系统提示词：说明六个候选 Agent 的职责与路由规则
    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
    2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    4. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    5. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    6. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.

    Make your decision based on these guidelines:
    - If the user has not uploaded any image, always route to the conversation agent.
    - If the user uploads a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    - If the user asks about recent medical developments or current health situations, use the web search pocessor agent.
    - If the user asks specific medical knowledge questions, use the RAG agent.
    - For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    # 【中文注释】共享的图片分析器实例：负责识别图片类型、胸片分类、皮肤病变分割
    image_analyzer = ImageAnalysisAgent(config=config)


# 【中文注释】贯穿整个图流程的状态包：每个节点读取并返回该字典，实现节点间数据传递
class AgentState(MessagesState):
    """State maintained across the workflow."""
    # 【中文注释】继承 MessagesState，自带 messages 字段用于累积对话历史
    # messages: List[BaseMessage]  # Conversation history
    # 【中文注释】当前激活的 Agent 名；多个 Agent 参与时会用逗号叠加（如 "RAG_AGENT, WEB_SEARCH_PROCESSOR_AGENT"）
    agent_name: Optional[str]  # Current active agent
    # 【中文注释】当前待处理的用户输入（纯文本字符串，或含 text/image 的字典）
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    # 【中文注释】本轮输入是否包含图片
    has_image: bool  # Whether the current input contains an image
    # 【中文注释】图片类型（如 brain_tumor / chest_xray / skin_lesion），无图片时为 None
    image_type: Optional[str]  # Type of medical image if present
    # 【中文注释】各节点产出的最终回复内容（AIMessage 或字符串）
    output: Optional[str]  # Final output to user
    # 【中文注释】是否需要人工审核（影像类诊断结果强制置 True）
    needs_human_validation: bool  # Whether human validation is required
    # 【中文注释】RAG 检索的信心分（0.0~1.0），用于决定是否改道联网搜索
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    # 【中文注释】安检拦截标记：为 True 时跳过路由决策，直接走出口安检
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    # 【中文注释】RAG 答复信息不足标记：为 True 时改道联网搜索
    insufficient_info: bool  # Flag indicating RAG response has insufficient information


# 【中文注释】决策模型的输出结构（TypedDict）：JSON 解析器的目标格式
class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    # 【中文注释】决策出的 Agent 名（六个候选之一）
    agent: str
    # 【中文注释】决策理由（LLM 的逐步推理文本）
    reasoning: str
    # 【中文注释】决策信心值（0.0~1.0）
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""
    # 【中文注释】构建并编译 LangGraph 状态机：定义全部节点、条件边与出口，是整个调度流程的骨架

    # Initialize guardrails with the same LLM used elsewhere
    # 【中文注释】出口安检器：与 RAG 共用同一个 LLM 实例（同一模型处理能力）
    guardrails = LocalGuardrails(config.rag.llm)

    # LLM
    # 【中文注释】路由决策所用的 LLM 实例
    decision_model = config.agent_decision.llm
    
    # Initialize the output parser
    # 【中文注释】把 LLM 返回的 JSON 解析成 AgentDecision 结构
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    # 【中文注释】决策提示词模板：系统提示 + 人类输入占位符
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    # 【中文注释】决策链：提示词 → 模型 → JSON 解析器，一次 invoke 完成"输入→决策结构"
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    # 【中文注释】========== 图内节点函数定义区（闭包内定义，供 StateGraph 注册）==========
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        # 【中文注释】节点1：入口分析。读取输入文本做安检；若含图片则调用视觉模型识别图片类型
        # 输入字段：current_input
        # 输出字段：has_image、image_type、bypass_routing；被安检拦截时还会写 messages、agent_name
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        # 【中文注释】从输入中提取纯文本：输入可能是字符串或 {"text": ..., "image": ...} 字典
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Check input through guardrails if text is present
        # 【中文注释】入口安检：文本不合法（如违规内容）则直接拦截
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                # 【中文注释】安检未通过：置 bypass_routing=True，后续条件边将绕过路由直接去出口
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True  # flag to end flow
                }
        
        # Original image processing code
        # 【中文注释】图片处理：字典输入且带 image 字段时，用视觉模型识别图片类型（决定路由到哪个影像 Agent）
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        # 【中文注释】正常流程：写回图片相关字段，并显式将 bypass_routing 置 False
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Explicitly set to False for normal flow
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        # 【中文注释】条件边函数：安检未通过则绕过路由直接出口；否则进入路由决策
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        # 【中文注释】节点2：路由决策。把用户问题+近期上下文+图片信息交给决策 LLM，选出一个专职 Agent
        # 输入字段：messages、current_input、has_image、image_type
        # 返回字段：agent_state（写回 agent_name）与 next（下一步目标节点名）
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        # Prepare input for decision model
        # 【中文注释】再次提取输入文本，作为决策 LLM 的主要判断依据
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        # 【中文注释】取最近 6 条消息（约 3 轮问答）拼成上下文，帮助决策 LLM 理解对话语境
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)  # Not provided control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        # 【中文注释】拼装决策输入：当前问题 + 近期上下文 + 是否含图及图片类型
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # Make the decision
        # 【中文注释】调用决策链：提示词 → LLM → JSON 解析，得到 agent/reasoning/confidence
        decision = decision_chain.invoke({"input": decision_input})

        # Decided agent
        print(f"Decision: {decision['agent']}")
        
        # Update state with decision
        # 【中文注释】先把决策出的 Agent 名写回状态
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # Route based on agent name and confidence
        # 【中文注释】信心低于 0.85 视为不可靠，走 needs_validation 分支（条件边映射表里默认丢给 RAG 兜底）
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        # 【中文注释】信心达标：next 直接取决策出的 Agent 名，条件边据此跳到对应执行节点
        return {"agent_state": updated_state, "next": decision["agent"]}

    # Define agent execution functions (these will be implemented in their respective modules)
    # 【中文注释】========== 六个专职执行节点 + 人工审核/安检节点定义区 ==========
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""
        # 【中文注释】执行节点：对话 Agent。把完整历史+当前问题交给对话 LLM 自由应答
        # 输入字段：messages、current_input
        # 输出字段：output（LLM 回复）、agent_name 置为 CONVERSATION_AGENT

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        
        # Prepare input for decision model
        # 【中文注释】提取当前输入文本
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history
        # 【中文注释】拼接完整对话历史（被注释掉的 -20 是旧版"只取最近 10 轮"的做法，现改为全量历史）
        recent_context = ""
        for msg in messages:#[-20:]:  # Get last 10 exchanges (20 messages)  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        # 【中文注释】对话提示词：设定医疗对话助手的角色、回答准则与回答格式
        conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}

        You are an AI-powered Medical Conversation Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and medical-related queries. You must respond naturally while ensuring medical accuracy and clarity.

        ### Role & Capabilities
        - Engage in **general conversation** while maintaining professionalism.
        - Answer **medical questions** using verified knowledge.
        - Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.
        - Handle **follow-up questions** while keeping track of conversation context.
        - Redirect **medical images** to the appropriate AI analysis agent.

        ### Guidelines for Responding:
        1. **General Conversations:**
        - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.
        - Keep responses **concise and engaging**, unless a detailed answer is needed.

        2. **Medical Questions:**
        - If you have **high confidence** in answering, provide a medically accurate response.
        - Ensure responses are **clear, concise, and factual**.

        3. **Follow-Up & Clarifications:**
        - Maintain conversation history for better responses.
        - If a query is unclear, ask **follow-up questions** before answering.

        4. **Handling Medical Image Analysis:**
        - Do **not** attempt to analyze images yourself.
        - If user speaks about analyzing or processing or detecting or segmenting or classifying any disease from any image, ask the user to upload the image so that in the next turn it is routed to the appropriate medical vision agents.
        - If an image was uploaded, it would have been routed to the medical computer vision agents. Read the history to know about the diagnosis results and continue conversation if user asks anything regarding the diagnosis.
        - After processing, **help the user interpret the results**.

        5. **Uncertainty & Ethical Considerations:**
        - If unsure, **never assume** medical facts.
        - Recommend consulting a **licensed healthcare professional** for serious medical concerns.
        - Avoid providing **medical diagnoses** or **prescriptions**—stick to general knowledge.

        ### Response Format:
        - Maintain a **conversational yet professional tone**.
        - Use **bullet points or numbered lists** for clarity when needed.
        - If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., "According to Mayo Clinic...").
        - If a user asks for a diagnosis, remind them to **seek medical consultation**.

        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"
        **You:** "I'm here and ready to help! How can I assist you today?"

        **User:** "I have a headache and fever. What should I do?"
        **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."

        Conversational LLM Response:"""

        # print("Conversation Prompt:", conversation_prompt)

        # 【中文注释】调用对话 LLM 生成回复
        response = config.conversation.llm.invoke(conversation_prompt)

        # print("Conversation respone:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        # 【中文注释】把 LLM 回复写入 output，并记录当前 Agent 名
        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # 【中文注释】执行节点：RAG Agent。基于已入库医学文献检索回答问题
        # 输入字段：messages、current_input
        # 输出字段：output、needs_human_validation(置 False)、retrieval_confidence、insufficient_info、agent_name
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        # 【中文注释】每次运行新建 MedicalRAG 实例（其内部含检索器与生成链）
        rag_agent = MedicalRAG(config)
        
        messages = state["messages"]
        query = state["current_input"]
        # 【中文注释】RAG 上下文窗口大小（从 config 读取，用于截取最近若干条历史）
        rag_context_limit = config.rag.context_limit

        # 【中文注释】按配置截取最近 rag_context_limit 条消息作为 RAG 的对话上下文
        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        # 【中文注释】执行检索增强生成，返回含 response/sources/confidence 的结果
        response = rag_agent.process_query(query, chat_history=recent_context)
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        # 【中文注释】检查 RAG 回复文本是否出现"信息不足"类措辞
        insufficient_info = False
        response_content = response["response"]
        
        # Extract the content properly based on type
        # 【中文注释】统一取回复文本：若是带 content 属性的消息对象则取 .content，否则直接当字符串用
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content
             
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        # 【中文注释】命中任一"信息不足"关键词短语即认为 RAG 无法可靠回答
        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text or 
            "I don't have enough information" in response_text or 
            "don't have enough information" in response_text.lower() or
            "not enough information" in response_text.lower() or
            "insufficient information" in response_text.lower() or
            "cannot answer" in response_text.lower() or
            "unable to answer" in response_text.lower()
            ):
            
            print("RAG response indicates insufficient information")
            print(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        print(f"Insufficient info flag set to: {insufficient_info}")

        # Store RAG output ONLY if confidence is high
        # 【中文注释】检索信心达标才保留答案；否则输出空消息（后续条件边会改道联网搜索）
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")
        
        # 【中文注释】写回状态：记录检索信心、信息不足标记；RAG 结果默认无需人工审核
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""
        # 【中文注释】执行节点：联网搜索 Agent。先检索网络再用 LLM 整理成最终答复
        # 输入字段：messages、current_input、agent_name
        # 输出字段：output、agent_name（在原有 Agent 名后叠加 ", WEB_SEARCH_PROCESSOR_AGENT"）

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        # 【中文注释】联网搜索的上下文窗口大小（从 config 读取）
        web_search_context_limit = config.web_search.context_limit

        # 【中文注释】截取最近若干条消息作为搜索查询的对话上下文
        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        # 【中文注释】新建联网搜索处理器并执行：网络检索 + LLM 精炼
        web_search_processor = WebSearchProcessorAgent(config)

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        # 【中文注释】Agent 名叠加：如从 RAG 改道而来，记录为 "RAG_AGENT, WEB_SEARCH_PROCESSOR_AGENT"，完整保留参与链路
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        # 【中文注释】用联网搜索的最终答复覆盖之前的输出（如 RAG 的空/低质量答案）
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # 【中文注释】RAG 节点后的条件边函数：根据检索信心与信息充分性决定去联网搜索还是进入人工审核检查
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # Redirect if confidence is low or if response indicates insufficient info
        # 【中文注释】检索信心低或信息不足时改道联网搜索，否则进入 check_validation
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis."""
        # 【中文注释】执行节点：脑肿瘤 Agent。目前为占位实现（固定文案回复）
        # 输入字段：无特殊依赖
        # 输出字段：output、needs_human_validation=True（影像诊断强制人工确认）、agent_name

        print(f"Selected agent: BRAIN_TUMOR_AGENT")

        response = AIMessage(content="This would be handled by the brain tumor agent, analyzing the MRI image.")

        # 【中文注释】影像类诊断结果必须人工审核，needs_human_validation 强制置 True
        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "BRAIN_TUMOR_AGENT"
        }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis."""
        # 【中文注释】执行节点：胸片 Agent。对上传胸片做新冠/正常二分类
        # 输入字段：current_input（含 image 路径）
        # 输出字段：output、needs_human_validation=True、agent_name

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: CHEST_XRAY_AGENT")

        # classify chest x-ray into covid or normal
        # 【中文注释】调用共享图片分析器对胸片分类（covid19 / normal / 其他）
        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)

        # 【中文注释】按分类结果生成对应文案
        if predicted_class == "covid19":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**.")
        elif predicted_class == "normal":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**.")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        # 【中文注释】影像诊断强制人工确认
        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT"
        }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""
        # 【中文注释】执行节点：皮肤病变 Agent。对皮肤图片做病变分割
        # 输入字段：current_input（含 image 路径）
        # 输出字段：output、needs_human_validation=True、agent_name

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: SKIN_LESION_AGENT")

        # classify chest x-ray into covid or normal
        # 【中文注释】调用共享图片分析器分割皮肤病变区域，返回分割掩码（有结果即为真）
        predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)

        # 【中文注释】分割成功与否决定回复文案
        if predicted_mask:
            response = AIMessage(content="Following is the analyzed **segmented** output of the uploaded skin lesion image:")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the skin lesion agent, analyzing the skin image.")

        # 【中文注释】影像诊断强制人工确认
        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT"
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        # 【中文注释】节点：人工审核检查。根据 needs_human_validation 决定是否进入人工确认，否则直接结束
        # 输入字段：needs_human_validation
        # 返回字段：agent_state、next（human_validation 或 END）、agent
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        # 【中文注释】节点：人工确认。在诊断结果后追加"请专业人士/患者确认"的提示
        # 输入字段：output、agent_name
        # 输出字段：output（追加确认提示）、agent_name（叠加 ", HUMAN_VALIDATION"）
        print(f"Selected agent: HUMAN_VALIDATION")

        # Append validation request to the existing output
        # 【中文注释】在原有诊断输出后追加人工审核请求文案
        validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

        # Create an AI message with the validation prompt
        # 【中文注释】包装成 AIMessage 作为新输出
        validation_message = AIMessage(content=validation_prompt)

        # 【中文注释】Agent 名叠加记录人工确认环节
        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        # 【中文注释】节点：出口安检。先处理人工确认应答（Yes/No），再对最终输出做合规清洗
        # 输入字段：output、current_input、messages
        # 输出字段：messages（把答案写回会话历史）、output
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        # 【中文注释】输出为空或类型不合法时原样返回，不做处理
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        # If the last message was a human validation message
        # 【中文注释】若输出中含人工审核请求，说明当前输入可能是用户对审核的应答
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            # 【中文注释】从当前输入中提取文本，检查是否为 Yes/No 应答
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                # 【中文注释】把审核结果作为人类消息写入会话历史
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                # 【中文注释】专业人士否决时：改输出为"需进一步复核"的兜底文案
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                # 【中文注释】审核通过：仅把确认结果写入历史
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        # 【中文注释】再次提取输入文本，供出口安检对照使用
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        # 【中文注释】出口安检：对输出做合规清洗（去敏感词、防注入等）
        sanitized_output = guardrails.check_output(output_text, input_text)
        # sanitized_output = output_text
        
        # For non-validation cases, add the sanitized output to messages
        # 【中文注释】统一包装为 AIMessage
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        # 【中文注释】把答案写回 messages 即进入会话历史：最终答复落库，供下一轮对话引用
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    
    # Create the workflow graph
    # 【中文注释】========== 组装状态机：注册节点、定义入口、连接边 ==========
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    # 【中文注释】把每个函数注册为图节点（节点名即条件边映射的键）
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    # 【中文注释】入口：每次请求都从安检+图片识别开始
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    # 【中文注释】条件边1：安检拦截则直通出口安检（apply_guardrails），否则进入路由决策（route_to_agent）
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    # 【中文注释】条件边2：路由决策节点的返回字典里取 "next" 字段作为目标节点名，映射到对应执行节点；
    # needs_validation（低信心）统一兜底到 RAG_AGENT
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    # 【中文注释】普通执行节点（对话/搜索/三个影像）完成后统一进入人工审核检查
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    # 【中文注释】条件边3：RAG 节点后按检索信心/信息充分性分流（联网搜索 或 人工审核检查）
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    # 【中文注释】人工确认后进入出口安检；出口安检后结束整个流程
    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    # 【中文注释】条件边4：check_validation 返回的 "next" 决定去人工确认（human_validation）还是出口安检（END 被映射到 apply_guardrails，保证任何路径都过安检）
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph
    # 【中文注释】编译图并绑定共享的 MemorySaver：每次 invoke 后状态自动落盘，实现多轮记忆
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    # 【中文注释】初始化状态包：11 个字段全部置默认值，供每次请求开局使用
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None) -> str:
    """
    Process a user query through the agent decision system.
    
    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages, NOT NEEDED ANYMORE since the state saves the conversation history now
        
    Returns:
        Response from the appropriate agent
    """
    # 【中文注释】统一 HTTP 入口：每次请求新建图实例（共享底层 MemorySaver），把用户输入注入状态后整体执行并返回最终状态
    # Initialize the graph
    # 【中文注释】重新创建图（节点闭包重建、开销小），但 checkpointer 始终是全局共享的 memory，历史会话因此得以延续
    graph = create_agent_graph()

    # # Save Graph Flowchart
    # image_bytes = graph.get_graph().draw_mermaid_png()
    # decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # cv2.imwrite("./assets/graph.png", decoded)
    # print("Graph flowchart saved in assets.")
    
    # Initialize state
    # 【中文注释】构建本次请求的初始状态（历史由 checkpointer 恢复，无需手工传入）
    state = init_agent_state()
    # if conversation_history:
    #     state["messages"] = conversation_history
    
    # Add the current query
    # 【中文注释】把用户原始输入（字符串或含图字典）放入 current_input，供各节点读取
    state["current_input"] = query

    # To handle image upload case
    # 【中文注释】图片类输入：把 text 加上"用户上传了图片"的说明，再作为人类消息入历史
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."
    
    state["messages"] = [HumanMessage(content=query)]

    # result = graph.invoke(state, thread_config)
    # 【中文注释】以 thread_id=1 执行整条流水线；checkpointer 自动合并该线程的历史状态
    result = graph.invoke(state, thread_config)
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Keep history to reasonable size (ANOTHER OPTION: summarize and store before truncating history)
    # 【中文注释】历史超限时截断：只保留最近 config.max_conversation_history 条消息，防止上下文无限膨胀
    if len(result["messages"]) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # visualize conversation history in console
    # 【中文注释】控制台打印本轮对话历史，便于调试
    for m in result["messages"]:
        m.pretty_print()
    
    # Add the response to conversation history
    # 【中文注释】返回完整状态（含更新后的 messages），供上层接口取最新回复
    return result
