from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

try:
    from genai.rag_pipeline import ASLRagRetriever
    from genai.llm_engine import HuggingFaceGlossLLM
except ImportError:
    from rag_pipeline import ASLRagRetriever
    from llm_engine import HuggingFaceGlossLLM


class ASLAgentState(TypedDict):
    raw_gloss: str
    cleaned_gloss: str
    rag_context: List[Dict[str, Any]]
    translation_result: Dict[str, Any]
    final_response: Dict[str, Any]


def gloss_cleaner_node(state: ASLAgentState) -> Dict[str, Any]:
    raw = state.get("raw_gloss", "")
    cleaned = raw.strip().upper()
    return {"cleaned_gloss": cleaned}


def rag_retriever_node(state: ASLAgentState) -> Dict[str, Any]:
    cleaned_gloss = state.get("cleaned_gloss", "")
    retriever = ASLRagRetriever()
    context = retriever.retrieve(query=cleaned_gloss)
    return {"rag_context": context}


def llm_translator_node(state: ASLAgentState) -> Dict[str, Any]:
    cleaned_gloss = state.get("cleaned_gloss", "")
    rag_context = state.get("rag_context", [])
    llm = HuggingFaceGlossLLM()
    result = llm.process(gloss_sequence=cleaned_gloss, rag_context=rag_context)
    return {"translation_result": result}


def response_synthesizer_node(state: ASLAgentState) -> Dict[str, Any]:
    translation = state.get("translation_result", {})
    final_output = {
        "status": "success",
        "original_gloss": state.get("raw_gloss", ""),
        "cleaned_gloss": state.get("cleaned_gloss", ""),
        "translation": translation.get("translated_text", ""),
        "intent": translation.get("intent", ""),
        "suggested_replies": translation.get("suggested_replies", []),
        "rag_context": state.get("rag_context", [])
    }
    return {"final_response": final_output}


class LangGraphASLAgent:
    """
    LangGraph Agent for ASL recognition and translation using StateGraph with four nodes:
    gloss_cleaner_node, rag_retriever_node, llm_translator_node, and response_synthesizer_node.
    """
    def __init__(self):
        workflow = StateGraph(ASLAgentState)
        
        workflow.add_node("gloss_cleaner", gloss_cleaner_node)
        workflow.add_node("rag_retriever", rag_retriever_node)
        workflow.add_node("llm_translator", llm_translator_node)
        workflow.add_node("response_synthesizer", response_synthesizer_node)
        
        workflow.add_edge(START, "gloss_cleaner")
        workflow.add_edge("gloss_cleaner", "rag_retriever")
        workflow.add_edge("rag_retriever", "llm_translator")
        workflow.add_edge("llm_translator", "response_synthesizer")
        workflow.add_edge("response_synthesizer", END)
        
        self.app = workflow.compile()

    def run(self, gloss_input: str) -> Dict[str, Any]:
        initial_state = {
            "raw_gloss": gloss_input,
            "cleaned_gloss": "",
            "rag_context": [],
            "translation_result": {},
            "final_response": {}
        }
        result_state = self.app.invoke(initial_state)
        return result_state.get("final_response", {})


if __name__ == "__main__":
    agent = LangGraphASLAgent()
    response = agent.run("hello need help")
    print(response)
