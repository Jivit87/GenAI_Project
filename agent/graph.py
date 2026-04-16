from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    intake_node,
    price_prediction_node,
    rag_retrieval_node,
    market_analysis_node,
    report_generation_node,
    disclaimer_node,
)


def build_graph():
 
    workflow = StateGraph(AgentState)

    workflow.add_node("intake",            intake_node)
    workflow.add_node("price_prediction",  price_prediction_node)
    workflow.add_node("rag_retrieval",     rag_retrieval_node)
    workflow.add_node("market_analysis",   market_analysis_node)
    workflow.add_node("report_generation", report_generation_node)
    workflow.add_node("disclaimer",        disclaimer_node)


    workflow.set_entry_point("intake")
    workflow.add_edge("intake",            "price_prediction")
    workflow.add_edge("price_prediction",  "rag_retrieval")
    workflow.add_edge("rag_retrieval",     "market_analysis")
    workflow.add_edge("market_analysis",   "report_generation")
    workflow.add_edge("report_generation", "disclaimer")
    workflow.add_edge("disclaimer",        END)

    return workflow.compile()
