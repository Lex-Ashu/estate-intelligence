from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (
    validate_input_node,
    predict_price_node,
    retrieve_trends_node,
    generate_report_node,
)


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("validate_input", validate_input_node)
    graph.add_node("predict_price", predict_price_node)
    graph.add_node("retrieve_trends", retrieve_trends_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "predict_price")
    graph.add_edge("predict_price", "retrieve_trends")
    graph.add_edge("retrieve_trends", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


advisory_graph = build_graph()
