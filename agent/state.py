from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    property_features: Dict[str, Any]   
    user_preferences: Dict[str, Any]    

    predicted_price: Optional[float]
    price_range: Optional[Dict[str, float]]      
    retrieved_market_docs: Optional[List[str]]
    market_analysis: Optional[str]
    comparable_properties: Optional[List[Dict]]

    advisory_report: Optional[Dict[str, str]]
    error: Optional[str]
