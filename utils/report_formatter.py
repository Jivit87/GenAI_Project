
def format_report(state: dict) -> dict:
    """Takes the populated agent state and creates a clean report."""
    features = state.get("property_features", {})
    prefs = state.get("user_preferences", {})
    predicted_price = state.get("predicted_price", 0.0)
    price_range = state.get("price_range", {"low": 0.0, "high": 0.0})
    comps = state.get("comparable_properties", [])
    analysis = state.get("market_analysis", "Market analysis not available.")

    # 1. Format the comparable properties into a nice list
    comp_lines = []
    for c in comps:
        comp_lines.append(
            f"- **{c['id']}**: ${c['price']:,.2f} | {c['size_sqft']} sqft | Sold {c['sold_days_ago']} days ago"
        )
    comps_formatted = "\n".join(comp_lines) if comp_lines else "No comparables found."

    # 2. Determine recommendation based on budget
    budget = prefs.get("budget", predicted_price * 1.5)
    if predicted_price <= budget:
        recommendation = "BUY — Property is within your budget and shows positive potential."
        action = "Proceed with evaluating the physical property and legal documents."
    else:
        recommendation = "HOLD — Property exceeds your current budget."
        action = "Re-evaluate budget or try negotiating with the seller."

    # 3. Assemble the report dictionary
    # The Streamlit UI will display these sections in different tabs
    return {
        "summary": (
            f"**Property Size:** {features.get('total_flat_area', 'N/A')} sqft | "
            f"**Rooms:** {features.get('num_bedrooms', 'N/A')} Bed / {features.get('num_bathrooms', 'N/A')} Bath\n\n"
            f"**Predicted Value:** ${predicted_price:,.2f}\n"
            f"**Confidence Range:** ${price_range['low']:,.2f} - ${price_range['high']:,.2f}\n\n"
            f"### Market Analysis\n{analysis}"
        ),
        "comparables": comps_formatted,
        "recommendation": recommendation,
        "action": action,
        "disclaimer": ""  # Added later by the disclaimer node
    }
