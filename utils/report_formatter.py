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
        # Distance score corresponds to the Euclidean distance from target lat/long
        comp_lines.append(
            f"- **House ID: {c['id']}**: ${c['price']:,.2f} | {int(c['size_sqft'])} sqft | {int(c['bedrooms'])} BR | *Rel. Distance: {c['distance_score']:.4f}*"
        )
    comps_formatted = "\n".join(comp_lines) if comp_lines else "No real comparables found in historical dataset."

    # 2. Extract Recommendation from AI Analysis if possible
    # (The AI writer now provides the definitive verdict based on User Goal)
    recommendation_header = "AI Investment Verdict"
    
    # 3. Assemble the report dictionary
    return {
        "summary": (
            f"### 📈 Valuation Summary\n"
            f"**Property Details:** {int(features.get('no_of_bedrooms', 0))} Bed, {int(features.get('no_of_bathrooms', 0))} Bath, {int(features.get('total_flat_area', 0))} sqft\n\n"
            f"**Predicted Market Value:** `${predicted_price:,.2f}`\n"
            f"**90% Confidence Range:** `${price_range['low']:,.2f}` - `${price_range['high']:,.2f}`\n"
        ),
        "analysis": analysis,
        "comparables": comps_formatted,
        "disclaimer": ""  # Added later by the disclaimer node
    }
