from pangram import Pangram

# Initialize the client with your API key
client = Pangram(api_key="7974a222-e515-4d55-a27c-01d0d11fe1ca")

text_to_analyze = """I keep seeing educators treat low pass rates like some kind of flex, and honestly it feels weird. If a huge chunk of the class is failing, that doesn’t automatically mean the teacher is brilliant or the course is “challenging” in a good way. At some point you have to wonder whether the exams are just unfair, the material isn’t being taught well, or the expectations are off. Maybe there’s a line where a low pass rate stops looking impressive and starts looking like a sign that something in the class structure is seriously broken."""

# Call the predict method and request the dashboard link
result = client.predict(text_to_analyze, public_dashboard_link=True)

print("=== OVERALL ANALYSIS ===")
# Basic classification text
print(f"Headline: {result.get('headline')}")
print(f"Prediction Short: {result.get('prediction_short')}")
print(f"Prediction Long: {result.get('prediction')}\n")

print("=== TEXT COMPOSITION ===")
# Breakdown of how much of the text falls into each category
print(f"Fraction AI-Generated: {result.get('fraction_ai', 0) * 100:.1f}%")
print(f"Fraction AI-Assisted: {result.get('fraction_ai_assisted', 0) * 100:.1f}%")
print(f"Fraction Human: {result.get('fraction_human', 0) * 100:.1f}%\n")

print("=== SEGMENT COUNTS ===")
print(f"AI Segments: {result.get('num_ai_segments')}")
print(f"AI-Assisted Segments: {result.get('num_ai_assisted_segments')}")
print(f"Human Segments: {result.get('num_human_segments')}\n")

print("=== DASHBOARD LINK ===")
# This is the ONLY way to see the "Triads" and specific word highlights
print(f"View Visual Highlights: {result.get('dashboard_link')}\n")

print("=== DETAILED WINDOW BREAKDOWN ===")
# This is where you get the exact text chunks that triggered the scores
for i, window in enumerate(result.get('windows', [])):
    print(f"--- Segment {i + 1} ---")
    print(f"Label: {window.get('label')}")
    print(f"AI Score: {window.get('ai_assistance_score')}")
    print(f"Confidence: {window.get('confidence')}")
    print(f"Word Count: {window.get('word_count')} (Tokens: {window.get('token_length')})")
    print(f"Location: Characters {window.get('start_index')} to {window.get('end_index')}")
    
    # Print the actual text of this segment so you can see what was flagged
    print(f"Flagged Text: \"{window.get('text')}\"\n")