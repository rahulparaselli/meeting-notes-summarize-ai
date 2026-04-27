"""Quick end-to-end test: ingest a transcript + query it."""
import asyncio
import sys

sys.path.insert(0, ".")

from src.ingestion.pipeline import ingest_text
from src.agents.graph import run_pipeline


SAMPLE_TRANSCRIPT = """Alice: Good morning everyone. Let us start the Q4 planning meeting.
Bob: Thanks Alice. First, I want to discuss the marketing budget. We need to increase it by 20 percent.
Carol: I agree with Bob. The social media campaigns performed well last quarter. I think we should allocate more to Instagram and TikTok.
Alice: Alright, so the decision is to increase the marketing budget by 20 percent with focus on Instagram and TikTok.
Bob: I will prepare the budget proposal by next Friday and send it to the CFO.
Carol: And I will draft the social media strategy document by Wednesday.
Alice: Great. Next topic - the product launch timeline.
Bob: The engineering team says they need two more weeks. So we are looking at a November 15th launch date.
Carol: That works for me. I will update the marketing timeline accordingly.
Alice: Perfect. Let us also discuss the new hire for the data team. I think we should post the job listing this week.
Bob: Agreed. I will talk to HR about the job description today.
Alice: Thank you everyone. Meeting adjourned."""


async def test():
    # 1. Ingest
    print("--- INGESTING ---")
    meta = await ingest_text(
        text=SAMPLE_TRANSCRIPT,
        title="Q4 Planning Meeting",
        attendees=["Alice", "Bob", "Carol"],
    )
    print(f"Meeting ID: {meta.id}")
    print(f"Title: {meta.title}")

    # 2. Test summary
    print("\n--- QUERY: Summarise this meeting ---")
    result = await run_pipeline(meta.id, "Summarise this meeting")
    qt = result.get("query_type")
    qt_str = qt.value if hasattr(qt, "value") else str(qt)
    print(f"Query type: {qt_str}")
    print(f"Steps: {result.get('steps', [])}")
    output = result.get("output", {})
    print(f"Output keys: {list(output.keys())}")
    tldr = output.get("tldr", "N/A")
    print(f"TL;DR: {tldr[:200]}")
    kp = output.get("key_points", [])
    print(f"Key points: {len(kp)}")
    for p in kp[:3]:
        print(f"  - {p[:100]}")

    # 3. Test action items
    print("\n--- QUERY: What are the action items? ---")
    result2 = await run_pipeline(meta.id, "What are the action items?")
    qt2 = result2.get("query_type")
    qt2_str = qt2.value if hasattr(qt2, "value") else str(qt2)
    print(f"Query type: {qt2_str}")
    items = result2.get("output", {}).get("action_items", [])
    print(f"Action items found: {len(items)}")
    for item in items:
        print(f"  - {item.get('task', '?')} -> {item.get('owner', '?')}")

    # 4. Test thinking/trace
    print(f"\nTrace steps: {len(result.get('thinking', []))}")
    for t in result.get("thinking", []):
        print(f"  {t.get('step', '?')}: {t.get('detail', '')[:80]}")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    asyncio.run(test())
