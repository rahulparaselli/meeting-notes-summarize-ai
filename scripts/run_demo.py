#!/usr/bin/env python3
"""
Usage:
  python scripts/run_demo.py --text "path/to/transcript.txt" --title "Sprint Retro"
  python scripts/run_demo.py --meeting-id abc123 --query "What were the action items?"
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import run_pipeline
from src.ingestion.pipeline import ingest_text


async def cmd_ingest(args: argparse.Namespace) -> None:
    text = Path(args.text).read_text()
    meta = await ingest_text(
        text=text,
        title=args.title,
        attendees=args.attendees.split(",") if args.attendees else [],
    )
    print(f"\n✅ Ingested: {meta.title}")
    print(f"   Meeting ID: {meta.id}")
    print(f"   Use this ID for queries: --meeting-id {meta.id}\n")


async def cmd_query(args: argparse.Namespace) -> None:
    print(f"\n🔍 Query: {args.query}")
    print("─" * 50)

    result = await run_pipeline(meeting_id=args.meeting_id, query=args.query)

    print("\n🧠 Agent thinking trace:")
    for step in result.get("thinking", []):
        print(f"  [{step['step']}] {step['detail']}")

    print(f"\n📋 Query type detected: {result.get('query_type', '?')}")
    print("\n📝 Output:")
    print(json.dumps(result.get("output", {}), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting Summariser CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest a transcript file")
    p_ingest.add_argument("--text", required=True, help="Path to .txt transcript")
    p_ingest.add_argument("--title", required=True, help="Meeting title")
    p_ingest.add_argument("--attendees", default="", help="Comma-separated attendees")

    p_query = sub.add_parser("query", help="Query a meeting")
    p_query.add_argument("--meeting-id", required=True)
    p_query.add_argument("--query", required=True)

    args = parser.parse_args()

    if args.cmd == "ingest":
        asyncio.run(cmd_ingest(args))
    elif args.cmd == "query":
        asyncio.run(cmd_query(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
