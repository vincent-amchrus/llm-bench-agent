#!/usr/bin/env python3
"""
Async Time-to-First-Token (TTFT) Benchmark Tool
✅ Timer starts BEFORE create()
✅ TTFT triggered by: content OR tool_calls OR reasoning
✅ All config via argparse (no hardcoded values)
"""

import argparse
import asyncio
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional
from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  Installing tqdm...", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tqdm"])
    from tqdm import tqdm


async def measure_ttft_single(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str,
    enable_thinking: bool,
    tools: Optional[List[Dict]],
    client: AsyncOpenAI,
    request_id: int
) -> Dict:
    """Measure TTFT for one prompt. Timer starts BEFORE create()."""
    
    result = {
        "request_id": request_id,
        "prompt_length": len(prompt),
        "prompt_preview": prompt[:60] + "..." if len(prompt) > 60 else prompt,
        "ttft_seconds": None,
        "first_token_type": None,
        "error": None,
        "success": False
    }

    try:
        # ✅ CRITICAL: Start timer BEFORE API call
        start = time.perf_counter()
        
        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }
        }
        if tools:
            create_kwargs.update({"tools": tools, "tool_choice": "auto"})
        
        stream = await client.chat.completions.create(**create_kwargs)
        
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, 'content', None)
            tool_calls = getattr(delta, 'tool_calls', None)
            reasoning = getattr(delta, 'reasoning', None)
            
            # ✅ TTFT on first meaningful signal (content/tool_calls/reasoning)
            if content or tool_calls or reasoning:
                end = time.perf_counter()
                result.update({
                    "ttft_seconds": round(end - start, 4),
                    "success": True,
                    "first_token_type": (
                        "reasoning" if reasoning else 
                        "tool_calls" if tool_calls else 
                        "content"
                    )
                })
                break
        else:
            result["error"] = "No tokens/tool_calls/reasoning in stream"
            
    except Exception as e:
        result.update({"error": str(e), "ttft_seconds": float('inf')})
    
    return result


async def measure_ttft_batch(
    prompts: List[str],
    base_url: str,
    model: str,
    api_key: str,
    enable_thinking: bool,
    tools: Optional[List[Dict]],
    concurrency: int,
    show_progress: bool
) -> List[Dict]:
    """Batch measurement with concurrency control + progress bar."""
    
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded(prompt: str, idx: int) -> Dict:
        async with semaphore:
            return await measure_ttft_single(
                prompt=prompt, base_url=base_url, model=model,
                api_key=api_key, enable_thinking=enable_thinking,
                tools=tools, client=client, request_id=idx
            )
    
    try:
        tasks = [bounded(p, i) for i, p in enumerate(prompts)]
        results = []
        
        if show_progress:
            with tqdm(total=len(tasks), desc="⏱️ TTFT", unit="req") as pbar:
                for coro in asyncio.as_completed(tasks):
                    r = await coro
                    results.append(r)
                    succ = sum(1 for x in results if x["success"])
                    valid = [x["ttft_seconds"] for x in results if x["success"] and isinstance(x["ttft_seconds"], (int, float)) and x["ttft_seconds"] != float('inf')]
                    avg = sum(valid)/len(valid) if valid else 0
                    pbar.set_postfix_str(f"✓{succ} avg:{avg:.3f}s")
                    pbar.update(1)
        else:
            results = await asyncio.gather(*tasks)
        
        results.sort(key=lambda x: x["request_id"])  # Preserve input order
        return results
    finally:
        await client.close()


def load_prompts(path: str) -> List[str]:
    """Load [{'user_message': '...'}, ...] format."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x["user_message"].strip() for x in data if x.get("user_message")]
    elif isinstance(data, dict) and data.get("user_message"):
        return [data["user_message"].strip()]
    raise ValueError(f"Invalid format in {path}")


def load_tools(path: Optional[str]) -> Optional[List[Dict]]:
    """Load tools JSON if path provided and exists."""
    if not path or not Path(path).exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calc_stats(results: List[Dict]) -> Dict:
    """Compute TTFT statistics."""
    successful = [r for r in results if r["success"] and isinstance(r["ttft_seconds"], (int, float)) and r["ttft_seconds"] != float('inf')]
    if not successful:
        return {"total": len(results), "success": 0, "failed": len(results), "stats": None}
    
    vals = sorted(r["ttft_seconds"] for r in successful)
    n = len(vals)
    return {
        "total": len(results),
        "success": len(successful),
        "failed": len(results) - len(successful),
        "stats": {
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "mean": round(sum(vals)/n, 4), "median": round(vals[n//2], 4),
            "p90": round(vals[int(n*0.9)], 4) if n >= 10 else None,
            "p99": round(vals[int(n*0.99)], 4) if n >= 100 else None,
        },
        "token_types": {
            "content": sum(1 for r in successful if r.get("first_token_type")=="content"),
            "tool_calls": sum(1 for r in successful if r.get("first_token_type")=="tool_calls"),
            "reasoning": sum(1 for r in successful if r.get("first_token_type")=="reasoning"),
        }
    }


def parse_args():
    """Define argparse interface (all config passed from shell)."""
    p = argparse.ArgumentParser(description="TTFT Benchmark (argparse-driven)")
    
    # Required
    p.add_argument("--base-url", type=str, required=True, help="API base URL")
    p.add_argument("--model", type=str, required=True, help="Model name")
    p.add_argument("--input-file", type=str, required=True, help="Input JSON with prompts")
    
    # Optional with defaults
    p.add_argument("--api-key", type=str, default="EMPTY", help="API key")
    p.add_argument("--output-file", type=str, default="ttft_results.json", help="Output JSON path")
    p.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (s)")
    
    # Flags
    p.add_argument("--enable-thinking", action="store_true", help="Enable reasoning mode")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm")
    p.add_argument("--tools-file", type=str, default=None, help="Optional tools JSON path")
    
    return p.parse_args()


async def main():
    args = parse_args()
    
    # Validate
    if not Path(args.input_file).exists():
        print(f"❌ Not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    prompts = load_prompts(args.input_file)
    tools = load_tools(args.tools_file)
    
    if not prompts:
        print("❌ No valid prompts in input file", file=sys.stderr)
        sys.exit(1)
    
    show_progress = not args.no_progress
    print(f"🚀 Starting: {len(prompts)} prompts | {args.concurrency} CCU | {args.model}")
    
    start_total = time.perf_counter()
    
    try:
        results = await asyncio.wait_for(
            measure_ttft_batch(
                prompts=prompts,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                enable_thinking=args.enable_thinking,
                tools=tools,
                concurrency=args.concurrency,
                show_progress=show_progress
            ),
            timeout=args.timeout * max(1, len(prompts) // args.concurrency)
        )
    except asyncio.TimeoutError:
        print(f"\n❌ Timeout after {args.timeout}s", file=sys.stderr)
        sys.exit(2)
    
    total_time = time.perf_counter() - start_total
    stats = calc_stats(results)
    
    # Report
    print("\n" + "═"*60)
    print(f"🏁 Done: {stats['success']}/{stats['total']} ✓ | {total_time:.1f}s total")
    if stats["stats"]:
        s = stats["stats"]
        print(f"⏱️  TTFT: min={s['min']:.3f}s | mean={s['mean']:.3f}s | p90={s['p90'] or 'N/A'}")
    print("═"*60)
    
    # Save
    output = {
        "config": {k: v for k, v in vars(args).items() if k != 'api_key'},
        "summary": stats,
        "total_time_seconds": round(total_time, 2),
        "throughput_req_per_sec": round(stats["success"]/total_time, 2) if total_time > 0 else 0,
        "results": results
    }
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())