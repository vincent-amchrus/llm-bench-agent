#!/usr/bin/env python3
"""
Async Time-to-First-Token (TTFT) Benchmark Tool
Measures TTFT for multiple prompts concurrently using streaming OpenAI API.
"""

import argparse
import asyncio
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
from openai import AsyncOpenAI


async def measure_ttft_single(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    enable_thinking: bool = False,
    client: Optional[AsyncOpenAI] = None,
    request_id: Optional[int] = None
) -> Dict:
    """
    Measure Time-to-First-Token for a single prompt using streaming.
    
    Returns dict with ttft, success status, and metadata.
    """
    # Create client per request if not provided (for connection isolation)
    if client is None:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        close_client = True
    else:
        close_client = False

    result = {
        "request_id": request_id,
        "prompt_length": len(prompt),
        "ttft_seconds": None,
        "error": None,
        "success": False
    }

    try:
        start = time.perf_counter()
        
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking
                }
            },
            stream=True
        )
        
        # Iterate until first content token arrives
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                end = time.perf_counter()
                result["ttft_seconds"] = round(end - start, 4)
                result["success"] = True
                break
        else:
            # No tokens generated
            result["error"] = "No tokens generated in stream"
            
    except Exception as e:
        result["error"] = str(e)
        result["ttft_seconds"] = float('inf')
        
    finally:
        if close_client:
            await client.close()
    
    return result


async def measure_ttft_batch(
    prompts: List[str],
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    enable_thinking: bool = False,
    concurrency: int = 1
) -> List[Dict]:
    """
    Measure TTFT for a batch of prompts with configurable concurrency.
    """
    # Create a shared client for better connection pooling
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_measure(prompt: str, idx: int) -> Dict:
        async with semaphore:
            return await measure_ttft_single(
                prompt=prompt,
                base_url=base_url,
                model=model,
                api_key=api_key,
                enable_thinking=enable_thinking,
                client=client,  # Reuse client
                request_id=idx
            )
    
    try:
        tasks = [
            bounded_measure(prompt, idx) 
            for idx, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        return results
    finally:
        await client.close()


def load_prompts_from_json(file_path: str) -> List[str]:
    """Load prompts from JSON file with format: [{'user_message': '...'}, ...]"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [item.get('user_message', '') for item in data if 'user_message' in item]
    elif isinstance(data, dict) and 'user_message' in data:
        return [data['user_message']]
    else:
        raise ValueError(f"Unexpected JSON format in {file_path}. Expected list of {{'user_message': ...}}")


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate TTFT statistics from results."""
    successful = [r for r in results if r["success"] and r["ttft_seconds"] not in [None, float('inf')]]
    
    if not successful:
        return {
            "total_requests": len(results),
            "successful_requests": 0,
            "failed_requests": len(results),
            "ttft_stats": None
        }
    
    ttft_values = [r["ttft_seconds"] for r in successful]
    ttft_values_sorted = sorted(ttft_values)
    
    return {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "ttft_stats": {
            "min": round(min(ttft_values), 4),
            "max": round(max(ttft_values), 4),
            "mean": round(sum(ttft_values) / len(ttft_values), 4),
            "median": round(ttft_values_sorted[len(ttft_values_sorted) // 2], 4),
            "p90": round(ttft_values_sorted[int(len(ttft_values_sorted) * 0.9)], 4) if len(ttft_values_sorted) >= 10 else None,
            "p99": round(ttft_values_sorted[int(len(ttft_values_sorted) * 0.99)], 4) if len(ttft_values_sorted) >= 100 else None
        }
    }


async def main():
    parser = argparse.ArgumentParser(description="Async TTFT Benchmark Tool")
    parser.add_argument("--base-url", type=str, required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key (default: EMPTY)")
    parser.add_argument("--model", type=str, required=True, help="Model name to test")
    parser.add_argument("--input-file", type=str, required=True, help="JSON file with prompts [{'user_message': '...'}]")
    parser.add_argument("--output-file", type=str, default="ttft_results.json", help="Output JSON file path")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests (default: 1)")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode in chat template")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Load prompts
    print(f"Loading prompts from {args.input_file}...")
    prompts = load_prompts_from_json(args.input_file)
    print(f"Loaded {len(prompts)} prompts")
    
    if not prompts:
        print("Error: No valid prompts found in input file", file=sys.stderr)
        sys.exit(1)
    
    # Run benchmark
    print(f"Starting TTFT benchmark: {args.concurrency} concurrent requests")
    print(f"Model: {args.model}, Base URL: {args.base_url}")
    
    start_total = time.perf_counter()
    
    results = await asyncio.wait_for(
        measure_ttft_batch(
            prompts=prompts,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            enable_thinking=args.enable_thinking,
            concurrency=args.concurrency
        ),
        timeout=args.timeout * len(prompts)  # Rough total timeout
    )
    
    total_time = time.perf_counter() - start_total
    
    # Calculate and display statistics
    stats = calculate_statistics(results)
    
    print("\n" + "="*60)
    print("TTFT Benchmark Results")
    print("="*60)
    print(f"Total requests:     {stats['total_requests']}")
    print(f"Successful:         {stats['successful_requests']}")
    print(f"Failed:             {stats['failed_requests']}")
    
    if stats["ttft_stats"]:
        ttft = stats["ttft_stats"]
        print(f"\nTTFT Statistics (seconds):")
        print(f"  Min:    {ttft['min']}")
        print(f"  Max:    {ttft['max']}")
        print(f"  Mean:   {ttft['mean']}")
        print(f"  Median: {ttft['median']}")
        if ttft['p90']:
            print(f"  P90:    {ttft['p90']}")
        if ttft['p99']:
            print(f"  P99:    {ttft['p99']}")
    
    print(f"\nTotal benchmark time: {round(total_time, 2)}s")
    print("="*60)
    
    # Save detailed results
    output_data = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "enable_thinking": args.enable_thinking,
            "input_file": args.input_file
        },
        "summary": stats,
        "total_time_seconds": round(total_time, 2),
        "results": results
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())