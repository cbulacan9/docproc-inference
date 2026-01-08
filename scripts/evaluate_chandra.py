#!/usr/bin/env python3
"""
Chandra vs dots.ocr Evaluation Script

Compares extraction accuracy and confidence calibration between
dots.ocr and Chandra on the same set of test documents.

Usage:
    python scripts/evaluate_chandra.py --config eval_config.yaml
    python scripts/evaluate_chandra.py --image-url http://localhost:8080/test.png --doc-type bank_statement
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Result from a single extraction."""
    model: str
    doc_type: str
    image_url: str
    success: bool
    data: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    latency_ms: int = 0
    error: str | None = None


@dataclass
class ComparisonResult:
    """Comparison between two models on same document."""
    image_url: str
    doc_type: str
    dots_ocr: EvaluationResult
    chandra: EvaluationResult
    field_comparison: dict = field(default_factory=dict)
    ground_truth: dict | None = None


class ChandraEvaluator:
    """Evaluate Chandra vs dots.ocr extraction."""

    def __init__(
        self,
        gateway_url: str,
        api_key: str,
        timeout: int = 120
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            headers={"X-API-Key": api_key},
            timeout=httpx.Timeout(timeout)
        )

    async def close(self):
        await self.client.aclose()

    async def extract_dots_ocr(
        self,
        image_urls: list[str],
        doc_type: str
    ) -> EvaluationResult:
        """Call dots.ocr extraction endpoint."""
        return await self._extract(
            endpoint="/v1/extract",
            model="dots-ocr",
            image_urls=image_urls,
            doc_type=doc_type
        )

    async def extract_chandra(
        self,
        image_urls: list[str],
        doc_type: str
    ) -> EvaluationResult:
        """Call Chandra extraction endpoint."""
        return await self._extract(
            endpoint="/v1/extract-chandra",
            model="chandra",
            image_urls=image_urls,
            doc_type=doc_type
        )

    async def _extract(
        self,
        endpoint: str,
        model: str,
        image_urls: list[str],
        doc_type: str
    ) -> EvaluationResult:
        """Make extraction request."""
        try:
            response = await self.client.post(
                f"{self.gateway_url}{endpoint}",
                json={
                    "image_urls": image_urls,
                    "doc_type": doc_type
                }
            )

            if response.status_code != 200:
                return EvaluationResult(
                    model=model,
                    doc_type=doc_type,
                    image_url=image_urls[0],
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )

            data = response.json()
            return EvaluationResult(
                model=model,
                doc_type=doc_type,
                image_url=image_urls[0],
                success=True,
                data=data.get("data", {}),
                confidence=data.get("confidence", {}),
                latency_ms=data.get("latency_ms", 0)
            )

        except Exception as e:
            return EvaluationResult(
                model=model,
                doc_type=doc_type,
                image_url=image_urls[0],
                success=False,
                error=str(e)
            )

    async def compare(
        self,
        image_urls: list[str],
        doc_type: str,
        ground_truth: dict | None = None
    ) -> ComparisonResult:
        """
        Run both models on same document and compare results.
        """
        # Run both extractions in parallel
        dots_ocr_task = self.extract_dots_ocr(image_urls, doc_type)
        chandra_task = self.extract_chandra(image_urls, doc_type)

        dots_ocr_result, chandra_result = await asyncio.gather(
            dots_ocr_task, chandra_task
        )

        # Compare fields
        field_comparison = self._compare_fields(
            dots_ocr_result.data,
            chandra_result.data,
            dots_ocr_result.confidence.get("fields", {}),
            chandra_result.confidence.get("fields", {}),
            ground_truth
        )

        return ComparisonResult(
            image_url=image_urls[0],
            doc_type=doc_type,
            dots_ocr=dots_ocr_result,
            chandra=chandra_result,
            field_comparison=field_comparison,
            ground_truth=ground_truth
        )

    def _compare_fields(
        self,
        dots_data: dict,
        chandra_data: dict,
        dots_conf: dict,
        chandra_conf: dict,
        ground_truth: dict | None
    ) -> dict:
        """Compare extracted fields between models."""
        comparison = {
            "matching_fields": [],
            "differing_fields": [],
            "dots_only": [],
            "chandra_only": [],
            "accuracy": {}
        }

        # Flatten nested dicts for comparison
        dots_flat = self._flatten_dict(dots_data)
        chandra_flat = self._flatten_dict(chandra_data)
        truth_flat = self._flatten_dict(ground_truth) if ground_truth else {}

        all_keys = set(dots_flat.keys()) | set(chandra_flat.keys())

        for key in sorted(all_keys):
            dots_val = dots_flat.get(key)
            chandra_val = chandra_flat.get(key)
            truth_val = truth_flat.get(key)

            if dots_val == chandra_val:
                comparison["matching_fields"].append({
                    "field": key,
                    "value": dots_val,
                    "dots_conf": dots_conf.get(key, "N/A"),
                    "chandra_conf": chandra_conf.get(key, "N/A")
                })
            else:
                comparison["differing_fields"].append({
                    "field": key,
                    "dots_value": dots_val,
                    "chandra_value": chandra_val,
                    "ground_truth": truth_val,
                    "dots_conf": dots_conf.get(key, "N/A"),
                    "chandra_conf": chandra_conf.get(key, "N/A")
                })

            # Calculate accuracy if ground truth available
            if truth_val is not None:
                dots_correct = self._values_match(dots_val, truth_val)
                chandra_correct = self._values_match(chandra_val, truth_val)
                comparison["accuracy"][key] = {
                    "dots_ocr_correct": dots_correct,
                    "chandra_correct": chandra_correct
                }

        return comparison

    def _flatten_dict(self, d: dict, parent_key: str = "") -> dict:
        """Flatten nested dict to dot-notation keys."""
        items = {}
        if not d:
            return items

        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(self._flatten_dict(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = v

        return items

    def _values_match(self, extracted: Any, truth: Any) -> bool:
        """Check if extracted value matches ground truth."""
        if extracted is None and truth is None:
            return True
        if extracted is None or truth is None:
            return False

        # Numeric comparison with tolerance
        if isinstance(truth, (int, float)) and isinstance(extracted, (int, float)):
            return abs(float(extracted) - float(truth)) < 0.01

        # String comparison (case-insensitive, whitespace-normalized)
        return str(extracted).strip().lower() == str(truth).strip().lower()


def print_comparison_report(result: ComparisonResult):
    """Print human-readable comparison report."""
    print("\n" + "=" * 80)
    print(f"EVALUATION REPORT: {result.doc_type}")
    print(f"Image: {result.image_url}")
    print("=" * 80)

    # Model results summary
    print("\n## Model Results\n")
    print(f"{'Model':<15} {'Success':<10} {'Latency':<12} {'Overall Conf':<15}")
    print("-" * 52)

    for model, res in [("dots-ocr", result.dots_ocr), ("chandra", result.chandra)]:
        success = "Yes" if res.success else "No"
        latency = f"{res.latency_ms}ms" if res.success else "N/A"
        conf = f"{res.confidence.get('overall', 0):.2f}" if res.success else "N/A"
        print(f"{model:<15} {success:<10} {latency:<12} {conf:<15}")

    if not result.dots_ocr.success or not result.chandra.success:
        print("\n## Errors\n")
        if not result.dots_ocr.success:
            print(f"dots-ocr: {result.dots_ocr.error}")
        if not result.chandra.success:
            print(f"chandra: {result.chandra.error}")
        return

    # Field comparison
    comp = result.field_comparison

    print(f"\n## Field Comparison\n")
    print(f"Matching fields: {len(comp['matching_fields'])}")
    print(f"Differing fields: {len(comp['differing_fields'])}")

    if comp["differing_fields"]:
        print("\n### Differences\n")
        print(f"{'Field':<35} {'dots-ocr':<20} {'chandra':<20} {'Truth':<15}")
        print("-" * 90)
        for diff in comp["differing_fields"]:
            field = diff["field"][:34]
            dots = str(diff["dots_value"])[:19]
            chandra = str(diff["chandra_value"])[:19]
            truth = str(diff.get("ground_truth", ""))[:14]
            print(f"{field:<35} {dots:<20} {chandra:<20} {truth:<15}")

    # Accuracy summary (if ground truth provided)
    if comp["accuracy"]:
        print("\n## Accuracy (vs Ground Truth)\n")
        dots_correct = sum(1 for v in comp["accuracy"].values() if v["dots_ocr_correct"])
        chandra_correct = sum(1 for v in comp["accuracy"].values() if v["chandra_correct"])
        total = len(comp["accuracy"])

        print(f"dots-ocr: {dots_correct}/{total} fields correct ({dots_correct/total*100:.1f}%)")
        print(f"chandra:  {chandra_correct}/{total} fields correct ({chandra_correct/total*100:.1f}%)")

    print("\n" + "=" * 80)


def print_batch_summary(results: list[ComparisonResult]):
    """Print summary of batch evaluation."""
    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 80)

    total = len(results)
    dots_success = sum(1 for r in results if r.dots_ocr.success)
    chandra_success = sum(1 for r in results if r.chandra.success)

    print(f"\nTotal test cases: {total}")
    print(f"dots-ocr success rate: {dots_success}/{total}")
    print(f"chandra success rate: {chandra_success}/{total}")

    # Average latency
    dots_latencies = [r.dots_ocr.latency_ms for r in results if r.dots_ocr.success]
    chandra_latencies = [r.chandra.latency_ms for r in results if r.chandra.success]

    if dots_latencies:
        print(f"\ndots-ocr avg latency: {sum(dots_latencies)/len(dots_latencies):.0f}ms")
    if chandra_latencies:
        print(f"chandra avg latency: {sum(chandra_latencies)/len(chandra_latencies):.0f}ms")

    # Average confidence
    dots_confs = [r.dots_ocr.confidence.get("overall", 0) for r in results if r.dots_ocr.success]
    chandra_confs = [r.chandra.confidence.get("overall", 0) for r in results if r.chandra.success]

    if dots_confs:
        print(f"\ndots-ocr avg confidence: {sum(dots_confs)/len(dots_confs):.2f}")
    if chandra_confs:
        print(f"chandra avg confidence: {sum(chandra_confs)/len(chandra_confs):.2f}")


def save_results(results: list[ComparisonResult], output_path: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": []
    }

    for r in results:
        output["results"].append({
            "image_url": r.image_url,
            "doc_type": r.doc_type,
            "dots_ocr": {
                "success": r.dots_ocr.success,
                "data": r.dots_ocr.data,
                "confidence": r.dots_ocr.confidence,
                "latency_ms": r.dots_ocr.latency_ms,
                "error": r.dots_ocr.error
            },
            "chandra": {
                "success": r.chandra.success,
                "data": r.chandra.data,
                "confidence": r.chandra.confidence,
                "latency_ms": r.chandra.latency_ms,
                "error": r.chandra.error
            },
            "field_comparison": r.field_comparison,
            "ground_truth": r.ground_truth
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


async def run_single_evaluation(
    gateway_url: str,
    api_key: str,
    image_url: str,
    doc_type: str,
    ground_truth: dict | None = None
):
    """Run evaluation on a single image."""
    evaluator = ChandraEvaluator(gateway_url, api_key)

    try:
        result = await evaluator.compare(
            image_urls=[image_url],
            doc_type=doc_type,
            ground_truth=ground_truth
        )
        print_comparison_report(result)
        return result

    finally:
        await evaluator.close()


async def run_batch_evaluation(
    gateway_url: str,
    api_key: str,
    config_path: str
):
    """Run evaluation on batch of images from config file."""
    if not YAML_AVAILABLE:
        print("Error: PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    evaluator = ChandraEvaluator(gateway_url, api_key)
    results = []

    try:
        for test_case in config.get("test_cases", []):
            print(f"\nEvaluating: {test_case['name']}...")

            result = await evaluator.compare(
                image_urls=test_case["image_urls"],
                doc_type=test_case["doc_type"],
                ground_truth=test_case.get("ground_truth")
            )
            results.append(result)
            print_comparison_report(result)

        # Summary
        print_batch_summary(results)

        # Save results to JSON
        output_path = config.get("output_path", "evaluation_results.json")
        save_results(results, output_path)

    finally:
        await evaluator.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Chandra vs dots.ocr extraction"
    )
    parser.add_argument(
        "--gateway-url",
        default=os.environ.get("GATEWAY_URL", "http://localhost:8000"),
        help="Gateway URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GATEWAY_API_KEY"),
        help="Gateway API key (or set GATEWAY_API_KEY env var)"
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config file for batch evaluation"
    )
    parser.add_argument(
        "--image-url",
        help="Single image URL for quick test"
    )
    parser.add_argument(
        "--doc-type",
        default="bank_statement",
        help="Document type (default: bank_statement)"
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: API key required. Set GATEWAY_API_KEY or use --api-key")
        sys.exit(1)

    if args.config:
        asyncio.run(run_batch_evaluation(
            args.gateway_url,
            args.api_key,
            args.config
        ))
    elif args.image_url:
        asyncio.run(run_single_evaluation(
            args.gateway_url,
            args.api_key,
            args.image_url,
            args.doc_type
        ))
    else:
        print("Error: Provide either --config for batch or --image-url for single test")
        print("\nExamples:")
        print("  python evaluate_chandra.py --image-url http://localhost:8080/doc.png --doc-type bank_statement")
        print("  python evaluate_chandra.py --config eval_config.yaml")
        sys.exit(1)


if __name__ == "__main__":
    main()
