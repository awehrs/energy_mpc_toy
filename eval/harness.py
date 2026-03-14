import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from dataset.eval_dataset import EvalDataset
from eval.grader import Grader, GradeResult


logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    uuid: str
    question: str
    original_completeness: float
    original_conciseness: float
    new_completeness_score: int
    new_completeness_rating: str
    new_conciseness_score: int
    new_conciseness_rating: str
    tools_used_pct: float
    order_correct: bool
    reasoning_completeness: str
    reasoning_conciseness: str


class EvalHarness:

    def __init__(
        self,
        dataset: EvalDataset,
        grader: Grader,
        agent=None,  # None until agent.act() is implemented — runs in baseline mode
    ):
        self.dataset = dataset
        self.grader = grader
        self.agent = agent

    def run(
        self,
        output_dir: Path,
        max_examples: Optional[int] = None,
    ) -> List[EvalResult]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n = min(max_examples, len(self.dataset)) if max_examples else len(self.dataset)
        logger.info(
            f"Running eval on {n} examples (agent={'live' if self.agent else 'baseline'})..."
        )

        results = []

        for i in range(n):
            example = self.dataset[i]

            try:
                if self.agent is not None:
                    messages = self._run_inference(example)
                else:
                    messages = example["messages"]

                grade = self.grader.grade(
                    question=example["question"],
                    target_tools=example["target_tools"],
                    messages=messages,
                )

                results.append(
                    EvalResult(
                        uuid=example["uuid"],
                        question=example["question"],
                        original_completeness=example["original_completeness"],
                        original_conciseness=example["original_conciseness"],
                        new_completeness_score=grade.completeness_score,
                        new_completeness_rating=grade.completeness_rating,
                        new_conciseness_score=grade.conciseness_score,
                        new_conciseness_rating=grade.conciseness_rating,
                        tools_used_pct=grade.tools_used_pct,
                        order_correct=grade.order_correct,
                        reasoning_completeness=grade.reasoning_completeness,
                        reasoning_conciseness=grade.reasoning_conciseness,
                    )
                )

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"  {i+1}/{n} — avg completeness delta: {self._avg_delta(results):+.2f}"
                    )

            except Exception as e:
                logger.warning(f"  [{i}] uuid={example['uuid']} failed: {e}")
                continue

        self._save(results, output_dir)
        self._print_summary(results)
        return results

    def _run_inference(self, example: Dict) -> List[Dict]:
        raise NotImplementedError(
            "agent.act() not yet implemented — pass agent=None for baseline mode"
        )

    def _save(self, results: List[EvalResult], output_dir: Path) -> None:
        out_path = output_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"Saved {len(results)} results to {out_path}")

    def _avg_delta(self, results: List[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(
            r.new_completeness_score - r.original_completeness for r in results
        ) / len(results)

    def _print_summary(self, results: List[EvalResult]) -> None:
        if not results:
            logger.info("No results.")
            return

        n = len(results)
        orig_c = sum(r.original_completeness for r in results) / n
        new_c = sum(r.new_completeness_score for r in results) / n
        orig_cc = sum(r.original_conciseness for r in results) / n
        new_cc = sum(r.new_conciseness_score for r in results) / n
        tools_pct = sum(r.tools_used_pct for r in results) / n
        order_pct = sum(r.order_correct for r in results) / n

        logger.info(f"\n{'='*50}")
        logger.info(f"Eval summary ({n} examples)")
        logger.info(
            f"  Completeness:  {orig_c:.2f} → {new_c:.2f}  (Δ {new_c - orig_c:+.2f})"
        )
        logger.info(
            f"  Conciseness:   {orig_cc:.2f} → {new_cc:.2f}  (Δ {new_cc - orig_cc:+.2f})"
        )
        logger.info(f"  Tools used:    {tools_pct:.1%}")
        logger.info(f"  Order correct: {order_pct:.1%}")
        logger.info(f"{'='*50}")
