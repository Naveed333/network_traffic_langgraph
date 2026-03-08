"""
main.py — Entry point for the LLM Network Traffic Prediction pipeline.

Builds and compiles the LangGraph StateGraph, initialises the pipeline
state, invokes the compiled graph, and prints a structured result summary.

Usage
-----
# Use synthetic data (no CSV needed):
    python main.py

# Use a real CSV:
    python main.py --data-path data/traffic_data.csv --target-date 2024-01-15

# Override hyperparameters:
    python main.py --max-iterations 3 --convergence-threshold 1.0

# Save results to JSON:
    python main.py --output results/run_001.json

Graph topology
--------------
build_initial_prompt
    └─► initial_predict   (API call 1)
            └─► mae_feedback    (API call 2, 5, 8, …)
                    └─► sine_feedback   (API call 3, 6, 9, …)
                            └─► assemble_feedback  (no LLM)
                                    └─► refine_predict  (API call 4, 7, 10, …)
                                            └─► check_convergence
                                                    ├─► [converged] ─► END
                                                    └─► [continue]  ─► mae_feedback
"""

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from typing import List, Optional

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config import settings
from data.loader import (
    generate_synthetic_data,
    load_deployment_input,
    load_traffic_data,
    validate_context_dates,
)
from evaluation.mae import compute_mae, compute_mape, compute_rmse
from nodes.assemble_feedback import assemble_feedback
from nodes.build_prompt import build_initial_prompt
from nodes.check_convergence import check_convergence, convergence_router
from nodes.initial_predict import api_call_initial_predict
from nodes.mae_feedback import api_call_mae_feedback
from nodes.refine_predict import api_call_refine
from nodes.sine_feedback import api_call_sine_feedback
from prompts.initial_prompt_template import build_p_exam_from_contexts
from state import TrafficPredictionState
from utils.logger import setup_logger
from utils.tracing import setup_tracing

logger = setup_logger(__name__, logging.INFO)

# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(enable_checkpointing: bool = True) -> "CompiledGraph":  # noqa: F821
    """
    Construct and compile the LangGraph StateGraph for the prediction pipeline.

    Args:
        enable_checkpointing: When True, attaches an in-memory MemorySaver
                              so individual runs can be resumed by thread_id.

    Returns:
        A compiled LangGraph application ready for ``.invoke()``.
    """
    workflow = StateGraph(TrafficPredictionState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("build_initial_prompt", build_initial_prompt)
    workflow.add_node("initial_predict",      api_call_initial_predict)
    workflow.add_node("mae_feedback",         api_call_mae_feedback)
    workflow.add_node("sine_feedback",        api_call_sine_feedback)
    workflow.add_node("assemble_feedback",    assemble_feedback)
    workflow.add_node("refine_predict",       api_call_refine)
    workflow.add_node("check_convergence",    check_convergence)

    # ── Define edges ──────────────────────────────────────────────────────────
    workflow.set_entry_point("build_initial_prompt")
    workflow.add_edge("build_initial_prompt", "initial_predict")
    workflow.add_edge("initial_predict",      "mae_feedback")
    workflow.add_edge("mae_feedback",         "sine_feedback")
    workflow.add_edge("sine_feedback",        "assemble_feedback")
    workflow.add_edge("assemble_feedback",    "refine_predict")
    workflow.add_edge("refine_predict",       "check_convergence")

    # ── Conditional loop edge ─────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "check_convergence",
        convergence_router,          # returns "converged" or "continue"
        {
            "continue":   "mae_feedback",   # loop back for next iteration
            "converged":  END,              # pipeline done
        },
    )

    # ── Compile (with optional checkpointing for resumability) ───────────────
    if enable_checkpointing:
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        logger.info("Graph compiled WITH MemorySaver checkpointing enabled")
    else:
        app = workflow.compile()
        logger.info("Graph compiled WITHOUT checkpointing")

    return app


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run_pipeline(
    data_path:            Optional[str]   = None,
    target_date:          Optional[str]   = None,
    max_iterations:       Optional[int]   = None,
    convergence_threshold: Optional[float] = None,
    thread_id:            str             = "default",
) -> dict:
    """
    Execute the complete LLM traffic prediction pipeline.

    Args:
        data_path:             Path to the hourly traffic CSV (or None for synthetic).
        target_date:           ISO date string to predict (or None → last date in CSV).
        max_iterations:        Override settings.max_iterations.
        convergence_threshold: Override settings.convergence_threshold.
        thread_id:             LangGraph thread ID for checkpointing / resumability.

    Returns:
        Result dict containing final prediction, MAE history, and telemetry.
    """
    setup_tracing()                          # enable LangSmith if API key is set
    logger.info("=" * 70)
    logger.info("LLM Network Traffic Prediction — Iterative Prompt Refinement")
    logger.info("Model: %-20s  Max iterations: %d  Threshold: %.4f",
                settings.model_name,
                max_iterations or settings.max_iterations,
                convergence_threshold or settings.convergence_threshold)
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    if data_path and os.path.exists(data_path):
        x_t, ground_truth, resolved_date = load_traffic_data(data_path, target_date)
        print(f"Loaded data from {data_path} | target_date={resolved_date},x_t={x_t},ground_truth={ground_truth}")

    else:
        if data_path:
            logger.warning(
                "Data file not found at %r — falling back to synthetic data", data_path
            )
        x_t, ground_truth, resolved_date = generate_synthetic_data()

    # ── Initialise state ──────────────────────────────────────────────────────
    initial_state: TrafficPredictionState = {
        # Input data
        "x_t":                x_t,
        "ground_truth":       ground_truth,
        "target_date":        resolved_date,
        # Prompt components (populated by NODE 1)
        "p_exam":             "",
        "p_input":            "",
        "p_ques":             "",
        # Prediction history
        "y_hat_current":      [],
        "y_hat_history":      [],
        # Feedback
        "mae_score":          float("inf"),
        "mae_history":        [],
        "sine_fit_actual":    "",
        "sine_fit_predicted": "",
        "pfeed_current":      "",
        "pfeed_history":      [],
        # Refinement
        "prefine_current":    "",
        "prefine_history":    [],
        # Control
        "iteration":          0,
        "max_iterations":     max_iterations or settings.max_iterations,
        "convergence_threshold": convergence_threshold or settings.convergence_threshold,
        "converged":          False,
        "context_days":       0,
        # Telemetry
        "api_calls_count":    0,
        "total_tokens_used":  0,
    }

    # ── Build and invoke graph ────────────────────────────────────────────────
    app = build_graph(enable_checkpointing=True)

    invoke_config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "Invoking graph | thread_id=%r | target_date=%s",
        thread_id, resolved_date,
    )

    final_state: TrafficPredictionState = app.invoke(
        initial_state, config=invoke_config
    )

    # ── Compile final metrics ─────────────────────────────────────────────────
    y_final      = final_state["y_hat_current"]
    mae_history  = final_state.get("mae_history", [])
    final_mae    = mae_history[-1] if mae_history else None

    # Supplementary metrics computed locally
    final_rmse   = compute_rmse(ground_truth, y_final) if y_final else None
    final_mape   = compute_mape(ground_truth, y_final) if y_final else None

    # API call accounting:
    # NODE 2 (initial) = 1
    # Each refinement loop = 3 calls (mae + sine + refine)
    total_refinements   = final_state["iteration"]          # incremented in refine node
    expected_api_calls  = 1 + 3 * total_refinements
    actual_api_calls    = final_state.get("api_calls_count", expected_api_calls)

    results = {
        # Core outputs
        "target_date":         final_state["target_date"],
        "y_final":             y_final,
        "y_hat_history":       final_state.get("y_hat_history", []),
        # MAE tracking
        "final_mae":           final_mae,
        "final_rmse":          final_rmse,
        "final_mape":          final_mape,
        "mae_history":         mae_history,
        # Convergence
        "total_iterations":    total_refinements,
        "converged":           final_state["converged"],
        "convergence_threshold": final_state["convergence_threshold"],
        # Telemetry
        "total_api_calls":     actual_api_calls,
        "total_tokens_used":   final_state.get("total_tokens_used", 0),
        # Sine patterns
        "sine_fit_actual":     final_state.get("sine_fit_actual", ""),
        "sine_fit_predicted":  final_state.get("sine_fit_predicted", ""),
        # Context pipeline fields (used by build_p_exam_from_contexts)
        "x_t":                 final_state.get("x_t", []),
        "ground_truth":        final_state.get("ground_truth", []),
        "pfeed_history":       final_state.get("pfeed_history", []),
        "prefine_history":     final_state.get("prefine_history", []),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(results)

    return results


def build_deployment_graph() -> "CompiledGraph":  # noqa: F821
    """
    Compile a minimal 2-node graph used for the deployment prediction call.

    Only NODE 1 (build_initial_prompt) and NODE 2 (initial_predict) run.
    The full evaluation loop is skipped — context comes from saved histories.

    Returns:
        A compiled LangGraph application ready for ``.invoke()``.
    """
    workflow = StateGraph(TrafficPredictionState)
    workflow.add_node("build_initial_prompt", build_initial_prompt)
    workflow.add_node("initial_predict",      api_call_initial_predict)
    workflow.set_entry_point("build_initial_prompt")
    workflow.add_edge("build_initial_prompt", "initial_predict")
    workflow.add_edge("initial_predict", END)
    app = workflow.compile()
    logger.info("Deployment graph compiled (2 nodes, no refinement loop)")
    return app


def run_context_pipeline(
    data_path:             str,
    target_date:           str,
    context_days:          int,
    max_iterations:        Optional[int]   = None,
    convergence_threshold: Optional[float] = None,
    thread_id:             str             = "default",
) -> dict:
    """
    Full context-aware pipeline:

    Phase 1 — Evaluation:
        For each of the *context_days* days before target_date, run the
        complete 7-node refinement pipeline and save the converged context.

    Phase 2 — Deployment:
        Load all saved contexts, build an enriched p_exam, then make a
        single LLM call to predict target_date.

    Args:
        data_path:             Path to the hourly traffic CSV.
        target_date:           ISO date string of the day to predict.
        context_days:          Number of previous days to evaluate.
        max_iterations:        Refinement loop limit for each evaluation day.
        convergence_threshold: MAE Δ threshold for each evaluation day.
        thread_id:             LangGraph thread ID prefix.

    Returns:
        Deployment result dict with final prediction and metadata.
    """
    setup_tracing()
    sep = "=" * 70
    logger.info(sep)
    logger.info("Context-Aware Prediction Pipeline")
    logger.info("Target: %s | Context days: %d | Max iterations: %d",
                target_date,
                context_days,
                max_iterations or settings.max_iterations)
    logger.info(sep)

    # ── Step 1: Validate all required dates exist in CSV ──────────────────────
    validate_context_dates(data_path, target_date, context_days)

    # ── Step 2: Determine evaluation dates (oldest → newest) ──────────────────
    tgt = pd.Timestamp(target_date).date()
    eval_dates = [
        str(tgt - timedelta(days=i))
        for i in range(context_days, 0, -1)
    ]
    logger.info("Evaluation dates: %s", eval_dates)

    # ── Step 3: Run full pipeline for each evaluation date ────────────────────
    contexts: List[dict] = []
    contexts_dir = os.path.join("results", "contexts")
    os.makedirs(contexts_dir, exist_ok=True)

    for i, eval_date in enumerate(eval_dates):
        logger.info("%s", "-" * 60)
        logger.info("EVALUATION %d/%d → %s", i + 1, context_days, eval_date)

        result = run_pipeline(
            data_path=data_path,
            target_date=eval_date,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            thread_id=f"{thread_id}_ctx_{eval_date}",
        )

        # Enrich result with eval_date for context builder
        result["eval_date"] = eval_date

        # Save individual context to disk
        ctx_path = os.path.join(contexts_dir, f"{eval_date}.json")
        with open(ctx_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        logger.info("Saved context → %s", ctx_path)

        contexts.append(result)

    # ── Step 4: Build enriched p_exam from all converged contexts ─────────────
    logger.info("%s", "=" * 70)
    logger.info("DEPLOYMENT — building enriched p_exam from %d contexts", len(contexts))
    enriched_p_exam = build_p_exam_from_contexts(contexts)

    # ── Step 5: Load x[t] for deployment (day before target) ──────────────────
    x_t, resolved_target = load_deployment_input(data_path, target_date)

    # ── Step 6: Build deployment state with enriched p_exam ───────────────────
    deploy_state: TrafficPredictionState = {
        "x_t":                   x_t,
        "ground_truth":          [],          # unknown — future day
        "target_date":           resolved_target,
        "p_exam":                enriched_p_exam,
        "p_input":               "",          # built by NODE 1
        "p_ques":                "",          # built by NODE 1
        "y_hat_current":         [],
        "y_hat_history":         [],
        "mae_score":             float("inf"),
        "mae_history":           [],
        "sine_fit_actual":       "",
        "sine_fit_predicted":    "",
        "pfeed_current":         "",
        "pfeed_history":         [],
        "prefine_current":       "",
        "prefine_history":       [],
        "iteration":             0,
        "max_iterations":        1,
        "convergence_threshold": convergence_threshold or settings.convergence_threshold,
        "converged":             False,
        "context_days":          context_days,
        "api_calls_count":       0,
        "total_tokens_used":     0,
    }

    # ── Step 7: Single LLM call via deployment graph ──────────────────────────
    app = build_deployment_graph()
    logger.info("Invoking deployment graph | target=%s", resolved_target)
    final_state = app.invoke(deploy_state)

    # ── Step 8: Compile deployment result ─────────────────────────────────────
    eval_api_calls = sum(c.get("total_api_calls", 0) for c in contexts)
    deploy_result = {
        # Core
        "mode":               "context_pipeline",
        "target_date":        resolved_target,
        "context_days":       context_days,
        "y_final":            final_state["y_hat_current"],
        # Metadata
        "eval_dates":         eval_dates,
        "eval_mae_finals":    [c.get("final_mae") for c in contexts],
        "total_eval_api_calls":   eval_api_calls,
        "deploy_api_calls":   1,
        "total_api_calls":    eval_api_calls + 1,
        "total_tokens_used":  (
            sum(c.get("total_tokens_used", 0) for c in contexts)
            + final_state.get("total_tokens_used", 0)
        ),
        "note": (
            f"Predicted using {context_days} converged evaluation contexts "
            f"as in-context learning examples."
        ),
    }

    _print_deployment_summary(deploy_result)
    return deploy_result


def _print_deployment_summary(results: dict) -> None:
    """Print a formatted deployment result summary."""
    sep = "─" * 70
    logger.info(sep)
    logger.info("CONTEXT PIPELINE COMPLETE")
    logger.info(sep)
    logger.info("Target date          : %s", results["target_date"])
    logger.info("Context days used    : %d  (%s)",
                results["context_days"],
                ", ".join(results["eval_dates"]))
    logger.info("Eval final MAEs      : %s",
                [f"{m:.4f}" for m in results["eval_mae_finals"] if m])
    logger.info(sep)
    logger.info("Total API calls      : %d  (eval=%d + deploy=1)",
                results["total_api_calls"],
                results["total_eval_api_calls"])
    logger.info("Total tokens used    : %d", results["total_tokens_used"])
    logger.info(sep)
    logger.info("Final prediction (ŷ): %s",
                ", ".join(f"{v:.2f}" for v in results["y_final"]))
    logger.info(sep)


def _print_summary(results: dict) -> None:
    """Print a formatted result summary to stdout."""
    sep = "─" * 70

    logger.info(sep)
    logger.info("PIPELINE COMPLETE — RESULT SUMMARY")
    logger.info(sep)
    logger.info("Target date          : %s", results["target_date"])
    logger.info("Total iterations     : %d", results["total_iterations"])
    logger.info("Converged            : %s", results["converged"])
    logger.info(sep)
    logger.info("Final MAE            : %.4f Mbps", results["final_mae"] or 0)
    logger.info("Final RMSE           : %.4f Mbps", results["final_rmse"] or 0)
    logger.info("Final MAPE           : %.2f %%",    results["final_mape"] or 0)
    logger.info("MAE history          : %s",
                [f"{m:.4f}" for m in results["mae_history"]])
    logger.info(sep)
    logger.info("Total API calls      : %d", results["total_api_calls"])
    logger.info("Total tokens used    : %d", results["total_tokens_used"])
    logger.info(sep)
    logger.info("Final prediction (ŷ): %s",
                ", ".join(f"{v:.2f}" for v in results["y_final"]))
    logger.info(sep)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LLM Network Traffic Prediction — Iterative Prompt Refinement\n"
            "Powered by LangGraph + LangChain (GPT-4o)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to hourly traffic CSV (columns: datetime, traffic_volume). "
             "Omit to use synthetic data.",
    )
    parser.add_argument(
        "--target-date", type=str, default=None,
        help="ISO date string to predict (e.g. 2024-01-15). "
             "Omit to use the last date in the CSV.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help=f"Max refinement iterations (default: {settings.max_iterations}).",
    )
    parser.add_argument(
        "--convergence-threshold", type=float, default=None,
        help=f"MAE delta below which we declare convergence "
             f"(default: {settings.convergence_threshold}).",
    )
    parser.add_argument(
        "--thread-id", type=str, default="default",
        help="LangGraph thread ID for checkpointing (default: 'default').",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results as JSON (e.g. results/run_001.json).",
    )
    parser.add_argument(
        "--context-days", type=int, default=None,
        help=(
            "Number of previous days to evaluate as in-context learning examples "
            "before predicting --target-date. When set, the full context pipeline "
            "runs: evaluate N days → save contexts → single deployment prediction. "
            "Requires --data-path and --target-date."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── Route: context pipeline vs single evaluation ───────────────────────────
    if args.context_days:
        if not args.data_path or not args.target_date:
            logger.error(
                "--context-days requires both --data-path and --target-date"
            )
            sys.exit(1)
        results = run_context_pipeline(
            data_path=args.data_path,
            target_date=args.target_date,
            context_days=args.context_days,
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold,
            thread_id=args.thread_id,
        )
    else:
        results = run_pipeline(
            data_path=args.data_path,
            target_date=args.target_date,
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold,
            thread_id=args.thread_id,
        )

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info("Results saved → %s", os.path.abspath(args.output))

    # Print a compact JSON summary to stdout for CI / scripting
    compact = {
        "final_mae":       results["final_mae"],
        "final_rmse":      results["final_rmse"],
        "final_mape":      results["final_mape"],
        "total_iterations": results["total_iterations"],
        "converged":       results["converged"],
        "total_api_calls": results["total_api_calls"],
        "total_tokens":    results["total_tokens_used"],
        "mae_history":     results["mae_history"],
    }
    print("\n" + json.dumps(compact, indent=2, default=str))
    sys.exit(0)
