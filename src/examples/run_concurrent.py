#!/usr/bin/env python
"""
Concurrent DefenderBench runner.

For parallelizable environments (Phishing, CTI MCQ, Code Vulnerability Detection,
CVEFix), each sample is processed independently in a thread pool.
CyberBattle environments fall back to sequential execution.

Usage:
    python -m src.examples.run_concurrent \
        --model "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
        --env_names CyberThreatIntelligenceMultiChoiceQuestions \
        --concurrency 4
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gymnasium as gym
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from termcolor import colored
from tqdm import tqdm

from src.agents.react_agent import ReActAgent
from src.utils.llm_api import LLM
from src.defenderbench import benchmark_v0, benchmark_small_v0
from src.defenderbench.utils import extract_code

# ── Environments that require sequential execution (stateful multi-step) ──
SEQUENTIAL_ENVS = {
    'CyberBattleChain2', 'CyberBattleChain4', 'CyberBattleChain10',
    'CyberBattleTiny', 'CyberBattleToyCTF',
}

# ── Environments that produce code (evaluated via CodeBLEU) ──
CODEFIX_ENVS = {'CVEFix', 'CVEFixSmall'}


def build_react_messages(instructions, observation):
    """Build ReAct‐style chat messages for a single sample."""
    system_prompt = (
        "You are a cybersecurity assistant that uses the ReAct framework to decide on actions. "
        "First, reason step by step by writing your thinking process "
        "(each step should begin with 'Thought:'). "
        "Once you are confident, output your final decision on a new line starting with "
        "'Action:' followed by only that final actionable decision."
    )
    user_prompt = f"Instructions: {instructions}\n\nObservation: {observation}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Per‑sample processing functions (run inside thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def process_classification_sample(llm, instructions, observation, class_labels,
                                  gold_label, max_retries=5):
    """Classify a single sample with up to *max_retries* attempts.

    Returns (prediction, reward).
    """
    messages = build_react_messages(instructions, observation)

    for _attempt in range(max_retries):
        try:
            response = llm(messages)
        except Exception:
            return '', 0

        if not response:
            continue  # LLM returned None / empty

        # Extract "Action: …" line
        match = re.search(r"Action:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        raw_action = match.group(1).strip() if match else response

        try:
            json_str = extract_code(raw_action, code_type="json")
            parsed = json.loads(json_str.lower().replace("'", '"'))
            prediction = parsed.get('answer', '') if isinstance(parsed, dict) else str(parsed)
            prediction = prediction.strip()

            labels_lower = [l.lower() for l in class_labels]
            if prediction.lower() in labels_lower:
                idx = labels_lower.index(prediction.lower())
                canonical = class_labels[idx]
                reward = int(canonical.lower() == gold_label.lower())
                return canonical, reward
        except Exception:
            pass

        # Append retry feedback
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content":
            f'Your answer is invalid. Your answer must be a JSON dictionary like '
            f'{{"answer": "{class_labels[0]}"}} where the value must be one of: '
            f'{", ".join(class_labels)}.'
        })

    return '', 0


def process_codefix_sample(llm, instructions, observation, gold_code, language,
                           max_retries=5):
    """Fix vulnerable code for a single sample.

    Returns (predicted_code, codebleu_score).
    LLM call is done here; CodeBLEU is deferred to the caller if needed.
    """
    messages = build_react_messages(instructions, observation)

    for _attempt in range(max_retries):
        try:
            response = llm(messages)
        except Exception:
            return '', 0.0

        match = re.search(r"Action:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        raw_action = match.group(1).strip() if match else response

        if "```" in raw_action:
            prediction = extract_code(raw_action, code_type=language)
            return prediction, None  # CodeBLEU computed later (not thread‑safe)

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content":
            "Your answer is invalid. Your answer must be a markdown code block "
            "of the entire source code once fixed including comments."
        })

    return '', None


# ─────────────────────────────────────────────────────────────────────────────
# Sample collection — pre‑extract all (observation, label) pairs from the env
# ─────────────────────────────────────────────────────────────────────────────

def collect_samples(env_name):
    """Create gym env, trigger data download, and return a list of sample dicts."""
    env = gym.make(f"defenderbench/{env_name}-v0", disable_env_checker=True)
    uw = env.unwrapped

    is_codefix = env_name in CODEFIX_ENVS or env_name.startswith('CVEFix')
    is_classification = hasattr(uw, 'CLASS_LABELS')

    # Reset loads the first sample
    _, _info = env.reset()
    samples = []

    if is_codefix:
        samples.append({
            'observation': uw.build_observation(),
            'instructions': uw.instructions,
            'gold_code': uw.target,
            'language': uw.language,
        })
        while uw.next_sample():
            samples.append({
                'observation': uw.build_observation(),
                'instructions': uw.instructions,
                'gold_code': uw.target,
                'language': uw.language,
            })
    elif is_classification:
        class_labels = uw.CLASS_LABELS
        samples.append({
            'observation': uw.build_observation(),
            'instructions': uw.instructions,
            'gold_label': uw.sample_label,
            'class_labels': class_labels,
        })
        while uw.next_sample():
            samples.append({
                'observation': uw.build_observation(),
                'instructions': uw.instructions,
                'gold_label': uw.sample_label,
                'class_labels': class_labels,
            })

    env.close()
    return samples, is_codefix, is_classification


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_concurrent(env_name, llm, concurrency, max_retries=5):
    """Run a parallelizable env with *concurrency* threads."""
    print(f"\n{'='*60}")
    print(f"  {env_name}  (concurrency={concurrency})")
    print(f"{'='*60}")

    t0 = time.time()
    print(f"  Loading data …")
    samples, is_codefix, is_classification = collect_samples(env_name)
    print(f"  {len(samples)} samples loaded in {time.time()-t0:.1f}s")

    results = [None] * len(samples)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for idx, s in enumerate(samples):
            if is_codefix:
                f = pool.submit(process_codefix_sample, llm,
                                s['instructions'], s['observation'],
                                s['gold_code'], s['language'], max_retries)
            else:
                f = pool.submit(process_classification_sample, llm,
                                s['instructions'], s['observation'],
                                s['class_labels'], s['gold_label'], max_retries)
            futures[f] = idx

        pbar = tqdm(total=len(samples), desc=env_name, leave=True)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(colored(f"\n  Error on sample {idx}: {e}", "red"))
                results[idx] = ('', 0) if not is_codefix else ('', None)
            pbar.update(1)
        pbar.close()

    elapsed = time.time() - t0

    # ── Metrics ──────────────────────────────────────────────────
    if is_codefix:
        # Compute CodeBLEU sequentially (tree‑sitter is not thread‑safe)
        from codebleu import calc_codebleu
        from src.defenderbench.utils import catch_logging

        scores = []
        for idx, (pred, _) in enumerate(results):
            s = samples[idx]
            if not pred:
                scores.append(0.0)
                continue
            try:
                with catch_logging() as log:
                    w = (0.1, 0.1, 0.4, 0.4)
                    m = calc_codebleu([s['gold_code']], [pred],
                                      lang=s['language'].replace("c++", "cpp"), weights=w)
                    log.seek(0)
                    if log.read():
                        w = (0.1, 0.1, 0.8, 0.0)
                        m = calc_codebleu([s['gold_code']], [pred],
                                          lang=s['language'].replace("c++", "cpp"), weights=w)
                scores.append(m["codebleu"])
            except Exception:
                scores.append(0.0)

        total_score = sum(scores)
        max_score = len(samples)
        normalized = total_score / max_score if max_score else 0

        metrics = {
            "avg_codebleu": normalized,
            "total_samples": max_score,
            "elapsed_s": round(elapsed, 2),
            "avg_latency_ms": round(elapsed / len(samples) * 1000, 0) if samples else 0,
        }

        print(f"\n  Score      : {total_score:.2f}/{max_score} ({normalized:.1%})")
        print(f"  Avg CodeBLEU: {normalized:.4f}")

    else:  # classification
        gold_labels = [s['gold_label'] for s in samples]
        predictions = [r[0] for r in results]
        rewards = [r[1] for r in results]

        class_labels = samples[0]['class_labels']
        gl = [g.lower() for g in gold_labels]
        pl = [p.lower() if p else '' for p in predictions]
        cl = [l.lower() for l in class_labels]

        acc = accuracy_score(gl, pl)
        macf1 = f1_score(gl, pl, average='macro', zero_division=0, labels=cl)
        macpre = precision_score(gl, pl, average='macro', zero_division=0, labels=cl)
        macrec = recall_score(gl, pl, average='macro', zero_division=0, labels=cl)

        total_score = sum(rewards)
        max_score = len(samples)
        normalized = total_score / max_score if max_score else 0

        metrics = {
            "acc": round(acc, 4),
            "macf1": round(macf1, 4),
            "macro_precision": round(macpre, 4),
            "macro_recall": round(macrec, 4),
            "correct": total_score,
            "total": max_score,
            "elapsed_s": round(elapsed, 2),
            "avg_latency_ms": round(elapsed / len(samples) * 1000, 0) if samples else 0,
        }

        print(f"\n  Score    : {total_score}/{max_score} ({normalized:.1%})")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {macf1:.4f}")
        print(f"  Precision: {macpre:.4f}")
        print(f"  Recall   : {macrec:.4f}")

    print(f"  Time     : {elapsed:.1f}s  ({metrics['avg_latency_ms']:.0f}ms avg/sample)")
    return normalized, metrics


def run_sequential(env_name, agent, args):
    """Fall back to the original sequential loop for CyberBattle envs."""
    print(f"\n{'='*60}")
    print(f"  {env_name}  (sequential)")
    print(f"{'='*60}")

    t0 = time.time()
    score = agent.run(env_name, args)
    elapsed = time.time() - t0

    metrics = {"normalized_score": score, "elapsed_s": round(elapsed, 2)}
    print(f"  Score: {score:.4f}  ({elapsed:.1f}s)")
    return score, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    llm = LLM(args.model, verbose=args.verbose)
    agent = None  # lazy‑created only if needed

    all_results = {}
    total_t0 = time.time()

    for env_name in args.env_names:
        if env_name in SEQUENTIAL_ENVS:
            if agent is None:
                agent = ReActAgent(llm)
            score, metrics = run_sequential(env_name, agent, args)
        else:
            score, metrics = run_concurrent(env_name, llm, args.concurrency,
                                            max_retries=args.max_retries)
        all_results[env_name] = {"score": score, "metrics": metrics}

    total_elapsed = time.time() - total_t0

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for env_name, r in all_results.items():
        print(f"  {env_name:55s} {r['score']:.4f}")
    if len(all_results) > 1:
        avg = sum(r['score'] for r in all_results.values()) / len(all_results)
        print(f"  {'─'*55}")
        print(f"  {'Average':55s} {avg:.4f}")
    print(f"\n  Total wall time: {total_elapsed:.1f}s")

    # ── Save results ─────────────────────────────────────────────
    model_short = args.model.split('/')[-1]
    ts = time.strftime('%Y%m%d_%H%M%S')
    results_file = f"results_{model_short}_{ts}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "model": args.model,
            "agent": args.agent,
            "concurrency": args.concurrency,
            "total_elapsed_s": round(total_elapsed, 2),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent DefenderBench runner")
    parser.add_argument('--agent', default='react',
                        help='Agent type for sequential envs (default: react)')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--env_names', nargs='+', help='Environments to run')
    parser.add_argument('--concurrency', type=int, default=4,
                        help='Concurrent LLM calls for parallelizable envs (default: 4)')
    parser.add_argument('--max-retries', type=int, default=5,
                        help='Max retries on invalid answers (default: 5)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--small', action='store_true',
                        help='Use benchmark_small_v0 (100 samples per task)')

    args = parser.parse_args()
    if args.env_names is None:
        args.env_names = benchmark_small_v0 if args.small else benchmark_v0

    main(args)
