#!/usr/bin/env python3
"""Générateur de charge équivalent au plan JMeter tests/load/churn_load_test.jmx.

Il reproduit le « Thread Group » JMeter : N utilisateurs virtuels démarrés
progressivement (ramp-up), chacun envoyant L requêtes POST /predict successives
sur une connexion HTTP persistante (keep-alive), sans temps de réflexion.

Il sert quand JMeter n'est pas installé. Les métriques calculées sont celles de
l'« Aggregate Report » JMeter : moyenne, médiane, p90/p95/p99, min/max, % d'erreurs,
débit (requêtes/s). Les percentiles utilisent la méthode du rang le plus proche.

Exemple :
    python tests/load/load_runner.py --users 100 --ramp-up 10 --loops 10 \
        --url http://localhost:8000 --label "100 utilisateurs" --json results/load_100.json
"""
import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

import httpx

PAYLOAD = {
    "age": 35,
    "anciennete": 5,
    "revenu": 450000,
    "nb_reclamations": 2,
    "montant_mensuel": 25000,
    "retards_paiement": 2,
}


def percentile(sorted_values, p):
    """Percentile par rang le plus proche (valeur d'un échantillon réel)."""
    if not sorted_values:
        return float("nan")
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


async def virtual_user(index, args, client, samples, t_origin):
    # Ramp-up : l'utilisateur i démarre à i * (ramp-up / nb utilisateurs) secondes
    await asyncio.sleep(index * args.ramp_up / args.users)
    for _ in range(args.loops):
        start = time.perf_counter()
        try:
            response = await client.post(args.path, json=PAYLOAD)
            code, ok = response.status_code, response.status_code == 200
        except httpx.HTTPError as exc:
            code, ok = type(exc).__name__, False
        end = time.perf_counter()
        samples.append((start - t_origin, end - t_origin, (end - start) * 1000.0, ok, code))


async def run(args):
    samples = []
    limits = httpx.Limits(max_connections=args.users, max_keepalive_connections=args.users)
    async with httpx.AsyncClient(
        base_url=args.url, limits=limits, timeout=httpx.Timeout(args.timeout), verify=False
    ) as client:
        t_origin = time.perf_counter()
        await asyncio.gather(
            *(virtual_user(i, args, client, samples, t_origin) for i in range(args.users))
        )
    return samples


def summarize(samples, args):
    durations = sorted(s[2] for s in samples)
    errors = [s for s in samples if not s[3]]
    wall = max(s[1] for s in samples) - min(s[0] for s in samples)
    error_codes = {}
    for s in errors:
        error_codes[str(s[4])] = error_codes.get(str(s[4]), 0) + 1
    return {
        "label": args.label,
        "users": args.users,
        "ramp_up_s": args.ramp_up,
        "loops": args.loops,
        "samples": len(samples),
        "average_ms": round(statistics.fmean(durations), 1),
        "median_ms": round(percentile(durations, 50), 1),
        "p90_ms": round(percentile(durations, 90), 1),
        "p95_ms": round(percentile(durations, 95), 1),
        "p99_ms": round(percentile(durations, 99), 1),
        "min_ms": round(durations[0], 1),
        "max_ms": round(durations[-1], 1),
        "error_count": len(errors),
        "error_pct": round(100.0 * len(errors) / len(samples), 2),
        "error_codes": error_codes,
        "duration_s": round(wall, 1),
        "throughput_rps": round(len(samples) / wall, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--path", default="/predict")
    parser.add_argument("--users", type=int, default=100, help="nombre d'utilisateurs virtuels (threads JMeter)")
    parser.add_argument("--ramp-up", type=float, default=10.0, help="durée de montée en charge, en secondes")
    parser.add_argument("--loops", type=int, default=10, help="requêtes par utilisateur")
    parser.add_argument("--timeout", type=float, default=60.0, help="timeout par requête, en secondes")
    parser.add_argument("--label", default="")
    parser.add_argument("--json", help="fichier JSON de sortie")
    args = parser.parse_args()
    args.label = args.label or f"{args.users} utilisateurs"

    samples = asyncio.run(run(args))
    summary = summarize(samples, args)

    print(f"\n=== {summary['label']} : {summary['users']} users x {summary['loops']} boucles, ramp-up {summary['ramp_up_s']} s ===")
    print(f"Échantillons : {summary['samples']}    Durée : {summary['duration_s']} s    Débit : {summary['throughput_rps']} req/s")
    print(f"Temps de réponse (ms) : moy {summary['average_ms']} | méd {summary['median_ms']} | "
          f"p90 {summary['p90_ms']} | p95 {summary['p95_ms']} | p99 {summary['p99_ms']} | "
          f"min {summary['min_ms']} | max {summary['max_ms']}")
    print(f"Erreurs : {summary['error_count']} ({summary['error_pct']} %) {summary['error_codes'] or ''}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
