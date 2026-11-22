> **Historical phase notes.** The up-to-date explanation is in `docs/algorithm.md`, `docs/results.md` and `docs/viva_guide.md`. (The heuristic weights mentioned here were later re-tuned and a monotonicity bug was fixed.)

# Phase 3: Reproducible agent experiments

`experiments.py` runs the same seeded game starts for the Random, Greedy, and
Expectimax agents. It writes two files:

- `per_game.csv`: one row per completed game, with the seed, final score,
  highest tile, moves, 2048 result, decision time, and expanded nodes.
- `summary.csv`: one row per agent setting, with average and median score,
  win rate, average highest tile, moves, decision time, and node counts.

## Quick smoke run

This short run is useful to check that the experiment system works:

```bash
python experiments.py --seeds 1-3 --depths 2 --output-dir results/smoke
```

## Fair project comparison

Use the same 50 game seeds for every agent:

```bash
python experiments.py --seeds 1-50 --depths 2 3 --max-chance-outcomes 8 --output-dir results/final
```

Depth 3 with every possible spawn can be slow. The command above limits each
chance node to eight deterministic, probability-stratified outcomes. The CSV
records this as `stratified_approximation` and stores the value `8`, so it is
clear that the result is not an exact full-branching search.

For an exact experiment, leave out `--max-chance-outcomes`. Start with a small
number of seeds because exact depth 3 can take a long time:

```bash
python experiments.py --seeds 1-3 --depths 2 3 --output-dir results/exact-small
```

## What makes the comparison fair?

Each agent receives the same seed for a game, which makes the starting board
and later random tile sequence reproducible for that agent's decisions. A
fresh agent is created for each game so its random generator and search cache
do not carry over from another seed. The runner measures only `choose_move`,
not board display or random-tile spawning.

## Viva explanation

The Random agent is the simple baseline. Greedy makes one move and picks the
board with the best immediate heuristic. Expectimax looks ahead through MAX
nodes for our choices and chance nodes for tile spawns. Comparing the CSV
summary shows whether looking ahead improves score or high-tile rate, and how
much extra decision time and search work it costs.

Use seeds outside 1-50, such as 51-60, when tuning heuristic weights. Keep
seeds 1-50 for the final report so the evaluation remains independent from
tuning.
