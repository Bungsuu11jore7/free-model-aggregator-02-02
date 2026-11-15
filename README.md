# Free-model aggregator 02-02

One server (Python 3.13, stdlib only) that aggregates make.fun's fourteen
free OpenRouter models behind an OpenAI-compatible and an
Anthropic-compatible wire, for competition `free-model-aggregator-02`.

## Strategy

**Epsilon-greedy multi-armed bandit over the top-rated eligible models,
with rating-ordered failover.** Each free model is one arm. An arm's
prior value is its house rating (`rating_milli / 2000`), held with 4
pseudo-pulls; every gateway call updates it — a served prompt banks a
reward equal to the serving model's rating, a refusal banks 0, so a model
that refuses loses value for the rest of the trial. Per prompt the router
takes the eligible arms (tools / vision / JSON / context, from the
brief's table), then with probability 0.10 explores — a seeded uniform
pick among the top 3 eligible arms by current value — and otherwise
exploits the best-valued arm. Failover walks the remaining eligible arms
in current-value order: a 5xx is retried once on the same model, a 429 or
a second failure moves to the next arm (max 3 attempts per prompt,
keeping the call count low for tie-breaks). Every refusal is reported on
the response in the `x-makefun-upstream-failures` header so the failover
column can see recovered retries.

All choices are functions of the brief, the request, the seeded RNG
(`MAKEFUN_TRIAL_SEED`) and the gateway answers seen so far: no wall
clock and no unseeded randomness, so a replay leaves the same state
export byte for byte.

## Run

```
python3 main.py --list     # print the catalog and exit 0
python3 main.py            # register with the env, serve, exit 0 when done
```

The trial sets `MAKEFUN_PROMPT_FILE`, `MAKEFUN_ENV_URL`,
`MAKEFUN_TRIAL_SEED`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Every free
model is dialled on the gateway's Chat Completions route; Anthropic-shaped
requests are translated into that body and the answer translated back.
