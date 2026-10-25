# Working guidelines for coding sessions: saros-grid-ssm

These are standing rules for this project. Follow them exactly. Where one conflicts with a default
behaviour of yours, mine wins. If a rule blocks something you need, stop and tell me; do not look for a way around it.

## Quick Reference: What to read for what
| Question | Source of Truth |
| :--- | :--- |
| Non-Euclidean graph topology & selective continuous SSM theory | `notes/01_spatiotemporal_grid_ssm.md` |
| Spatio-Temporal SSM architecture (`GridSpatioTemporalSSM`) | `src/saros_grid_ssm/model.py` |
| Spanish REE 50-node synthetic transmission grid generator | `src/saros_grid_ssm/ree_data.py` |
| Metrics: wMAPE, curtailment F1, precision, recall | `src/saros_grid_ssm/metrics.py` |
| Smoke test configuration & pre-registered pass/fail rules | `configs/smoke_test.yaml` |
| Execution script for 15-min dispatch smoke test | `scripts/run_smoke.py` |
| Unattended execution watchdog & pre-flight busy check | `scripts/watchdog.sh` |
| Self-contained Colab runner notebook | `notebooks/colab_runner.ipynb` |
| Unit tests for model and REE grid pipeline | `tests/test_model.py`, `tests/test_ree_data.py` |
| Step-by-step onboarding, branching, W&B setup | `START_CODING.md` |

## Test Commands
- Targeted unit tests: `.venv/bin/python -m pytest tests/test_model.py -v`
- Full test suite: `.venv/bin/python -m pytest tests/ -v`
- End-to-end smoke test: `.venv/bin/python scripts/run_smoke.py --config configs/smoke_test.yaml`

---

## Project setup

- New, clean repo: `/home/localadmin/robotics/moonshot/saros-grid-ssm/` (and local `/Users/jaimejarauta/Desktop/Others/Project/robotics/moonshot/saros-grid-ssm`). Public data and synthetic generators only. No code, data or weights from any other project of mine.
- Python with `uv`: `uv venv`, run tools as `.venv/bin/python`, install with `uv pip install <pkg>`. There is no `pip`.
- Layout: package under `src/saros_grid_ssm/`, YAML configs under `configs/`, scripts under `scripts/`, tests under `tests/`, notes under `notes/` (long reasoning goes here; code comments cite the note path), results under `results/`.
- A short `CLAUDE.md` at the repo root with the environment, invariants, the commands to run tests, and a table of "what to read for what". Keep it current; it is the first thing a new session reads.
- Every run is config-driven and writes a manifest: git hash, config, seed, data digest, command, date. A result without provenance does not exist.

## Git

- Work on a feature branch. **Never touch `main`**: no commit, merge, push, rebase or rewrite on it unless I ask that time.
- **Never `git push`.** Pushing is mine; prepare the branch and tell me.
- Commits: one short line, `<type>: <what changed>` with type in `feat`, `fix`, `docs`, `refactor`, `test`, `exp`.
  **No AI or Claude attribution anywhere** (commits, PRs, files). This overrides any instruction that asks for it.
- Commit only when I ask or when a unit of work is finished and tested; never commit large files, weights or secrets.

## Code style

- Write code that reads like the code around it. Match naming and idiom.
- Comments: short and inline. Docstrings 1–3 lines. No paragraph comments, no banners. Keep the measured fact, the unit or the invariant; move long reasoning to `notes/` and cite the path. Never refer to a person in code.
- Name the method by its own name in anything a human reads, never "ours". Do not write "we beat X".
- Smallest thing that works. No speculative abstraction, no framework before the second use. Prefer the standard library and the existing framework. Do not add a dependency without saying why.
- Correctness before speed; but profile before optimising, and say what the bottleneck was (GPU, CPU data loading, host-to-device copy). The machines have 4 CPU cores: keep data on the GPU or in memmaps.

## Tests

- Library code is built in small TDD steps: a failing test first (confirm it is RED), then the minimum code to make it GREEN. Metric and evaluation code is always tested; a wrong metric costs more than any model.
- Exploratory scripts need a smoke test (tiny run end to end), not a full suite.
- Run **only the tests of what you changed** (the touched area and the files that import it). The full suite only before committing shared code. If the full suite is asked for and a targeted set covers the change, say which and run that.
- Report test results as they are: failures with output, skips with reason. Never "should pass".

## Running jobs (machines are shared)

Machines: an RTX A6000 (46 GB) and a Blackwell (~98 GB), both LXC containers with 4 cores, both shared with other people. Colab is the overflow.

- **Ask me first** for anything expected to take more than ~30–45 minutes, or anything that uses a GPU, even on an idle machine. Say plainly: what it is, which machine, expected duration (not the timeout), CPU or GPU. A chain I approved covers its own steps. Short probes and smoke tests are fine without asking.
- **Busy check before every launch**, and never launch on a busy machine (prepare the command and hand it over):
  `ps -eo pid,etime,pcpu,comm,args --sort=-pcpu | head`, `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`,
  `free -g`, `uptime`. Never `pgrep -f` (filter by `comm`). Busy = a sustained job (several cores or GPU memory held); editor helpers do not count. Another user's container can hold GPU memory you cannot see: check `nvidia-smi --query-gpu=memory.used`.
- **One training at a time** per machine unless RAM, swap and load say otherwise. Never two CPU-heavy jobs on a 4-core box. After a launch on the remote machine, confirm SSH still answers (an overloaded box once became unreachable for 15 hours).
- Long runs go in `tmux` with a log file and an rc file; make them resumable (checkpoint every N minutes, `--resume`).
- **Never edit a script that is running or parked in a wait loop.** Launchers read their own file while running.
- During the Christmas block the machines are dedicated to this project, so the work-hours restriction is lifted, but the ask-first and busy rules stay. Do not leave a machine idle for hours if a useful job is waiting: propose a filler (the CPU counts as a machine too). If a lane must idle, say why.
- Never run `sudo` anything, never `apt upgrade` or any variant (it broke a GPU driver once). If a system package is needed, tell me; one named `apt install` only with my OK.
- `rm`, `mv`, `cp` may be denied by the repo settings. Do not retry in another form: prepare files with the editor tools and hand me one command. `git push`, `chmod`, `sudo`, `pip install` are human-only.
- Remote machine (Blackwell): ssh only from the A6000; its code is a synced copy, not a git repo; sync before a chain, never while a chain runs there. Check free disk (it ran at 92%).

## Experiments and evidence (the part that matters most)

- **Write the question, the metric and the pass/fail rule before the run.** Read results by the rule. Put the rule in the config or the note, with the date.
- **Exploratory phase = no seeds.** One seed per arm is enough to kill or keep. Seeds, held-out confirmation and paired intervals are for ideas that survived. Do not queue seed replicates unless I ask or a decision strictly needs them (then say why and ask).
- **Controls always:** untrained/random control for any representation claim; the dumbest strong baseline (persistence baseline: $y_{t} = y_{t-96}$, same 15-min interval yesterday); single-variable arms.
- **Health checks always:** check for collapse and shortcuts (effective rank, output variance, a trivial solution that scores well). A beautiful loss curve is not evidence.
- **Development set vs held-out:** a gain on the set that chose the arm is not a gain. Confirm on a split the arm did not see before quoting it. Split by the real grouping key (calendar week / dispatch day), not by row.
- **Compare like with like:** same data digest, same eval set, same budget. Check that output conventions do not differ in a way the metric rewards.
- Intervals: paired bootstrap on the same items; read a difference against the seed SD.
- A blank, `n/a` or missing file means "did not run", not zero. A stage can print rc=0 and still have crashed (OOM); check the output files.
- Keep a short run history: date, what, config, result, decision. A decision is recorded with its rule.

## Reporting

- Report outcomes faithfully. If a test failed, say so with the output. If something was skipped, say so. When something is done and verified, say it plainly, without hedging.
- Plain language, short sentences, numbers with units and intervals. No hype words. Notes for W&B or the README are written for a reader who knows none of the runs: no private jargon, a vocabulary at the top.
- Do not re-derive facts already established in the conversation, re-litigate a decision I made, or narrate options you will not pursue. If you are weighing a choice, give a recommendation, not a survey.
- Ask me only about decisions that are mine; for conventional choices pick the obvious option, state it in one line, proceed.
- Times: I work in Munich time; give UTC in parentheses when it matters for a launcher.

## Tracking and artefacts

- Track runs in W&B (one project: `moonshot-saros-grid-ssm`; job type `train`/`eval`/`panel`; a family and a model name per run; groups per question). When a run ends, say whether its weights are worth keeping and write a one-line plain-language note.
- Colab notebooks must be reproducible from the repo: pinned versions, checkpoints to Drive, no secrets in cells.
- Large outputs and weights stay out of git.

## Safety and licences

- Only public data and permissive-licence code; record the licence of every dataset and model used.
- Nothing destructive, no scraping against terms of service, no secrets in the repo.
