# The AI researcher: design, and why the loop is not shipped

The autonomous loop is specified here and deliberately **not implemented**.
This is not an omission: the audited project ran its loop with
`--dangerously-skip-permissions` for up to 120 turns on a machine holding the
research record (see `ARCHITECTURE_AUDIT.md`, finding F1), and an agent that
can edit its own history cannot produce a trustworthy record of what it found.
Shipping the loop before the sandbox exists would be the most dangerous thing
in this package.

`exchange research-loop` exists only to print these requirements and exit
non-zero.

---

## 1. What the loop would do

One iteration:

1. **Read the ledger.** How many configurations have been tested on this
   dataset, and which hypotheses were already rejected and why. A rejected
   idea is never re-tested; the ledger's trial count is charged to whatever
   comes next.
2. **Scan before searching.** Run the structural scanner. If nothing is
   measured, the correct action is to stop, not to sweep families until
   something passes.
3. **Write the hypothesis first.** Statement, rationale, expected market
   mechanism, expected regime, and explicit falsification criteria — before
   any backtest. The record is hash-locked at registration.
4. **Run the funnel.** Unchanged from the human path, including the automatic
   trial charge.
5. **Attack the survivor.** Eleven attacks; the robustness score can only
   lower the verdict.
6. **Record either way.** A rejection is written with the same completeness as
   a success. Rejection is the loop's *success* case: it removes a candidate
   from the search space permanently and makes the next hypothesis' statistics
   honest.
7. **Stop conditions.** A fixed iteration budget, a fixed total trial budget
   per dataset, and a hard stop when the rejection rate collapses — a loop
   that stops rejecting has either found something remarkable or disabled its
   own gates, and the second is far more common.

The loop's output is hypotheses and records, never code changes to the
framework, and never edits to an earlier verdict.

## 2. What must exist before it runs

**Sandbox.** A container with no credentials of any kind, no host network, a
read-only mount of the package and the datasets, and a single writable path:
the experiment registry plus a scratch directory. Nothing else.

**Filesystem allowlist.** Deny by default. Read: `src/`, `configs/`, the
dataset directory. Write: `experiments/`, `reports/`, `/tmp/scratch`.
Explicitly denied: the package source, the CI configuration, the
orchestrator's own configuration, and every file under `.git/`.

**Network allowlist.** Deny by default. A research iteration needs no network
at all: the data is already on disk and the statistics are local. If a data
fetch is ever required it belongs to a separate, human-triggered step with its
own allowlist of hosts.

**No secrets.** No API keys, no broker credentials, no tokens in the
environment, the filesystem or the process arguments. An agent that cannot
reach a broker cannot send an order by mistake or by prompt injection.

**Resource budgets, enforced outside the agent.** Wall-clock per iteration,
CPU shares, memory limit, maximum number of backtests per iteration, and a
disk quota on the registry path. Enforced by the container runtime, not by
asking the agent to behave.

**Append-only registry.** The writable registry path permits creating new
experiment directories and forbids modifying existing ones. The only exception
is attaching a falsification report, which can lower a verdict and never raise
one. Without this the distinction between "rejected" and "never run" is not
preserved by the storage, and the ledger stops being evidence.

**Permission model.** The agent runs with the smallest tool set that lets it
do the work — read files, run `exchange` subcommands, write to the registry —
and with permission checks *on*. `--dangerously-skip-permissions` or its
equivalent is not an acceptable configuration for an unattended loop at any
iteration count.

**Human review before capital.** The highest verdict the funnel can award is
`PAPER_TRADING_CANDIDATE`. No automated path from a verdict to an order
exists, and none should.

## 3. Why the ordering matters

Every item above is cheap to add *before* the loop exists and expensive to add
afterwards, because by then the loop has produced a record whose integrity
nobody can verify. The research cost of running an unsandboxed loop is not
hypothetical either: the first time an agent edits an earlier verdict to be
consistent with a later one, the entire catalogue becomes unusable, and there
is no way to tell from the outside that it happened.

Until the sandbox exists, the intended workflow is the explicit one:

```
exchange scan <source>
exchange research <source> --hypothesis <file>
exchange falsify <experiment-id>
exchange reproduce <experiment-id>
```

which is the same pipeline the loop would drive, with a human deciding what to
test next.
