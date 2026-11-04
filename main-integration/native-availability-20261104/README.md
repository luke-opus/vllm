# Native runner availability snapshot, 2026-11-04

RISE Personal was installed for `luke-opus/vllm` only. Its standard native
scalar BF16 comparison completed successfully in
[run 41830864789](https://github.com/luke-opus/vllm/actions/runs/41830864789);
see the [separate complete result](../native-20261104/).

The public read-only scheduler confirmed at 13:52 UTC that all three additional
jobs were accepted and stored for this fork. This establishes that the app and
label routing worked, while GitHub still reported them queued with no runner:

| Run | Requested pool | Scheduler state |
| --- | --- | --- |
| [41829702428](https://github.com/luke-opus/vllm/actions/runs/41829702428) | K3, Ubuntu 24.04 + rva23 | pending, no assigned worker |
| [41834148442](https://github.com/luke-opus/vllm/actions/runs/41834148442) | K3, Ubuntu 26.04 | pending, no assigned worker |
| [41840242348](https://github.com/luke-opus/vllm/actions/runs/41840242348) | EM-RV1, Ubuntu 24.04, full CPU/OpenMP worker attempt | pending, no assigned worker |

The same live snapshot reports 1,675 pending and 48 running jobs in the standard
EM-RV1 pool, with seven pending and 48 running workers. The K3 pool reports three
pending jobs and no active workers in this response. These are a snapshot of
jobs/workers, not an inventory of physical node capacity or a wait-time estimate.
No result is inferred from labels, accepted jobs or queue position.

The [FAQ](https://riscv-runners.riseproject.dev/docs/faq.html) still describes
26.04/RVA23 as not enabled. The deployed scheduler nevertheless accepted both
K3 label combinations at this timestamp. Acceptance is stronger evidence of
routing than repository source alone; it does not establish usable K3 hardware.
The [official scheduler docs](https://riscv-runners.riseproject.dev/docs/architecture/scheduler.html)
and [runbooks](https://riscv-runners.riseproject.dev/docs/operations/runbooks.html)
document public read-only `.json` dashboards. The endpoint in the snapshot was
obtained from the public production service address in the official diagnostic
script and the scheduler container name in its deployment configuration.
No provider secret or administrative interface was used.

`scheduler-snapshot.json` intentionally retains only this fork's job rows and
global aggregate counts. The original response SHA256 is recorded locally and
here; other projects' individual rows are not redistributed. The snapshot is
not a continuously updated feed. Existing successful software/QEMU checks do
not satisfy these still-unexecuted native RVV/full-worker requirements.
