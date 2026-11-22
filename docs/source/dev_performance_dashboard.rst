Performance Dashboard
=====================

MBIRTorch's reconstruction performance — run time, peak memory, and **correctness** — is tracked
automatically over time, on both CPU and GPU, by a companion project,
`mbirtorch_metrics <https://github.com/cabouman/mbirtorch_metrics>`__.  When a tracked branch changes, a
scheduled job re-measures it and publishes an interactive dashboard:

**Live dashboard:** https://cabouman.github.io/mbirtorch_metrics/

The dashboard has a ``cpu`` and a ``gpu`` platform, each with one line per tracked branch.
The dashboard rebuilds and republishes automatically whenever new measurements are pushed, so that
link is always current; you do not need to run anything to read it.

What it measures
----------------

For each tracked branch, the job runs MBIRTorch's reconstruction operators — the
direct-reconstruction filter, forward projection, back projection, and the iterative VCD
reconstruction — across a range of problem sizes and device counts, on both CPU and GPU.  For every
configuration it records:

- **run time** (the minimum over repeated trials),
- **peak memory**, and
- a numeric **fingerprint** of the output, used to detect correctness changes.

How to read it
--------------

The dashboard explains itself: open the live page and expand the **"How to read this dashboard"**
panel at the top.  It describes the tiles, the red correctness banner, the History and Scaling
views, and the colors & marks.

That reading guide is authored *inside* the dashboard and ships in the page itself, so it can never
drift from the UI it describes — which is why this page links to it rather than duplicating it.

Running it yourself
-------------------

The dashboard is a single self-contained page generated from a YAML time series; no server is needed.
The measurement engine, the nightly harness, and the build script all live in the
`mbirtorch_metrics <https://github.com/cabouman/mbirtorch_metrics>`__ repository — see its ``README`` and
the ``action_scripts/`` and ``tooling/`` guides there for how runs are measured, gated, and scheduled,
and how to build the dashboard locally.
