# opdlens Trackio frontend

This directory serves Trackio's bundled dashboard with an additional **Eval
Matrix** page. The page uses Trackio's existing project/run sidebar and batch
log API; it does not require a custom backend.

```bash
TRACKIO_DIR=$PWD/runs/trackio uv run trackio show \
  --frontend frontend --host 0.0.0.0
```

The files in `assets/` are the unmodified Trackio 0.30.3 production bundle.
The project-specific extension lives in `eval-matrix/`.
