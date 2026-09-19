# Reproduction and handoff checklist

This scaffold does not claim any successful training run.

- [ ] Import tested AIM-ZO code and its minimal runtime dependencies.
- [ ] Import only baselines used in the paper; retain upstream attribution/licenses.
- [ ] Preserve a mapping from original algorithm entry points to released entry points.
- [ ] Populate dependencies and pin tested versions; confirm Python compatibility.
- [ ] Supply configurable model/data/cache paths, never personal absolute paths.
- [ ] Add dataset preparation and official evaluation commands.
- [ ] Freeze actual main/baseline/ablation/diagnostic configurations.
- [ ] Import verified compact artifacts into the result registry.
- [ ] Implement registry validation and CSV/LaTeX export.
- [ ] Install in a clean environment and run a small end-to-end smoke test.
- [ ] Verify that main reproduction commands agree with paper budgets and selection rules.
- [ ] Review code, paths, metadata, history, external links, and artifacts for anonymity/secrets.
- [ ] Confirm redistribution rights and choose the project license; preserve third-party notices.
- [ ] Create an anonymity-reviewed release snapshot and verify anonymous access/download.
- [ ] Add the verified anonymous URL to the paper.

Prefer a small tested code extraction over a large algorithmic refactor during submission preparation. Do not replace the internal development repository. Keep incomplete functionality explicitly marked rather than providing success-looking placeholder scripts.

