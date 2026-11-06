# StabilityArc / ESMC

Planned comparison against standalone StabilityArc, without target-assay supervised Kermut adaptation. The protocol is marked planned, so the training launcher must reject it rather than silently select another dataset.

Required work: pin source release and upstream code, implement the importer, materialize the outer protein holdouts, specify development-only model selection, audit overlap/labels, and reproduce upstream evaluation on identical rows. The ESMC recipe is a starting configuration, not a validated optimal recipe. No folds or accuracy results are claimed by this scaffold.
