PYTHON ?= $(HOME)/anaconda3/envs/mutant_cache/bin/python
EXP ?= experiments/stability_paper/me_esm2_v1/experiment.json
ARGS ?=
CONFIG ?= experiments/stability_paper/me_esm2_v1/comparison.json

.PHONY: help data verify resolve launch queue status compare
help:
	@echo 'make data DATASET=mutate-everything | esm-msr | stabilityarc'
	@echo 'make verify                 # processed data content hashes'
	@echo 'make resolve EXP=path/to/experiment.json ARGS="--seed 0"'
	@echo 'make launch EXP=path/to/experiment.json ARGS="--seed 0"'
	@echo 'make queue JOBS=path/to/jobs.json ARGS="--dry-run"'
	@echo 'make status ROOT=/path/to/existing/queue'
	@echo 'make compare CONFIG=path/to/comparison.json OUTPUT=/fresh/report/path'
	@echo 'Python CLIs work independently; training and comparison are explicit actions.'
data:
	@test -n "$(DATASET)" || (echo 'Set DATASET'; exit 2)
	$(MAKE) -C datasets $(DATASET)
verify:
	$(MAKE) -C datasets verify
queue:
	@test -n "$(JOBS)" || (echo 'Set JOBS'; exit 2)
	$(PYTHON) -m mutant_cache.queue_runner run "$(JOBS)" $(ARGS)
resolve:
	$(PYTHON) -m mutant_cache.train --experiment "$(EXP)" $(ARGS) --dry-run
launch:
	$(PYTHON) -m mutant_cache.train --experiment "$(EXP)" $(ARGS)
status:
	@test -n "$(ROOT)" || (echo 'Set ROOT to an existing tuning queue'; exit 2)
	$(PYTHON) -m mutant_cache.tuning report --root "$(ROOT)"
compare:
	@test -n "$(OUTPUT)" || (echo 'Set OUTPUT to a fresh report directory'; exit 2)
	$(PYTHON) -m mutant_cache.matched_suite --config "$(CONFIG)" --output "$(OUTPUT)" $(ARGS)
