# Raw dataset downloads and processed stability benchmarks. Everything lands in
# $(SCRATCH) (default ~/mutant_cache_scratch), never in the repo. See README.md.
#
#   make -C datasets tsuboyama            # raw downloads -> $(SCRATCH)/raw/<dataset>/
#   make -C datasets all
#   make -C datasets check                # verify recorded checksums
#   make -C datasets mutate-everything    # -> processed/mutate_everything_cdna (+ holdout1/2 splits)
#                                         #    + mutate_everything/external
#   make -C datasets esm-msr              # -> processed/esm_msr_mega
#   make -C datasets verify               # content hashes vs processed_hashes.json
#   make -C datasets stabilityarc         # planned only: fails, listing missing artifacts
#
# Raw downloads use curl -C - so an interrupted pull resumes.
# Checksums live in checksums.md5 next to this file; they are the md5s Zenodo
# reports for its own files, and the ones we computed for the GitHub-hosted
# files (GitHub publishes none).
#
# Processed targets are file targets: existing outputs are up to date and are
# never rebuilt silently. If a processed output is missing or stale but its
# directory already holds files, the rule refuses to write unless FORCE=1
# (live runs read these directories; stop them first). Preview with make -n.

SCRATCH ?= $(if $(MUTANT_CACHE_SCRATCH),$(MUTANT_CACHE_SCRATCH),$(HOME)/mutant_cache_scratch)
RAW     := $(SCRATCH)/raw
CURL    := curl -L -C - --retry 5 --retry-delay 5 --fail-with-body

.PHONY: all tsuboyama tsuboyama-raw-ngs mavedb flip_gb1 proteingym combingym check record clean-partial dirs \
        mutate-everything mutate-everything-raw mutate-everything-external mutate-everything-holdouts \
        mutate-everything-check esm-msr esm-msr-raw esm-msr-check stabilityarc verify

all: tsuboyama mavedb flip_gb1 proteingym

# ---------------------------------------------------------------- Tsuboyama
# Zenodo 7844779, "Mega-scale experimental analysis of protein folding
# stability", version v2_230420 (checked 2026-10-16: still the newest version).
TSU := $(RAW)/tsuboyama
TSU_URL := https://zenodo.org/api/records/7844779/files

tsuboyama: $(TSU)/Processed_K50_dG_datasets.zip $(TSU)/K50_dG_tables.zip

$(TSU)/Processed_K50_dG_datasets.zip:
	mkdir -p $(TSU)
	$(CURL) -o $@ "$(TSU_URL)/Processed_K50_dG_datasets.zip/content"

$(TSU)/K50_dG_tables.zip:
	mkdir -p $(TSU)
	$(CURL) -o $@ "$(TSU_URL)/K50_dG_tables.zip/content"

# Only if you want your own noise model. 243 MB.
tsuboyama-raw-ngs: $(TSU)/Raw_NGS_count_tables.zip
$(TSU)/Raw_NGS_count_tables.zip:
	mkdir -p $(TSU)
	$(CURL) -o $@ "$(TSU_URL)/Raw_NGS_count_tables.zip/content"

# ------------------------------------------------------------------- MaveDB
# Zenodo 14172004, "MaveDB Bulk Download", version 2 (2024-11-14). 1.4 GB.
MAVE := $(RAW)/mavedb

mavedb: $(MAVE)/mavedb-dump.20241114101443.zip

$(MAVE)/mavedb-dump.20241114101443.zip:
	mkdir -p $(MAVE)
	$(CURL) -o $@ "https://zenodo.org/api/records/14172004/files/mavedb-dump.20241114101443.zip/content"

# ------------------------------------------------------------------ FLIP GB1
# NOTE: DATASETS.md points at data.bioembeddings.com/public/FLIP for the raw
# data. That host 404s for every path we tried (2026-10-16). The GB1 data and
# splits are committed in the FLIP repo itself, so we pull them from there.
FLIP := $(RAW)/flip_gb1
FLIP_URL := https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1

flip_gb1: $(FLIP)/four_mutations_full_data.csv.zip $(FLIP)/splits.zip $(FLIP)/5LDE_1.fasta $(FLIP)/README.md

$(FLIP)/%:
	mkdir -p $(FLIP)
	$(CURL) -o $@ "$(FLIP_URL)/$*"

# --------------------------------------------------------------- ProteinGym
# Reference file only (the full substitution benchmark is 5.9 GB on Zenodo and
# is not needed to build the exclusion list).
PG := $(RAW)/proteingym

proteingym: $(PG)/DMS_substitutions.csv

$(PG)/DMS_substitutions.csv:
	mkdir -p $(PG)
	$(CURL) -o $@ "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"

# ---------------------------------------------------------------- CombinGym
# Repo clone, not a file download; kept here so the path is recorded.
CG := $(RAW)/combingym

combingym: $(CG)/.git

$(CG)/.git:
	mkdir -p $(RAW)
	git clone --depth 1 https://github.com/sitonglab/CombinGym $(CG)

# ----------------------------------------------------------------- checksums
check:
	cd $(RAW) && md5sum -c $(CURDIR)/checksums.md5

record:
	cd $(RAW) && md5sum \
	  tsuboyama/Processed_K50_dG_datasets.zip \
	  tsuboyama/K50_dG_tables.zip \
	  mavedb/mavedb-dump.20241114101443.zip \
	  flip_gb1/four_mutations_full_data.csv.zip \
	  flip_gb1/splits.zip \
	  flip_gb1/5LDE_1.fasta \
	  proteingym/DMS_substitutions.csv > $(CURDIR)/checksums.md5

# Drop zero-length / obviously truncated files so a resume starts clean.
clean-partial:
	find $(RAW) -size 0 -name '*.zip' -print -delete

# ======================================================= processed benchmarks
# Every path below derives from $(SCRATCH); nothing assumes a home directory.
# The preparation code is mutant_cache.data.{mutate_everything,mutate_everything_raw,esm_msr}
# (the raw scripts in experiments/ that first built these dirs are kept as run
# and are not called from here).
PYTHON    ?= $(firstword $(wildcard $(HOME)/anaconda3/envs/mutant_cache/bin/python) python)
REPO      := $(abspath $(CURDIR)/..)
PROCESSED := $(SCRATCH)/processed
FORCE     ?= 0
RUNPY      = cd $(REPO) && MUTANT_CACHE_SCRATCH=$(SCRATCH) PYTHONPATH=$(REPO) $(PYTHON)

# $(call guard_dir,DIR): refuse to write into a non-empty DIR unless FORCE=1.
define guard_dir
	@if [ -n "$$(ls -A '$(1)' 2>/dev/null)" ] && [ "$(FORCE)" != 1 ]; then \
	  echo "REFUSING to (re)build into non-empty $(1)" >&2; \
	  echo "  Live runs may read it. Inspect it, stop dependent jobs, then rerun with FORCE=1." >&2; \
	  exit 1; fi
endef

# --------------------------------------------------------- Mutate Everything
# Upstream github.com/jozhang97/MutateEverything @ 034dd91e7a449e68b0f0993043bd32ce281f056c,
# data archive = Google Drive id 1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y (upstream README:
# `gdown 1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y`; fetched 2026-11-04 from
# https://drive.usercontent.google.com/download?id=<id>&export=download&confirm=t).
# Processed: dataset mutate_everything_cdna, split mutate_everything_cdna
# (104 train / 12 val proteins from the released cDNA train set, seed 0, 10%
# of sequence groups; released cDNA2 test = 18 proteins), plus the seed 1/2
# development re-splits mutate_everything_cdna_holdout{1,2}.
ME_ROOT   := $(SCRATCH)/mutate_everything
ME_ZIP    := $(ME_ROOT)/mutate_everything_data.zip
ME_ZIP_SHA256   := 4e4a579902db8308196f4c1bf0fed25ee31b1573c28613525ff2f3def2350f9f
ME_TRAIN_SHA256 := 1d1c901f940a207872f0c554237cb24c077118dad81c55ca6b979bc7d401bc5b
ME_TEST_SHA256  := e82b937c0d9f2d05c01951a95221b5494086b13aa341e0e31237c5d3c8afdb25
ME_TRAIN  := $(ME_ROOT)/data/cdna_train.csv
ME_TEST   := $(ME_ROOT)/data/cdna2_test.csv
ME_OUT    := $(PROCESSED)/mutate_everything_cdna
ME_EXT    := $(ME_ROOT)/external
ME_HOLDOUTS := $(ME_OUT)/splits/mutate_everything_cdna_holdout1.parquet \
               $(ME_OUT)/splits/mutate_everything_cdna_holdout2.parquet

mutate-everything: $(ME_OUT)/manifest.json $(ME_EXT)/inventory.json $(ME_HOLDOUTS)
mutate-everything-raw: $(ME_ZIP) $(ME_TRAIN) $(ME_TEST)
mutate-everything-external: $(ME_EXT)/inventory.json
mutate-everything-holdouts: $(ME_HOLDOUTS)

# Downloads only if the archive is absent (Google Drive; may need gdown or a
# browser if Drive starts serving an interstitial, see README).
$(ME_ZIP):
	$(RUNPY) -m mutant_cache.data.mutate_everything_raw fetch
	cd $(ME_ROOT) && echo "$(ME_ZIP_SHA256)  mutate_everything_data.zip" | sha256sum -c

mutate-everything-check: $(ME_ZIP)
	cd $(ME_ROOT) && echo "$(ME_ZIP_SHA256)  mutate_everything_data.zip" | sha256sum -c

# sha256-checks the archive, then extracts the two cDNA tables. Order-only on
# the archive: extracted tables must not be re-extracted because of its mtime.
$(ME_TRAIN) $(ME_TEST) &: | $(ME_ZIP)
	$(RUNPY) -m mutant_cache.data.mutate_everything_raw extract
	cd $(ME_ROOT)/data && printf '%s  %s\n%s  %s\n' \
	  $(ME_TRAIN_SHA256) cdna_train.csv $(ME_TEST_SHA256) cdna2_test.csv | sha256sum -c

# Canonical tables, protein split, exclusions, comparison label CSVs, manifest.
# The ESM-MSR processed dir is only an audit input (overlap + sign check), so
# it is order-only: rebuilding ESM-MSR does not cascade into this live dir.
# NOT rebuilt here (no recorded generating script, see README.md):
# sign_and_overlap_audit.json, largest_external_label_differences.csv.
$(ME_OUT)/manifest.json: $(ME_TRAIN) $(ME_TEST) | $(PROCESSED)/esm_msr_mega/splits/esm_msr_hyperopt.parquet
	$(call guard_dir,$(ME_OUT))
	$(RUNPY) -m mutant_cache.data.mutate_everything \
	  --train-csv $(ME_TRAIN) --test-csv $(ME_TEST) \
	  --audit-dataset-dir $(PROCESSED)/esm_msr_mega

# Seed 1/2 re-splits of the development proteins (test unchanged). Existing
# files are compared with the recomputed split and left alone if identical
# (the step fails if they differ); FORCE=1 rewrites them.
$(ME_HOLDOUTS) &: $(ME_OUT)/manifest.json
	$(RUNPY) -m mutant_cache.data.mutate_everything_raw holdouts $(if $(filter 1,$(FORCE)),--overwrite)

# External panels (S669, ProTherm Multiple, cDNA1, S669 reverse) labels.
$(ME_EXT)/inventory.json: $(ME_ZIP) $(ME_OUT)/manifest.json
	$(call guard_dir,$(ME_EXT))
	$(RUNPY) -m mutant_cache.data.mutate_everything_raw external

# -------------------------------------------------------------------- ESM-MSR
# Upstream github.com/SKTeamLab/esm-msr @ c34c315913b539070095b8af63b4bd7fbcd503b7,
# data Zenodo 10.5281/zenodo.21539277 (v0.1) preprocessed_data.tar.gz.
# Processed: dataset esm_msr_mega, split esm_msr_hyperopt (released hyperopt
# train/val/test protein assignment).
MSR_ROOT     := $(SCRATCH)/esm_msr
MSR_REPO_URL ?= https://github.com/SKTeamLab/esm-msr.git
MSR_COMMIT   := c34c315913b539070095b8af63b4bd7fbcd503b7
MSR_TAR      := $(MSR_ROOT)/preprocessed_data.tar.gz
MSR_TAR_MD5  := 56ae9af3f800065b7a7ff4f4a42b7b92
MSR_TAR_SHA256 := 9c36e94df4ba6c2f035e2c395301ed98f288f6bd95abc20a05c67e97dce22086
MSR_CSV      := $(MSR_ROOT)/preprocessed/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv
MSR_CSV_MD5  := 5810ab153d292af2c71cdf9af572c040
MSR_SPLITS   := $(MSR_ROOT)/repo/data/hyperopt_splits.csv
MSR_SPLITS_SHA256 := e3fccc655fd3ec1fa808da22f85f853ad688486b96582f8ddd0b1eb2ada0959d
MSR_OUT      := $(PROCESSED)/esm_msr_mega

esm-msr: $(MSR_OUT)/splits/esm_msr_hyperopt.parquet $(MSR_OUT)/audit/homology_ge30.csv
esm-msr-raw: $(MSR_CSV) $(MSR_SPLITS)

# The original fetch command was not recorded, and the guessed Zenodo API URL
# (records/21539277/files/preprocessed_data.tar.gz/content) returned 404 on
# 2026-11-05. Fetch it manually from the DOI record; these rules only verify
# and extract it.
define msr_tar_missing
@echo "Missing $(MSR_TAR). Download preprocessed_data.tar.gz from https://doi.org/10.5281/zenodo.21539277" >&2; \
	echo "into $(MSR_ROOT)/ and rerun (expected md5 $(MSR_TAR_MD5))." >&2; exit 1
endef

$(MSR_TAR):
	$(msr_tar_missing)

esm-msr-check: $(MSR_TAR)
	echo "$(MSR_TAR_MD5)  $(MSR_TAR)" | md5sum -c
	echo "$(MSR_TAR_SHA256)  $(MSR_TAR)" | sha256sum -c

# No prerequisite on the archive: its members keep 2023 mtimes (older than the
# archive), and an existing CSV must not require the archive to be kept.
$(MSR_CSV):
	@test -f $(MSR_TAR) || { echo "Missing $(MSR_TAR); download it from https://doi.org/10.5281/zenodo.21539277" >&2; exit 1; }
	echo "$(MSR_TAR_MD5)  $(MSR_TAR)" | md5sum -c
	mkdir -p $(MSR_ROOT)/preprocessed
	tar -xzf $(MSR_TAR) -C $(MSR_ROOT)/preprocessed
	echo "$(MSR_CSV_MD5)  $(MSR_CSV)" | md5sum -c

# MSR_REPO_URL can point at an existing clone (file://...) to avoid the network.
$(MSR_SPLITS):
	git clone $(MSR_REPO_URL) $(MSR_ROOT)/repo
	git -C $(MSR_ROOT)/repo -c advice.detachedHead=false checkout $(MSR_COMMIT)
	echo "$(MSR_SPLITS_SHA256)  $(MSR_SPLITS)" | sha256sum -c

# One invocation writes tables, the split and audit/ (homology audit last).
$(MSR_OUT)/splits/esm_msr_hyperopt.parquet $(MSR_OUT)/audit/homology_ge30.csv &: $(MSR_CSV) $(MSR_SPLITS)
	$(call guard_dir,$(MSR_OUT))
	$(RUNPY) -m mutant_cache.data.esm_msr

# --------------------------------------------------------------- verification
# Content hashes of every processed table/JSON (and the raw inputs) against
# datasets/processed_hashes.json; see mutant_cache/data/verify_processed.py.
# Restrict with VERIFY_GROUPS="esm-msr mutate-everything ...".
VERIFY_GROUPS ?=
verify:
	$(RUNPY) -m mutant_cache.data.verify_processed $(addprefix --group ,$(VERIFY_GROUPS))

# --------------------------------------------------------------- StabilityArc
# Planned only. No importer, release pin or protocol artifacts exist yet, so
# this target fails (also under make -n, via +) instead of building anything.
stabilityarc:
	+@echo "stabilityarc: PLANNED, nothing to build (datasets/protocols/stabilityarc_v1.json status=planned)." >&2; \
	echo "Still needed before this target can exist:" >&2; \
	echo "  1. pinned source release (upstream code commit + data archive version/checksum)" >&2; \
	echo "  2. importer -> \$$SCRATCH/processed/<dataset> with manifest/provenance" >&2; \
	echo "  3. released outer protein holdout memberships, materialized as split parquet(s)" >&2; \
	echo "  4. development-only model-selection partition/protocol (no outer-holdout access)" >&2; \
	echo "  5. overlap and label audit (sequence/homology overlap, sign/units vs other sets)" >&2; \
	echo "  6. reproduction of upstream StabilityArc evaluation on the identical rows" >&2; \
	exit 1
