# Raw dataset downloads. Everything lands in $(SCRATCH)/raw/<dataset>/, never in the repo.
#
#   make -C datasets tsuboyama
#   make -C datasets all
#   make -C datasets check            # verify recorded checksums
#
# Every download uses curl -C - so an interrupted pull resumes.
# Checksums live in checksums.md5 next to this file; they are the md5s Zenodo
# reports for its own files, and the ones we computed for the GitHub-hosted
# files (GitHub publishes none).

SCRATCH ?= $(if $(MUTANT_CACHE_SCRATCH),$(MUTANT_CACHE_SCRATCH),$(HOME)/mutant_cache_scratch)
RAW     := $(SCRATCH)/raw
CURL    := curl -L -C - --retry 5 --retry-delay 5 --fail-with-body

.PHONY: all tsuboyama tsuboyama-raw-ngs mavedb flip_gb1 proteingym combingym check clean-partial dirs

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
