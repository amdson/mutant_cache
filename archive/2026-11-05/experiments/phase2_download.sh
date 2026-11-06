#!/bin/bash
# Phase 2 raw downloads, exactly as run on 2026-10-16.
# datasets/Makefile is the maintained version of this; this file is the record
# of what actually happened, including the URL that turned out to be dead.

mkdir -p /home/andrew/mutant_cache_scratch/raw/tsuboyama /home/andrew/mutant_cache_scratch/logs
cd /home/andrew/mutant_cache_scratch/raw/tsuboyama
nohup bash -c '
curl -L -C - -o Processed_K50_dG_datasets.zip "https://zenodo.org/api/records/7844779/files/Processed_K50_dG_datasets.zip/content"
curl -L -C - -o K50_dG_tables.zip "https://zenodo.org/api/records/7844779/files/K50_dG_tables.zip/content"
echo TSUBOYAMA_DONE
' > /home/andrew/mutant_cache_scratch/logs/tsuboyama_dl.log 2>&1 &

mkdir -p /home/andrew/mutant_cache_scratch/raw/mavedb
cd /home/andrew/mutant_cache_scratch/raw/mavedb
nohup curl -L -C - -o mavedb-dump.20241114101443.zip \
  "https://zenodo.org/api/records/14172004/files/mavedb-dump.20241114101443.zip/content" \
  > /home/andrew/mutant_cache_scratch/logs/mavedb_dl.log 2>&1 &

# DATASETS.md says the FLIP raw data lives at data.bioembeddings.com/public/FLIP.
# It does not any more -- both of these 404:
#   curl -sIL -o /dev/null -w "%{http_code}\n" http://data.bioembeddings.com/public/FLIP/fasta/gb1/gb1.fasta
#   curl -sIL -o /dev/null -w "%{http_code}\n" http://data.bioembeddings.com/public/FLIP/all_fastas.zip
# The GB1 data and splits are committed in the FLIP repo, so pull them there.
mkdir -p /home/andrew/mutant_cache_scratch/raw/flip_gb1 /home/andrew/mutant_cache_scratch/raw/proteingym
cd /home/andrew/mutant_cache_scratch/raw/flip_gb1
for f in 5LDE_1.fasta README.md four_mutations_full_data.csv.zip splits.zip; do
  curl -sL -C - -o "$f" "https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/$f"
done

curl -sL -C - -o /home/andrew/mutant_cache_scratch/raw/proteingym/DMS_substitutions.csv \
  "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"

# md5s, all matching what Zenodo reports:
#   27a0936c8f80e5b18ed330ed6b98a3a6  tsuboyama/Processed_K50_dG_datasets.zip
#   1cd1e460db6aa907e17c1249b35fbb82  tsuboyama/K50_dG_tables.zip
#   073dc45b3f0e57d7573c7d64414442cd  mavedb/mavedb-dump.20241114101443.zip
