#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

: "${HYBRIDARGME_CHECKPOINT_URL:?Set HYBRIDARGME_CHECKPOINT_URL in Render environment variables}"
: "${CONSENSUS_MODEL_URL:?Set CONSENSUS_MODEL_URL in Render environment variables}"
: "${DATA_ARCHIVE_URL:?Set DATA_ARCHIVE_URL in Render environment variables}"

curl_auth=()
if [ -n "${HF_TOKEN:-}" ]; then
  curl_auth=(-H "Authorization: Bearer $HF_TOKEN")
fi

mkdir -p checkpoints data-11

curl --fail --location --retry 3 "${curl_auth[@]}" "$HYBRIDARGME_CHECKPOINT_URL" \
  --output checkpoints/pretrained_esm_v4.pt
curl --fail --location --retry 3 "${curl_auth[@]}" "$CONSENSUS_MODEL_URL" \
  --output checkpoints/consensus_agent_meta_learner.pkl

rm -rf data-11.tmp
tmp_dir="$(mktemp -d data-11.tmp.XXXXXX)"
curl --fail --location --retry 3 "${curl_auth[@]}" "$DATA_ARCHIVE_URL" \
  --output "$tmp_dir/data-11.zip"
unzip -q "$tmp_dir/data-11.zip" -d "$tmp_dir/extracted"

if [ -d "$tmp_dir/extracted/data-11" ]; then
  rm -rf data-11
  mv "$tmp_dir/extracted/data-11" data-11
else
  rm -rf data-11
  mv "$tmp_dir/extracted" data-11
fi

rm -rf "$tmp_dir"
test -s checkpoints/pretrained_esm_v4.pt
test -s checkpoints/consensus_agent_meta_learner.pkl
test -d data-11
printf 'Render assets ready: checkpoint, consensus model, and data-11 directory.\n'
