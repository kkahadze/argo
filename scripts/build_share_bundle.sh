#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bundle_root="${1:-share/argo-share}"
bundle_parent="$(dirname "$bundle_root")"
bundle_name="$(basename "$bundle_root")"
private_data_source="${ARGO_PRIVATE_DATA_DIR:-}"

runtime_files=(
  "sentence_pairs.tsv"
  "gal.tsv"
  "kk.tsv"
  "context_source.txt"
  "harris.txt"
  "harris_compact.txt"
)

optional_runtime_files=(
  "master-lexicon-mkhedruli.csv"
  "translation_overrides.tsv"
)

choose_private_data_source() {
  if [[ -n "$private_data_source" ]]; then
    echo "$private_data_source"
    return
  fi

  for candidate in "private_data" "fastapi_app/data"; do
    for file in "${runtime_files[@]}"; do
      if [[ -f "$candidate/$file" ]]; then
        echo "$candidate"
        return
      fi
    done
  done
}

copy_file_if_present() {
  local source_path="$1"
  local target_path="$2"

  if [[ -f "$source_path" ]]; then
    mkdir -p "$(dirname "$target_path")"
    cp "$source_path" "$target_path"
    return 0
  fi

  return 1
}

data_source="$(choose_private_data_source)"
if [[ -z "$data_source" || ! -d "$data_source" ]]; then
  echo "Could not find private data. Put files in private_data/ or set ARGO_PRIVATE_DATA_DIR." >&2
  exit 1
fi

missing=()
for file in "${runtime_files[@]}"; do
  if [[ ! -f "$data_source/$file" ]]; then
    missing+=("$file")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Private data folder is missing required files:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

rm -rf "$bundle_root" "$bundle_root.zip"
mkdir -p "$bundle_parent"

git archive --format=tar --prefix="$bundle_name/argo/" HEAD | tar -x -C "$bundle_parent"

target_data_dir="$bundle_root/argo/private_data"
mkdir -p "$target_data_dir"
for file in "${runtime_files[@]}"; do
  cp "$data_source/$file" "$target_data_dir/$file"
done

for file in "${optional_runtime_files[@]}"; do
  copy_file_if_present "$data_source/$file" "$target_data_dir/$file" >/dev/null || true
done

copy_file_if_present \
  "private_data/eval-datasets/notion-mingrelian-lesson-notes-triples.csv" \
  "$bundle_root/argo/private_data/eval-datasets/notion-mingrelian-lesson-notes-triples.csv" >/dev/null || true

copy_file_if_present \
  "eval/datasets/notion-mingrelian-lesson-notes-triples.csv" \
  "$bundle_root/argo/private_data/eval-datasets/notion-mingrelian-lesson-notes-triples.csv" >/dev/null || true

copy_file_if_present \
  "$bundle_root/argo/private_data/eval-datasets/notion-mingrelian-lesson-notes-triples.csv" \
  "$bundle_root/argo/eval/datasets/notion-mingrelian-lesson-notes-triples.csv" >/dev/null || true

cat > "$bundle_root/README.md" <<'EOF'
# Argo Share Bundle

This bundle includes the Argo backend code plus private runtime data in
`argo/private_data/`.

## Run

```bash
cd argo
bash run_local.sh
```

Add API keys to `argo/.env` if you want the backend to use server-side LLM
credentials. Browser clients can also send their own provider key per request.
EOF

if command -v zip >/dev/null 2>&1; then
  (
    cd "$bundle_parent"
    zip -qr "$bundle_name.zip" "$bundle_name"
  )
  echo "Created $bundle_root and $bundle_root.zip"
else
  echo "Created $bundle_root"
  echo "zip was not found; zip the folder manually if you want one archive."
fi
