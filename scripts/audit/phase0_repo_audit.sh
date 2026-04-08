#!/usr/bin/env bash
# phase0_repo_audit.sh — Audit the smart-meter-analysis repo and write /tmp/repo_audit.md
# Designed to run on EC2 where ripgrep is NOT installed. Uses grep -rn throughout.
set -euo pipefail

REPO="/ebs/home/griffin_switch_box/smart-meter-analysis"
OUT="/tmp/repo_audit.md"
RUNS_ROOT="/ebs/home/griffin_switch_box/runs"

# ── helpers ──────────────────────────────────────────────────────────────────
section()  { printf '\n## %s\n\n' "$1" >> "$OUT"; }
subsection() { printf '\n### %s\n\n' "$1" >> "$OUT"; }
fence()    { echo '```' >> "$OUT"; }
fenced()   { fence; cat >> "$OUT"; fence; }

grep_patterns() {
    # $1 = label, $2 = extended-regex pattern
    subsection "$1"
    grep -rn --include='*.py' --include='*.sh' -E "$2" "$REPO" 2>/dev/null \
        | sed "s|$REPO/||" \
        | fenced \
        || echo '_No matches._' >> "$OUT"
}

# ── start report ─────────────────────────────────────────────────────────────
cat > "$OUT" <<'HEADER'
# Smart-Meter-Analysis — Repo Audit (Phase 0)

HEADER
printf 'Generated: %s\n\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$OUT"

# ── 1. Git state ─────────────────────────────────────────────────────────────
section "Git State"
(
    cd "$REPO"
    echo "HEAD: $(git rev-parse --short HEAD)"
    echo "Branch: $(git branch --show-current)"
    echo ""
    echo "Last commit:"
    git log -1 --format='  %h %s (%ai)'
    echo ""
    echo "Status:"
    git status --short || echo '(clean)'
) | fenced

# ── 2. Repo tree (depth 2) ──────────────────────────────────────────────────
section "Repo Tree (depth 2)"
(cd "$REPO" && find . -maxdepth 2 -not -path './.git/*' -not -path './.git' | sort) | fenced

# ── 3. Directory listings ────────────────────────────────────────────────────
section "Directory Listings"

for dir in scripts scripts/csv_to_parquet config; do
    subsection "$dir/"
    if [ -d "$REPO/$dir" ]; then
        ls -la "$REPO/$dir" | fenced
    else
        echo "_Directory does not exist._" >> "$OUT"
    fi
done

# ── 4. Justfile ──────────────────────────────────────────────────────────────
section "Justfile"
if [ -f "$REPO/Justfile" ]; then
    cat "$REPO/Justfile" | fenced
else
    echo '_No Justfile found._' >> "$OUT"
fi

# ── 5. Pattern searches ─────────────────────────────────────────────────────
section "Pattern Searches"

grep_patterns "S3 Input List Generation" \
    's3_paths|s3_list|input_list|aws s3 ls'

grep_patterns "Wide→Long Conversion" \
    'wide.*long|melt|unpivot|pivot'

grep_patterns "Parquet Writing" \
    'write_parquet|sink_parquet|ParquetWriter'

grep_patterns "Compaction" \
    'compact|merge.*parquet|k.way'

grep_patterns "Validation" \
    'validate|check_sort|check_dup|schema_check|iter_batches'

grep_patterns "Overwrite / Resume Flags" \
    'overwrite|resume|--force|_SUCCESS|state_file'

# ── 6. Production output inventory ──────────────────────────────────────────
section "Production Output Inventory"

prod_dirs=()
while IFS= read -r d; do
    prod_dirs+=("$d")
done < <(find "$RUNS_ROOT" -maxdepth 1 -type d -name 'out_*_production' 2>/dev/null | sort)

if [ ${#prod_dirs[@]} -eq 0 ]; then
    echo '_No out\_\*\_production directories found._' >> "$OUT"
else
    echo "Count: ${#prod_dirs[@]}" >> "$OUT"
    echo "" >> "$OUT"

    subsection "Directory List"
    printf '%s\n' "${prod_dirs[@]}" | sed "s|$RUNS_ROOT/||" | fenced

    subsection "Per-Month File Counts and Sizes"
    {
        printf '%-40s  %8s  %s\n' "MONTH (YYYY/MM)" "FILES" "SIZE"
        printf '%-40s  %8s  %s\n' "---" "---" "---"
        for prod_dir in "${prod_dirs[@]}"; do
            base=$(basename "$prod_dir")
            # Walk plain YYYY/MM directories (not Hive-style)
            find "$prod_dir" -mindepth 2 -maxdepth 2 -type d 2>/dev/null \
                | grep -E '/[0-9]{4}/[0-9]{2}$' | sort | while read -r mdir; do
                rel=$(echo "$mdir" | grep -oE '[0-9]{4}/[0-9]{2}$')
                nfiles=$(find "$mdir" -maxdepth 1 -name 'part-*.parquet' -type f 2>/dev/null | wc -l)
                sz=$(du -sh "$mdir" 2>/dev/null | cut -f1)
                printf '%-40s  %8d  %s\n' "$base/$rel" "$nfiles" "$sz"
            done
        done
    } | fenced
fi

# ── 7. s3_paths_*.txt inventory ──────────────────────────────────────────────
section "s3_paths_*.txt Files"

for search_dir in /home/griffin_switch_box /ebs/home/griffin_switch_box; do
    subsection "$search_dir"
    if [ -d "$search_dir" ]; then
        found=$(find "$search_dir" -maxdepth 1 -name 's3_paths_*.txt' 2>/dev/null | sort)
        if [ -n "$found" ]; then
            ls -lh $found | fenced
        else
            echo '_None found._' >> "$OUT"
        fi
    else
        echo "_Directory does not exist._" >> "$OUT"
    fi
done

# ── done ─────────────────────────────────────────────────────────────────────
echo "" >> "$OUT"
echo "---" >> "$OUT"
echo "_End of audit._" >> "$OUT"

echo "Audit written to $OUT ($(wc -l < "$OUT") lines)"
