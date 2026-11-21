#!/usr/bin/env bash
set -euo pipefail
S2E_SIMPLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$S2E_SIMPLE/../../environment/activate.sh"
cd "$S2E_SIMPLE"
exec ./x86_64/special -nobanner -nogui -python opt_inh_forward.py --setting all_hh --mode verify_counts --hh-table-off --run-id "$1" --counts-file "$2" --output-root "$3"
