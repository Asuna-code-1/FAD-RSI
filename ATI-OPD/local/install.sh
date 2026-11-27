#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/data/fd/donghaosen/opd-shared/cache/uv
export TMPDIR=/data/fd/donghaosen/ati-opd/tmp
export UV_LINK_MODE=copy
if [[ ! -x /workspace/fudan/donghaosen/ATI-OPD/.venv/bin/python ]]; then
    uv venv --python /usr/bin/python3.12 /workspace/fudan/donghaosen/ATI-OPD/.venv
fi
uv pip install --python /workspace/fudan/donghaosen/ATI-OPD/.venv/bin/python -r /workspace/fudan/donghaosen/ATI-OPD/local/environment-freeze.txt
uv pip check --python /workspace/fudan/donghaosen/ATI-OPD/.venv/bin/python
