# Source only. No model loading or service startup.
fad_rsi_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$fad_rsi_root/ATI-OPD/local/activate.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online
export WANDB_DIR=/data/fd/donghaosen/fad-rsi/wandb
export HF_HUB_OFFLINE=1
