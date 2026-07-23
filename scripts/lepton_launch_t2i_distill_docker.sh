cd /home/xingqianx/Project/imaginaire4_distill/packages/cosmos-model-arena
uv run --extra deploy python -m cosmos_model_arena.scripts.cosmos3_admin build-image \
    --ref xingqianx/t2i_distill_aa_leaderboard \
    --image-tag xingqianx-distill \
    --push
