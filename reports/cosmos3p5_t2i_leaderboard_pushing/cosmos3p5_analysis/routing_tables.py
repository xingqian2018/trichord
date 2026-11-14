"""Print the markdown tables of the generation-tower MoE routing report from calibrate_gen_moe_routing.py output.

Usage:
    python routing_tables.py gen_moe_routing_stats_iter741260_mha32_ugb1170.pt [gen_moe_routing_stats_iter741260_mha32_ugb10.pt]
"""

import math
import sys

import torch

COLUMNS = [
    "Layer",
    "Eff. experts",
    "Eff. experts (10 prompts)",
    "Top-8 load share",
    "Experts for 50% load",
    "Experts < 0.1x mean",
    "Experts > 2x mean",
    "Mean token entropy",
    "Prob-load corr",
    "Per-image JS",
    "Per-image top-8 overlap",
    "Per-image eff. experts",
]
ROW_FORMAT = "| %d | %.1f | %.1f | %.3f | %d | %d | %d | %.2f | %+.2f | %.3f | %.0f | %.1f |"


def effective_experts(load: torch.Tensor) -> float:
    f = load / load.sum()
    return math.exp(-(f * torch.log(f.clamp_min(1e-12))).sum().item())


def js_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (a + b)

    def kl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x * (torch.log(x.clamp_min(1e-12)) - torch.log(y.clamp_min(1e-12)))).sum(-1)

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def main() -> None:
    stats = torch.load(sys.argv[1], weights_only=True)
    small = torch.load(sys.argv[2], weights_only=True)["layers"] if len(sys.argv) > 2 else None
    layers = stats["layers"]
    per_prompt = stats["per_prompt"]
    layer_ids = stats["per_prompt_layer_ids"]
    num_experts = layers[layer_ids[0]]["mass"].numel()

    mass = torch.stack([per_prompt[p]["mass"] for p in per_prompt])  # [N,L,E]
    dist = mass / mass.sum(-1, keepdim=True)
    aggregate = dist.mean(0)  # [L,E]
    jsd = js_divergence(dist, aggregate.unsqueeze(0))  # [N,L]
    top8_image = dist.topk(8, -1).indices  # [N,L,8]
    top8_aggregate = aggregate.topk(8, -1).indices  # [L,8]
    overlap = (top8_image.unsqueeze(-1) == top8_aggregate.unsqueeze(0).unsqueeze(-2)).any(-1).sum(-1).float()  # [N,L]
    eff_image = torch.exp(-(dist * torch.log(dist.clamp_min(1e-12))).sum(-1))  # [N,L]

    print(
        f"prompts={len(per_prompt)} layers={len(layers)} experts={num_experts} "
        f"tokens_per_layer={layers[layer_ids[0]]['num_tokens']} ln(E)={math.log(num_experts):.3f}\n"
    )
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "|".join(["---"] * len(COLUMNS)) + "|")
    summary = []
    for i, layer in enumerate(layer_ids):
        load = layers[layer]["mass"]
        share = load / load.sum()
        sorted_share, _ = share.sort(descending=True)
        ratio = load / load.mean()
        prob = layers[layer]["router_prob_sum"] / layers[layer]["router_prob_sum"].sum()
        row = (
            layer,
            effective_experts(load),
            effective_experts(small[layer]["mass"]) if small is not None else float("nan"),
            sorted_share[:8].sum().item(),
            int((sorted_share.cumsum(0) < 0.5).sum().item()) + 1,
            int((ratio < 0.1).sum().item()),
            int((ratio > 2).sum().item()),
            (layers[layer]["sum_token_entropy"] / layers[layer]["num_tokens"]).item(),
            torch.corrcoef(torch.stack([share, prob]))[0, 1].item(),
            jsd[:, i].median().item(),
            overlap[:, i].median().item(),
            eff_image[:, i].median().item(),
        )
        summary.append(row)
        print(ROW_FORMAT % row)

    if small is not None:
        effs = torch.tensor([[r[1] for r in summary], [r[2] for r in summary]])
        print(f"\ncorr(eff. experts, 1170 vs 10 prompts) across layers = {torch.corrcoef(effs)[0, 1]:.3f}")
    entropies = [r[7] for r in summary]
    print(f"mean token entropy over layers = {sum(entropies) / len(entropies):.3f}, min = {min(entropies):.3f}")


if __name__ == "__main__":
    main()
