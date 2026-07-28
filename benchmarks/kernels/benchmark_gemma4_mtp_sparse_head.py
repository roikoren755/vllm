# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Gemma4 MTP centroid-masked sparse LM head.

Compares three implementations of the same argmax:

- ``gather``: scattered row gather + einsum (current implementation)
- ``moe``:    centroid-major weight layout + fused_moe grouped GEMM
- ``dense``:  plain full-vocab GEMM, as the crossover reference

Also reports argmax parity between the sparse backends, and between each
sparse backend and dense (which is expected to disagree — dense sees the
whole vocabulary, the sparse paths only the selected centroids).

Usage:
    python benchmarks/kernels/benchmark_gemma4_mtp_sparse_head.py \
        --hidden-size 256 --vocab-size 262144 \
        --num-centroids 2048 --top-k 32
"""

import torch

from vllm.model_executor.models.gemma4_mtp import _decode_top_token, _grouped_gemm
from vllm.triton_utils import tl, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser


@triton.jit
def _topk_indices_kernel(
    scores_ptr,
    out_ptr,
    num_experts,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Top-k indices only, one program per token, iterative argmax-and-mask.

    torch.topk runs a general multi-pass radix select; here k is small, the
    row fits in registers, and the values are discarded, so repeated argmax
    over a masked vector may win. Emits descending-value order, matching
    torch.topk(sorted=True).
    """
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    vals = tl.load(
        scores_ptr + t * num_experts + offs,
        mask=offs < num_experts,
        other=-float("inf"),
    )
    for i in tl.static_range(TOP_K):
        best = tl.argmax(vals, axis=0)
        tl.store(out_ptr + t * TOP_K + i, best)
        vals = tl.where(offs == best, -float("inf"), vals)


def _topk_triton(scores, top_k):
    num_tokens, num_experts = scores.shape
    out = torch.empty(num_tokens, top_k, dtype=torch.int32, device=scores.device)
    _topk_indices_kernel[(num_tokens,)](
        scores,
        out,
        num_experts,
        TOP_K=top_k,
        BLOCK=triton.next_power_of_2(num_experts),
    )
    return out


def _topk_indices(scores, top_k, sorted_):
    return torch.topk(scores, k=top_k, dim=-1, sorted=sorted_)[1]


def _fused_topk_indices(fused_topk, hidden_states, scores, top_k):
    return fused_topk(hidden_states, scores, top_k, renormalize=False)[1]


def router_bench(args, x_by_t, centroids_w):
    """Isolate the router: the linear, then each top-k implementation."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )

    print("\nRouter breakdown (ms)\n")
    print(
        f"{'T':>4} {'linear':>8} {'tk_sort':>8} {'tk_nosrt':>9} {'tk_fused':>9} "
        f"{'tk_triton':>10} {'agree':>7}"
    )
    for num_tokens in args.sweep_batch_sizes:
        x = x_by_t[num_tokens]
        t_linear = _bench(torch.nn.functional.linear, x, centroids_w)
        scores = torch.nn.functional.linear(x, centroids_w)

        t_sorted = _bench(_topk_indices, scores, args.top_k, True)
        t_nosort = _bench(_topk_indices, scores, args.top_k, False)
        try:
            t_fused = _bench(_fused_topk_indices, fused_topk, x, scores, args.top_k)
            fused_str = f"{t_fused:>9.4f}"
        except Exception:
            fused_str = f"{'unsup':>9}"

        try:
            ref = torch.topk(scores, k=args.top_k, dim=-1)[1]
            got = _topk_triton(scores, args.top_k)
            agree = torch.equal(got.to(torch.int64), ref)
            t_triton = _bench(_topk_triton, scores, args.top_k)
            triton_str = f"{t_triton:>10.4f}"
        except Exception as e:
            agree = False
            triton_str = f"{str(e)[:10]:>10}"

        print(
            f"{num_tokens:>4} {t_linear:>8.4f} {t_sorted:>8.4f} "
            f"{t_nosort:>9.4f} {fused_str} {triton_str} {str(agree):>7}"
        )


_decode_fused = _decode_top_token


def _select_and_score_gather(
    hidden_states, lm_head_weight, token_ordering, num_centroids, top_k, centroids_w
):
    """Mirrors the original Gemma4MTPMaskedEmbedder._select_and_score."""
    num_tokens = hidden_states.shape[0]
    vocab_per_centroid = token_ordering.numel() // num_centroids
    _, top_k_indices = torch.topk(
        torch.nn.functional.linear(hidden_states, centroids_w), k=top_k, dim=-1
    )
    clusters = token_ordering.view(num_centroids, vocab_per_centroid)
    selected = clusters[top_k_indices]
    embeddings = lm_head_weight[selected.reshape(-1)].view(
        num_tokens, top_k * vocab_per_centroid, hidden_states.shape[-1]
    )
    logits = torch.einsum("td,tsd->ts", hidden_states, embeddings)
    return logits, selected.view(num_tokens, -1)


def _top_tokens_gather(*args):
    logits, indices = _select_and_score_gather(*args)
    return indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)


def _top_tokens_moe(
    hidden_states,
    centroid_weight,
    token_ordering,
    num_centroids,
    top_k,
    centroids_w,
    config_override=None,
):
    """Mirrors Gemma4MTPMaskedEmbedder.get_top_tokens on the moe backend."""
    vocab_per_centroid = token_ordering.numel() // num_centroids
    _, top_k_indices = torch.topk(
        torch.nn.functional.linear(hidden_states, centroids_w),
        k=top_k,
        dim=-1,
        sorted=False,
    )
    top_k_indices = top_k_indices.to(torch.int32)
    logits = _grouped_gemm(
        hidden_states,
        centroid_weight,
        top_k_indices,
        top_k,
        num_centroids,
        config_override,
    )
    return _decode_top_token(
        logits.view(hidden_states.shape[0], -1),
        top_k_indices,
        token_ordering,
        top_k,
        vocab_per_centroid,
    )


def _top_tokens_gather_plus(
    hidden_states, lm_head_weight, token_ordering, num_centroids, top_k, centroids_w
):
    """Original gather+einsum, with the decode fusion and unsorted top-k.

    Isolates the grouped-GEMM contribution: this differs from _top_tokens_moe
    only in how the sparse dot product is computed. The gather path's flattened
    logits are k-major/j-minor, the same layout _decode_top_token assumes, so
    the fused decode drops straight in.
    """
    num_tokens = hidden_states.shape[0]
    vocab_per_centroid = token_ordering.numel() // num_centroids
    _, topk_ids = torch.topk(
        torch.nn.functional.linear(hidden_states, centroids_w),
        k=top_k,
        dim=-1,
        sorted=False,
    )
    topk_ids = topk_ids.to(torch.int32)
    clusters = token_ordering.view(num_centroids, vocab_per_centroid)
    selected = clusters[topk_ids.long()]
    embeddings = lm_head_weight[selected.reshape(-1)].view(
        num_tokens, top_k * vocab_per_centroid, hidden_states.shape[-1]
    )
    logits = torch.einsum("td,tsd->ts", hidden_states, embeddings)
    return _decode_top_token(
        logits, topk_ids, token_ordering, top_k, vocab_per_centroid
    )


# dynamic=True: one graph for all batch sizes. dynamic=False exhausts
# dynamo's recompile_limit (8) partway through a sweep and silently falls
# back to eager, which is slower than the einsum it replaced. Variable
# drafter batch size makes dynamic the production-relevant setting too.
@torch.compile(dynamic=True)
def _sparse_dot_compiled(hidden_states, selected, lm_head_weight):
    """Gather + dot, written so inductor can fuse away the transient.

    einsum lowers to bmm (an extern kernel), which forces the
    (T, num_selected, hidden) intermediate to be materialized. Written as an
    explicit multiply-reduce, inductor can codegen a single Triton reduction
    with the row gather folded in as an indirect load — no transient, and no
    centroid-major weight copy either.

    Whether it actually fuses shows up in the timing: close to gather+ means
    it materialized anyway, close to moe means it fused.
    """
    return (lm_head_weight[selected] * hidden_states[:, None, :]).sum(-1)


def _top_tokens_gather_compiled(
    hidden_states, lm_head_weight, token_ordering, num_centroids, top_k, centroids_w
):
    num_tokens = hidden_states.shape[0]
    vocab_per_centroid = token_ordering.numel() // num_centroids
    _, topk_ids = torch.topk(
        torch.nn.functional.linear(hidden_states, centroids_w),
        k=top_k,
        dim=-1,
        sorted=False,
    )
    topk_ids = topk_ids.to(torch.int32)
    clusters = token_ordering.view(num_centroids, vocab_per_centroid)
    selected = clusters[topk_ids.long()].view(num_tokens, -1)
    logits = _sparse_dot_compiled(hidden_states, selected, lm_head_weight)
    return _decode_top_token(
        logits, topk_ids, token_ordering, top_k, vocab_per_centroid
    )


def _top_tokens_dense(hidden_states, lm_head_weight):
    return torch.nn.functional.linear(hidden_states, lm_head_weight).argmax(-1)


def _bench(fn, *fn_args, **fn_kwargs):
    """Time fn(*fn_args) under CUDA graph replay.

    Arguments are passed through rather than captured in a closure, so call
    sites inside loops don't bind loop variables late (ruff B023).
    """
    ms, *_ = triton.testing.do_bench_cudagraph(
        lambda: fn(*fn_args, **fn_kwargs), quantiles=[0.5, 0.2, 0.8]
    )
    return ms


def sweep_config(args, x_by_t, centroid_weight, token_ordering, centroids_w, lm_head):
    """Block-shape sweep at small T, where the kernel is occupancy-bound.

    Grid is top_k * (N / BLOCK_SIZE_N) blocks, so BLOCK_SIZE_N controls how
    much of the GPU is filled. num_warps matters because the tiles are tiny.
    """
    vocab_per_centroid = args.vocab_size // args.num_centroids
    print("\nBlock-shape sweep (moe backend, ms)\n")
    header = f"{'BLOCK_N':>8} {'warps':>6} {'stages':>7}"
    for num_tokens in args.sweep_batch_sizes:
        header += f" {'T=' + str(num_tokens):>9}"
    print(header)

    for block_n in args.sweep_block_n:
        if block_n > vocab_per_centroid:
            continue
        for num_warps in args.sweep_warps:
            for num_stages in args.sweep_stages:
                config = {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
                row = f"{block_n:>8} {num_warps:>6} {num_stages:>7}"
                for num_tokens in args.sweep_batch_sizes:
                    x = x_by_t[num_tokens]
                    try:
                        ms = _bench(
                            _top_tokens_moe,
                            x,
                            centroid_weight,
                            token_ordering,
                            args.num_centroids,
                            args.top_k,
                            centroids_w,
                            config,
                        )
                        row += f" {ms:>9.4f}"
                    except Exception:
                        row += f" {'fail':>9}"
                print(row)

    print("\nbaselines (ms)")
    base = f"{'gather':>8} {'':>6} {'':>7}"
    dense = f"{'dense':>8} {'':>6} {'':>7}"
    for num_tokens in args.sweep_batch_sizes:
        x = x_by_t[num_tokens]
        gather_args = (
            x,
            lm_head,
            token_ordering,
            args.num_centroids,
            args.top_k,
            centroids_w,
        )
        base += f" {_bench(_top_tokens_gather, *gather_args):>9.4f}"
        dense += f" {_bench(_top_tokens_dense, x, lm_head):>9.4f}"
    print(base)
    print(dense)


def dedup_ceiling(args, centroid_weight, device, dtype):
    """Upper bound on what cross-token weight dedup could buy.

    moe_align_block_size can't take E=2048 (its CUDA kernel asserts
    padded_num_experts < 1024), so _grouped_gemm always uses naive block
    assignment and never dedups in-kernel. Fixing that kernel is only
    worthwhile if dedup is worth much.

    Times the GEMM alone under three routing patterns at fixed T:
      random  — every token picks its own centroids (realistic, no reuse)
      pool8   — tokens share one of 8 routing patterns (moderate reuse)
      shared  — every token picks the SAME centroids (perfect reuse)

    'shared' touches only top_k tiles total, so after the first token
    everything is L2-resident. random-vs-shared is therefore a generous
    upper bound: it conflates dedup with cache residency, and real routing
    is never that degenerate.
    """
    print("\nDedup ceiling — grouped GEMM only (ms)\n")
    print(
        f"{'T':>6} {'random':>9} {'pool8':>9} {'shared':>9} "
        f"{'max gain':>9} {'saved ms':>9}"
    )

    def bench_ids(x, ids):
        return _bench(
            _grouped_gemm,
            x,
            centroid_weight,
            ids.to(torch.int32).contiguous(),
            args.top_k,
            args.num_centroids,
        )

    for num_tokens in args.batch_sizes:
        x = torch.randn(num_tokens, args.hidden_size, device=device, dtype=dtype)

        rand_ids = torch.rand(num_tokens, args.num_centroids, device=device).topk(
            args.top_k, dim=-1
        )[1]

        pool = torch.rand(8, args.num_centroids, device=device).topk(
            args.top_k, dim=-1
        )[1]
        pool_ids = pool[torch.arange(num_tokens, device=device) % 8]

        shared_ids = rand_ids[0:1].expand(num_tokens, -1)

        t_rand = bench_ids(x, rand_ids)
        t_pool = bench_ids(x, pool_ids)
        t_shared = bench_ids(x, shared_ids)

        print(
            f"{num_tokens:>6} {t_rand:>9.4f} {t_pool:>9.4f} {t_shared:>9.4f} "
            f"{t_rand / max(t_shared, 1e-9):>8.2f}x {t_rand - t_shared:>9.4f}"
        )

    print()
    print("'max gain' bounds what fixing moe_align_block_size could deliver.")
    print("If it is small, the CUDA change is not worth the rebuild or the")
    print("risk of touching shared MoE code.")


def _router_stage(hidden_states, centroids_w, top_k):
    scores = torch.nn.functional.linear(hidden_states, centroids_w)
    return torch.topk(scores, k=top_k, dim=-1)[1]


def _router_fused_stage(fused_topk, hidden_states, centroids_w, top_k):
    scores = torch.nn.functional.linear(hidden_states, centroids_w)
    return fused_topk(hidden_states, scores, top_k, renormalize=False)[1]


def _decode_stage(logits, topk_ids, token_ordering, vocab_per_centroid):
    """Eager argmax + vocab-id decode, the baseline _decode_top_token replaces."""
    flat = logits.argmax(-1)
    slot = flat // vocab_per_centroid
    within = flat % vocab_per_centroid
    centroid = topk_ids.gather(-1, slot.unsqueeze(-1)).squeeze(-1)
    return token_ordering[centroid * vocab_per_centroid + within]


def breakdown(args, x_by_t, centroid_weight, token_ordering, centroids_w, lm_head):
    """Isolate where small-T time actually goes.

    Times the router, the grouped GEMM, and the argmax/decode tail
    separately against the full chain, plus a single trivial kernel as the
    harness floor. If GEMM << full, the path is overhead-bound on the
    surrounding elementwise/index kernels rather than on the matmul.
    """
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )

    vocab_per_centroid = args.vocab_size // args.num_centroids
    print("\nPer-stage breakdown (ms)\n")
    print(
        f"{'T':>4} {'floor':>8} {'router':>8} {'rtr_fus':>8} {'gemm':>8} "
        f"{'decode':>8} {'dec_fus':>8} {'full':>8} {'gather':>8} {'dense':>8} "
        f"{'proj':>8}"
    )

    for num_tokens in args.sweep_batch_sizes:
        x = x_by_t[num_tokens]

        # Single trivial elementwise kernel -> harness/launch floor.
        scratch = torch.empty(num_tokens, device=x.device, dtype=x.dtype)
        t_floor = _bench(torch.Tensor.add_, scratch, 1.0)

        t_router = _bench(_router_stage, x, centroids_w, args.top_k)

        # fused_topk: softmax is monotonic so top-k indices match torch.topk,
        # and it returns int32 ids directly (saves a cast kernel). May not
        # support num_experts=2048 / topk=32.
        try:
            fused_ids = _router_fused_stage(fused_topk, x, centroids_w, args.top_k)
            fused_ok = torch.equal(
                fused_ids.sort(-1)[0].to(torch.int64),
                _router_stage(x, centroids_w, args.top_k).sort(-1)[0].to(torch.int64),
            )
            t_router_fused = _bench(
                _router_fused_stage, fused_topk, x, centroids_w, args.top_k
            )
            rtr_fus = f"{t_router_fused:>8.4f}" if fused_ok else f"{'MISMATCH':>8}"
        except Exception:
            t_router_fused = t_router
            rtr_fus = f"{'unsup':>8}"

        topk_ids = _router_stage(x, centroids_w, args.top_k).to(torch.int32)
        t_gemm = _bench(
            _grouped_gemm,
            x,
            centroid_weight,
            topk_ids,
            args.top_k,
            args.num_centroids,
        )

        logits = _grouped_gemm(
            x, centroid_weight, topk_ids, args.top_k, args.num_centroids
        ).view(num_tokens, -1)

        decode_args = (logits, topk_ids, token_ordering, vocab_per_centroid)
        t_decode = _bench(_decode_stage, *decode_args)

        fused_args = (
            logits,
            topk_ids,
            token_ordering,
            args.top_k,
            vocab_per_centroid,
        )
        dec_ok = torch.equal(_decode_fused(*fused_args), _decode_stage(*decode_args))
        t_decode_fused = _bench(_decode_fused, *fused_args)
        dec_fus = f"{t_decode_fused:>8.4f}" if dec_ok else f"{'MISMATCH':>8}"

        moe_args = (
            x,
            centroid_weight,
            token_ordering,
            args.num_centroids,
            args.top_k,
            centroids_w,
        )
        t_full = _bench(_top_tokens_moe, *moe_args)

        gather_args = (
            x,
            lm_head,
            token_ordering,
            args.num_centroids,
            args.top_k,
            centroids_w,
        )
        t_gather = _bench(_top_tokens_gather, *gather_args)
        t_dense = _bench(_top_tokens_dense, x, lm_head)

        # Best achievable: cheapest router variant + gemm + fused decode.
        projected = min(t_router, t_router_fused) + t_gemm + t_decode_fused

        print(
            f"{num_tokens:>4} {t_floor:>8.4f} {t_router:>8.4f} {rtr_fus} "
            f"{t_gemm:>8.4f} {t_decode:>8.4f} {dec_fus} {t_full:>8.4f} "
            f"{t_gather:>8.4f} {t_dense:>8.4f} {projected:>8.4f}"
        )


def main(args):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    vocab_per_centroid = args.vocab_size // args.num_centroids
    if vocab_per_centroid * args.num_centroids != args.vocab_size:
        raise ValueError("vocab_size must be divisible by num_centroids")

    lm_head = torch.randn(args.vocab_size, args.hidden_size, device=device, dtype=dtype)
    centroids_w = torch.randn(
        args.num_centroids, args.hidden_size, device=device, dtype=dtype
    )
    token_ordering = torch.randperm(args.vocab_size, device=device)
    centroid_weight = (
        lm_head[token_ordering]
        .view(args.num_centroids, vocab_per_centroid, -1)
        .contiguous()
    )

    print(
        f"vocab={args.vocab_size} hidden={args.hidden_size} "
        f"centroids={args.num_centroids} top_k={args.top_k} "
        f"num_selected={args.top_k * vocab_per_centroid} "
        f"({100 * args.top_k / args.num_centroids:.2f}% of vocab)"
    )
    if args.dedup_ceiling:
        dedup_ceiling(args, centroid_weight, device, dtype)
        return

    if args.sweep_config or args.breakdown or args.router_bench:
        x_by_t = {
            t: torch.randn(t, args.hidden_size, device=device, dtype=dtype)
            for t in args.sweep_batch_sizes
        }
        if args.router_bench:
            router_bench(args, x_by_t, centroids_w)
            return
        run = sweep_config if args.sweep_config else breakdown
        run(args, x_by_t, centroid_weight, token_ordering, centroids_w, lm_head)
        return

    print(
        f"{'T':>5} {'gather':>10} {'gather+':>10} {'compiled':>10} {'moe':>10} "
        f"{'dense':>10} {'g+/moe':>8} {'c/moe':>7} {'g+==m':>7} {'c==m':>7}"
    )

    for num_tokens in args.batch_sizes:
        x = torch.randn(num_tokens, args.hidden_size, device=device, dtype=dtype)

        gather_args = (
            x,
            lm_head,
            token_ordering,
            args.num_centroids,
            args.top_k,
            centroids_w,
        )
        moe_args = (
            x,
            centroid_weight,
            token_ordering,
            args.num_centroids,
            args.top_k,
            centroids_w,
        )

        out_gather_plus = _top_tokens_gather_plus(*gather_args)
        out_moe = _top_tokens_moe(*moe_args)
        try:
            out_compiled = _top_tokens_gather_compiled(*gather_args)
            agree_cm = (out_compiled == out_moe).float().mean().item()
            compiled_ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"  compiled variant failed: {str(exc)[:120]}")
            agree_cm = float("nan")
            compiled_ok = False

        agree_gpm = (out_gather_plus == out_moe).float().mean().item()

        # CUDA-graph timing: these kernels are small enough that launch
        # overhead dominates wall clock, and the real path is graph-captured
        # by Gemma4Proposer anyway. Doubles as a capturability check.
        t_gather = _bench(_top_tokens_gather, *gather_args)
        t_gather_plus = _bench(_top_tokens_gather_plus, *gather_args)
        t_moe = _bench(_top_tokens_moe, *moe_args)
        t_dense = _bench(_top_tokens_dense, x, lm_head)
        if compiled_ok:
            t_compiled = _bench(_top_tokens_gather_compiled, *gather_args)
            compiled_cols = f"{t_compiled:>9.4f}m"
            ratio_c = f"{t_compiled / t_moe:>6.2f}x"
        else:
            compiled_cols = f"{'fail':>10}"
            ratio_c = f"{'-':>7}"

        print(
            f"{num_tokens:>5} {t_gather:>9.4f}m {t_gather_plus:>9.4f}m "
            f"{compiled_cols} {t_moe:>9.4f}m {t_dense:>9.4f}m "
            f"{t_gather_plus / t_moe:>7.2f}x {ratio_c} "
            f"{agree_gpm:>7.3f} {agree_cm:>7.3f}"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=262144)
    parser.add_argument("--num-centroids", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--sweep-config",
        action="store_true",
        help="Sweep grouped-GEMM block shapes at small T instead of the "
        "three-way comparison.",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Time router / GEMM / decode separately to locate small-T cost.",
    )
    parser.add_argument(
        "--router-bench",
        action="store_true",
        help="Compare top-k implementations, which dominate small-T cost.",
    )
    parser.add_argument(
        "--dedup-ceiling",
        action="store_true",
        help="Bound what fixing moe_align_block_size for E>=1024 could buy.",
    )
    parser.add_argument(
        "--sweep-batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8]
    )
    parser.add_argument(
        "--sweep-block-n", type=int, nargs="+", default=[16, 32, 64, 128]
    )
    parser.add_argument("--sweep-warps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--sweep-stages", type=int, nargs="+", default=[2, 3, 4])
    main(parser.parse_args())
