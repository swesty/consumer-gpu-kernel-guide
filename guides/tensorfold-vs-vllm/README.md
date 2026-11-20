# TensorFold vs vLLM on DGX Spark

A study of [TensorFold](https://github.com/ashhart/TensorFold) (a Zig + hand-written-CUDA inference engine popular on DGX Spark) against vLLM. It also measures one port, a fix to vLLM's RoCE all-reduce, on a two-Spark vLLM lane. Date: 2026-11-18.

- **Report:** [`report.html`](report.html), published at <https://pages.swesty.net/kernels/tensorfold-vs-vllm/>.
- **Sibling study:** [SM120 / SM121 attention kernels](../sm12x-attention/REPORT.md).

## Findings

1. **TensorFold's GLM-5.3-Flash decode lead over vLLM (1.45–2.26×) comes mostly from weight bytes, not kernels.**
   - TensorFold requantizes the checkpoint's BF16 non-expert weights to 4-bit at load. With BF16 non-experts it runs 0.90–1.26× the vLLM kit, measured on the same weights and the same client ([jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark), `docs/CHANGES-SUMMARY.md`).
   - Per byte, vLLM's Marlin kernels are faster than TensorFold's.
2. **Real structural differences**, which are not about speed per byte:
   - bit-exact guarantees (batched equals solo, drafted equals serial);
   - recurrent-state snapshots every 64 tokens for hybrid attention models;
   - per-stream draft depth chosen by a cost model.
3. **Worth porting into vLLM:**
   - the b12x RoCE one-shot all-reduce for TP across Sparks;
   - "log and fold" rollback of linear-attention state under speculative decoding;
   - L2 prefetch during collectives, gated per model.
4. **The RoCE all-reduce already exists** in the `local-inference-lab/vllm` fork that eugr's `--exp-b12x` images build, behind `VLLM_ENABLE_ROCE_ALLREDUCE=1`.
   - **The bug:** it never initialised on our lane. b12x read the launcher's `NCCL_IB_GID_INDEX=-1` as a literal index, so `ibv_query_gid` failed with `Invalid argument`.
   - **The fix:** [`roce-gid-mod/`](roce-gid-mod/) selects the IPv4 RoCE v2 GID per node, as NCCL does. The right index differs per node (5 vs 4), so a fixed value cannot work.
   - **Measured, A/B/A on fresh starts:** +7–11% decode at 1 stream; +2–6% at 4 streams, which is within noise.
5. **Two side findings:**
   - a 13-day-old vLLM container decoded 17–20% slower than a fresh start of the same configuration;
   - RoCE's startup work needs about 1.3 GiB more, enough to fail vLLM's startup memory check on a tight GB10.

## Layout

| Path | What it is |
|---|---|
| `report.html` | The full report. Standalone page; the kernels site injects its nav when it publishes. |
| `bench/decode_bench.py` | Stdlib-only decode benchmark: greedy, thinking off, 512 tokens; 1 stream (median of 3) and 4 concurrent streams. |
| `bench/ab-arm.sh` | Flips `VLLM_ENABLE_ROCE_ALLREDUCE` in the GLM recipe on spark-3 and reloads through `~/glm53-load.sh` (one A/B arm). |
| `bench/results/` | Raw JSON for each arm, plus MTP acceptance per arm. |
| `roce-gid-mod/` | The eugr-style launch-time mod (`run.sh`, `patch_roce_gid_auto.py`) and its test against fixtures copied from both nodes' GID tables. |

## Reproduce

```bash
# Benchmark the served model (run from any host that can reach the lane)
python3 -I bench/decode_bench.py http://10.10.10.46:8888/v1/chat/completions glm-5.3-flash my-label

# One A/B arm: 0 = PyNCCL, 1 = RoCE (run from the Mac Studio; reloads the lane, ~5 min, lane dark meanwhile)
ssh spark-3 'bash -s 1' < bench/ab-arm.sh

# Test the GID patch against a copy of b12x's roce_oneshot.py taken from the serving image
python3 -I roce-gid-mod/test_gid_auto.py /path/to/roce_oneshot.py
```

After any load, check that the head log contains `Using RoCEnante (b12x one-shot RoCE collectives)` and `Using ['B12X_ROCENANTE', 'PYNCCL'] … group 'tp:0'`.

## Evidence labels and caveats

The report labels every claim as **Measured** (a published or locally measured number), **Source** (read in the code) or **Inference** (reasoning). Two caveats:
- Comparisons across different weights or clients are marked as not apples-to-apples.
- The weight-bytes attribution assumes the vLLM kit ran BF16 dense layers. That is inferred from its 82 GiB-per-node load. If it actually ran FP8, the bytes share drops to about 10–14%.

## Licensing

- **Ported code:** `roce-gid-mod/patch_roce_gid_auto.py` patches b12x (Apache-2.0, now in FlashInfer as `flashinfer/experimental/b12x`).
- **Sources read for the study:** TensorFold (Apache-2.0 from v0.6.0) and jayleaton's repos (Apache-2.0).
- **Not copied from:** MiaAI-Lab's vLLM kits, which are AGPL-3.0 since 2026-09-07.
