<!-- DRAFT for review -- not posted. Answers for HyperQwen's "Field report" issue form
     (.github/ISSUE_TEMPLATE/field-report.yml), one section per form field, ready to paste.
     The broader feedback (lookup port, 1M context) is a separate issue: hyperqwen-feedback-issue-DRAFT.md -->

## Title
Field report: 1x H200 141 GB (sm90) — setup B, full protocol

## Card, and how many
1x NVIDIA H200 141 GB SXM (sm90)

## Power limit during the run
700 W (the card's default, no cap set; `Current Power Limit : 700.00 W`)

## Driver, CUDA, OS, and container or venv
580.105.08, CUDA 13.0, Ubuntu 22.04.5 bare metal, native venv per docs/install.md (torch 2.13.0+cu130)

## Commit you ran
ce9615c

## Which setup
B — single, default (SPEC=dflash2 PREFIX_CACHE=1)

## Harness output — the second run
```
ROW cohort C1 real prompts T=default | e2e=252.03 tok/s | decode(C/meanTPOT)=258.4 | tok/step=3.24 | meanTTFT=99.07 ms
ROW cohort C2 real prompts T=default | e2e=429.41 tok/s | decode(C/meanTPOT)=501.3 | tok/step=3.19 | meanTTFT=117.19 ms
ROW cohort C4 real prompts T=default | e2e=668.20 tok/s | decode(C/meanTPOT)=856.5 | tok/step=3.01 | meanTTFT=141.10 ms
ROW cohort C8 real prompts T=default | e2e=927.78 tok/s | decode(C/meanTPOT)=1369.9 | tok/step=3.19 | meanTTFT=909.89 ms
ROW cohort C1 real prompts T=0 | e2e=253.27 tok/s | decode(C/meanTPOT)=263.2 | tok/step=3.19 | meanTTFT=88.99 ms
ROW cohort C2 real prompts T=0 | e2e=444.62 tok/s | decode(C/meanTPOT)=515.5 | tok/step=3.21 | meanTTFT=115.63 ms
ROW cohort C4 real prompts T=0 | e2e=721.85 tok/s | decode(C/meanTPOT)=954.7 | tok/step=3.30 | meanTTFT=138.64 ms
ROW cohort C8 real prompts T=0 | e2e=829.51 tok/s | decode(C/meanTPOT)=1307.2 | tok/step=3.25 | meanTTFT=866.88 ms
```

## Quality check
```
h200-setupB GSM8K n=200 acc=0.955 mean_tokens=382 44s
```

## Anything that did not work
Everything in the protocol worked out of the box on sm90. Model prep, patch apply and both harness runs went fine,
and GSM8K matches your 0.955. Roughly 2x the 3090 numbers, at the default 700 W. The first run (discarded) was only
a few percent below the second, e.g. C1 T=default 254.5 vs 258.4.

Two sm90 notes from running past this setup, in case they're useful:

- **`SPEC_ATTN` / `VLLM_SPEC_DECODE_ATTN` on H200:** neutral at short context. With `SPEC_ATTN=0`, the eight rows
  above moved −7% to +9% with no consistent sign. At long context it falls behind vLLM's stock attention: decode was
  −36% at 100K and −68% at 600K. That second measurement was on vLLM 0.32 with the patch ported, since 0.31 can't
  reach 600K. The `VLLM_USE_FLASHINFER_SAMPLER=0` fallback had no measurable cost at any length. So on sm90, gating
  it by context length might be worth it.
- **The DFlash2 drafters (`incoai/...-DFlash2` and your W4A16 one) have `max_position_embeddings: 262144`.** Prompts
  past that crash the server (`index out of bounds ... < 262144`). Your profiles stay under it, so this doesn't happen
  today, but raising the value in the drafter config fixes it if `MAX_LEN` ever goes past 262K.

More detail on both, and on porting the lookup drafting to vLLM 0.32, is in a separate issue: <link once posted>.
