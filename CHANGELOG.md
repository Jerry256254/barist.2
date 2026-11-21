# Changelog

No silent overwrites of the same version: a new checkpoint ships as a new version
(Barist.1.1, ...) or a git tag.

## Barist.2 — fifth adversarial review (released; own repos: `Jerry256254/barist.2` + `KucLab/Barist.2`, git tag `barist-2`)
Checkpoint `checkpoints_b2w/best.pt` (phase W on top of `b1q`). All six items of the
Barist.1.1 review fixed:

1. CSAM/self-harm hard block: for cs/sk/default text, category score
   sexual_abuse or self_harm >= 0.3 forces `unsafe` with `override: category_block`
   regardless of the language threshold (in `kuclab/infer.py`, `kuclab/cli.py`,
   `scripts/censer.py` incl. the chunked long-text path). Regex extended:
   `vzít si život`, `přestat žít`, `pořezat/podřezat`, `zemřít doma`, `otrávit se/prášky`,
   `dětský materiál`, `nezletil*.{0,20}(fotk|vide)`, `děti bez oblečení`, hate-group
   violence patterns. The false "0 CSAM/self-harm passes" claim is REMOVED from
   README/model card.
2. `mustcatch_heldout2.jsonl` restored to the full 91 rows; the 20 train rows with
   >0.8 5-gram overlap were purged FROM TRAIN (not from the test);
   `reports/heldout2_dedup.json` gate passes (max overlap 0.75).
3. cs language threshold FIXED back to 0.5 (tuned value kept as diagnostics only);
   `eval_gates.py` measures cs_benign FPR with the deployed pipeline.
4. Unsafe verdicts always carry >=1 category (top-1 fallback) and categories are
   sorted by severity priority — now also in the chunked path of `censer.py`.
5. Regex holes closed: school patterns in both word orders + multilingual
   (es/fr/de/it/ru/uk), `dan mode`, multilingual poison patterns.
6. `gates.json` reads report-level metrics (ECE, ONNX parity, regex gate, dedup)
   from each model's OWN reports dir; user benign set is 43 sentences with matching
   denominators.
- Phase W training data (`scripts/extend_barist1w.py`): 360+ unsafe formulations
  (self-harm without "zabít", school threats, poison, group hate, ru/uk, market)
  each paired 1:1 with a benign twin sharing the frame words; must_catch_weight =
  hard_negative_weight = 11.
- Renamed to Barist.2 everywhere: `barist.2` CLI launcher, `pyproject.toml`
  (`kuclab-barist2`), Dockerfile `ARG BARIST_REPO=KucLab/Barist.2`, Hub bundle
  name `Barist.2`.

## Barist.1.1 — fourth adversarial review (released; git tag `barist-1.1`)
Checkpoint `checkpoints_b1q/best.pt` (phases M–V). Release gates measured by
`scripts/eval_gates.py` -> `reports/gates.json` (8 PASS / 7 FAIL, FAILs documented in README).

- Regex pre-filter v3.1: legal-weapon / pyrotechnics / pest-control exceptions
  (downgrade to `needs_review`, not a block); `fet` only in drug context; napalm only
  instruction forms; extended self-harm patterns (`řežu se`, `podřezat si žíly`,
  `prášky na zabití`, `předávkovat se`); regex gate `scripts/test_regex.py`
  (321 benign trigger sentences, 0 hard FP) wired into pytest.
- New clean held-out set `data/mustcatch_heldout2.jsonl` (91 rows); old set renamed
  `data/mustcatch_dev.jsonl` (dev only). Report renamed `reports/mustcatch_heldout2.json`.
- Training phases M–R (must_catch ×15–25, benign counter-pairs); checkpoint `b1q`.
- CLI/API: `needs_review` for jailbreak patterns (not a hard block), `needs_support`
  (Linka bezpečí) for self-harm support contexts, `override: must_catch` field.
- Files published in-repo: `data/mustcatch_heldout2.jsonl`, `data/mustcatch_dev.jsonl`,
  `data/cs_benign.jsonl`, `data/regex_benign_test.jsonl`, `kuclab/probes.py`,
  `reports/onnx_parity.json`.

## Barist.1 — third adversarial review
- Renamed from Censer line; pre-filter v2; phases I–L; checkpoint `b1l`.
- Successor banners added to Censer.8 / Censer.9 READMEs (GitHub + Hugging Face).
