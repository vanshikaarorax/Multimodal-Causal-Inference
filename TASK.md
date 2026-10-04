# MemoLogs: AI/ML Engineer Take-Home Assessment

Thanks for taking the time to do this. Please read the whole file before you start.

## At a glance

| | |
| --- | --- |
| Time budget | 6–8 hours of focused work. We do not reward going far beyond it. |
| Deadline | 5 days after you receive this pack |
| Language | Python 3.11+, any open-source libraries |
| LLMs | Allowed (any provider or local model). A mock-LLM mode must also work offline. |
| AI coding assistants | Allowed. You must be able to explain and defend every line. |
| Afterwards | A 60-minute call where you walk us through your work |

You will build two things:

- **Part A (about 65% of the weight):** a persona suggester for ad creatives that shows its evidence and abstains when the evidence is weak.
- **Part B (about 35%):** an estimate of the revenue lift caused by a new creative, graded by how far it can be trusted.

## Why this task

MemoLogs turns marketing data into decisions, and every answer carries an explicit grade of how far it can be trusted. When the data is not good enough, we say so and name the gap instead of giving a weaker number. **A system that abstains with a reason beats one that always answers.** We are looking for engineers who build that way by default.

The ad creatives are real programmatic display ads from the public AdImageNet dataset; see `README.md` for the source and licence. Persona labels, launch dates and the geo panel are synthetic.

## The data

See `README.md` for column-level detail.

| File | Rows | Contents |
| --- | --- | --- |
| `creatives.parquet` | 2,882 | `creative_id`, `ad_size`, `ad_text` (text extracted from the ad, noisy), `image_path`, `launched_at` |
| `images/` | 2,882 | Real display ads in 8 standard sizes, from 320×50 banners to 300×600 skyscrapers. Many campaigns appear more than once, often in different sizes. |
| `confirmed_personas.csv` | 600 | Human-confirmed persona labels, multi-label, pipe-separated. They may contain errors. |
| `personas.yaml` | 8 | The persona names and definitions. These are the only allowed labels. |
| `dup_pairs_eval.csv` | 120 | Hand-labelled duplicate / non-duplicate pairs. Hold these out for evaluation. |
| `geo_panel.csv` | 10,800 | Daily `revenue` and `spend` for 60 geos over 180 days, plus `treated_group` |

## Part A: persona suggester with evidence

1. **Embed** every creative, using both image and text. Any open model is fine (for example CLIP or SigLIP). Justify the model choice, how you handle the very different ad sizes, and how you combine the two modalities.
2. **Detect near-duplicates**: the same campaign appearing more than once, at the same or a different ad size. Report precision and recall on `dup_pairs_eval.csv`, and explain how you chose the threshold. Do not tune on the eval pairs more than once.
3. **Score personas** for unlabelled creatives. Compare at least two approaches (for example kNN label propagation vs. a classifier trained on the embeddings). Your train/test split must not let near-duplicates leak across it.
4. **Abstain when the evidence is weak.** Each result returns `suggest`, `suggest_with_caution` or `abstain`, with the reason. Show the trade-off between coverage and precision.
5. **Explain** each `suggest` with a 1–2 sentence rationale written by an LLM. The prompt may only use the retrieved evidence: the nearest neighbours, their confirmed labels and the scores. The output must validate against a JSON schema. If the output is invalid or unsupported by the evidence, fall back to a deterministic template.

The CLI command `suggest --creative-id <id>` should return this shape:

```json
{
  "creative_id": "c_0412",
  "decision": "suggest",
  "personas": [{"name": "Traveler & Leisure", "score": 0.81}],
  "evidence": [{"creative_id": "c_0077", "similarity": 0.93, "near_duplicate": true}],
  "rationale": "...",
  "rationale_source": "llm"
}
```

`rationale_source` is `llm` or `template`.

## Part B: lift estimate with a Trust State

The 6 geos with `treated_group = 1` started running a new creative on **2026-06-04** (day 150 of the panel). Assignment was not randomised.

1. Estimate the incremental revenue caused by the creative in those 6 geos from 2026-06-04 to 2026-07-03. Use any method you can defend, for example synthetic control, difference-in-differences or a Bayesian structural time series.
2. Report a point estimate, a 90% interval and one of these four Trust States, with written rules for how you assign it:

   | Trust State | Meaning |
   | --- | --- |
   | `trusted` | Direction and size both support action |
   | `directionally_trusted` | Direction is reliable, size is approximate |
   | `magnitude_uncertain` | Effect size is too unstable to act on its size |
   | `not_trusted` | The evidence supports no recommendation |

3. Run at least one placebo or pre-period check, and state what would change your Trust State.
4. Do not assume the panel contains only the creative change. If something else is going on, find it and explain how you handled it.

## Optional stretch goals

Pick at most one. A strong core beats a weak core plus a stretch goal.

- **Calibration:** show whether persona scores are calibrated (reliability plot, ECE) and fix them if not.
- **Serving:** a FastAPI endpoint with a vector index (FAISS or pgvector) that answers `suggest` in under 200 ms at p95 with the mock LLM.
- **LLM faithfulness check:** an automated test that rationales cite only evidence IDs present in their input, with a pass rate.
- **Experiment as a prior:** use the Part B result as a prior in a small Bayesian marketing mix model, and explain why that same experiment must not also be used to validate the model.

## What to submit

Send a private Git repo (or a zip) containing:

- [ ] A Python package with `pyproject.toml` and pinned dependencies
- [ ] One command (`make all` or similar) that reproduces every number in your report from the raw data, with fixed seeds
- [ ] Tests for the parts most likely to break: the abstention rules, the JSON-schema fallback, the leakage-safe split and the Trust State rules
- [ ] A `MOCK_LLM=1` mode so we can run everything offline without API keys
- [ ] `REPORT.md`, at most 2 pages: your decisions, results tables, what didn't work, and what you would do with two more weeks
- [ ] A short note on how long you spent and which AI tools you used

Rules:

- Do not commit secrets or API keys.
- Everything must regenerate on a clean machine, on CPU, in under 20 minutes. You may cache embeddings, but include the script that created them.
- Do not commit the dataset to a public repository, and use the ads only for this assessment.

## How we evaluate

| Area | Weight | What great looks like |
| --- | --- | --- |
| Retrieval and near-duplicates | 15% | A sound way of combining modalities; a threshold chosen from data; analysis of the misses |
| Persona modelling and evaluation | 20% | A leakage-safe split, an honest baseline, per-persona metrics, sensible handling of label noise and rare classes |
| Abstention and trust logic | 15% | Explicit rules, a coverage–precision curve, reasons a person can act on |
| LLM layer | 10% | Prompt bounded to the evidence, schema validation, deterministic fallback, no invented IDs |
| Lift estimate (Part B) | 20% | A defensible method, placebo checks, other effects found, a Trust State that matches the evidence |
| Engineering quality | 10% | Reproducible, tested, readable, clean module boundaries |
| Communication | 10% | The report leads with results, states limitations plainly and explains trade-offs |

Things that count heavily against a submission: tuning on the test set, a single lift number with no uncertainty, an LLM rationale that cites creatives not in its evidence, or a pipeline that cannot run offline.

## Questions

If something is ambiguous, make a reasonable assumption, write it down in `REPORT.md` and move on. Stating your assumptions clearly is part of what we assess. For anything blocking, email us and we will reply within one working day.

Good luck, and we look forward to talking it through with you.
