# Brief Tutor — Mission

## Problem Statement

Campaign brief grooming at Cox is a **project-level consistency challenge**, not a one-off task. C&C evaluators must review every campaign on a spreadsheet, note observations across the full brief (or selected campaigns), and align feedback with Cox Automotive campaign standards.

Today, a single brief typically takes **about one hour** of careful manual review. That does not scale well when volume is high—especially at month start, when peak loads hit.

**Brief Tutor** exists to compress that cycle while keeping humans in control: run automated consistency analysis, then let the evaluator verify and refine before delivery.

---

## Vision

**Near term:** A reliable assistant that helps C&C Grooming evaluators produce **consistent, explainable diagnoses faster**—targeting roughly **15 minutes** per brief (run workflow → manual verification → deliver), versus ~60 minutes fully manual.

**Longer term:** A team-standard tool that improves feedback quality to the **Campaign Creative** department—clear, grounded observations backed by guidelines and historical context, without replacing human judgment.

Brief Tutor **assists** grooming; it does **not** replace C&C evaluators.

---

## Target Audience

| Role | Relationship |
|------|----------------|
| **C&C Grooming evaluators** (primary) | Run workflows, verify outputs, deliver diagnosis lists |
| **Campaign Project Managers** (secondary) | Authors of task instructions; consumers of clearer feedback loops |
| **QA** (secondary) | Quality checks on process and outputs |

**Usage pattern:** Many briefs per day; **peak load at the beginning of the month**.

**Interaction:** Must support a future **UI-first** experience; evaluators should not depend on CLI literacy.

---

## Scope

### In scope

- Ingest and evaluate **Campaign Brief** spreadsheets (task types: **Theme**, **New Creative**, **Campaign Update**).
- **Consistency analysis** and critique aligned with Cox Automotive campaign guidance (via RAG).
- **Grounded diagnoses** with explainable rationale tied to retrieved context.
- **Suggestions / rewrites** only when applicable—final creative changes remain on the client/creative side using delivered feedback.
- Structured outputs: formatted diagnoses, **brief resume**, and **full diagnoses listing**.
- **Human-in-the-loop:** every run is reviewed and approved by an evaluator before results are treated as final.
- Delivery as **downloadable text** (`.txt`) for handoff.

### Out of scope (for now)

- Fully autonomous approval or publishing without human review.
- Replacing C&C evaluators or owning final creative rewrites end-to-end.
- Formal compliance / governance programs (none identified today).
- Vertex-only or multi-cloud mandates (see [tech-stack.md](./tech-stack.md)).

### Explicitly deferred (see [roadmap.md](./roadmap.md))

- Full **Vertex AI** production path (blocked / not working in current environment).
- **Dropbox** (or similar) access for cross-campaign dealership matching.
- Broader source systems beyond current Drive + local-file constraints for Campaign Update.

---

## Principles

1. **Grounding over fluency** — Recommendations must follow **RAG-backed Cox Automotive campaign guide** rules; no invented campaign facts or guidelines.
2. **Critique first** — Primary value is accurate, consistent observation; rewrites are optional and secondary.
3. **Human final say** — Automation proposes; evaluators dispose.
4. **Explainability** — Evaluators must be able to trust *why* a diagnosis was raised.
5. **Consistency** — Same standards applied across briefs, task types, and evaluators.

---

## Success Criteria

### Measurable

| Metric | Target |
|--------|--------|
| Time per brief (run + manual verify + deliver) | **10–15 minutes** |
| Workflow / output failure rate | **≤ 5%** |
| Groundedness (diagnoses tied to retrieved evidence) | Maintain eval thresholds (`EVAL_MIN_GROUNDEDNESS`, field completeness gates) |

### Qualitative

- **Consistency** — Observations align with team norms and Cox guidelines.
- **Explainability** — Evaluators can trace diagnoses to source guidance or retrieved briefs.
- **Speed** — Noticeable reduction in manual scanning time without sacrificing thoroughness.

### Quality bar

- **Assist, not replace** — Brief Tutor is successful when evaluators deliver **better diagnosis lists faster**, not when headcount is removed.
- **Failure modes to minimize:** wrong diagnoses, **missed critical issues** (false negatives are worse than noisy true positives).

---

## What “Done” Looks Like for a Grooming Session

1. Evaluator submits a brief (via UI → workflow invocation).
2. Workflow returns grounded diagnoses and summary documents.
3. Evaluator reviews, corrects, or drops items (**required**).
4. Evaluator downloads `.txt` (or equivalent) and delivers feedback to creative / project stakeholders.

---

## Name

**Brief Tutor** is the official project name.
