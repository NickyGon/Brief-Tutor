# Brief Tutor — Roadmap

Phases and features in **small, shippable steps**. No fixed calendar—ordered by dependency and impact. Fits a vibe-coding pace: finish one slice, validate with evaluators, then pull the next.

Aligns with [mission.md](./mission.md) and [tech-stack.md](./tech-stack.md).

---

## Current Baseline (already in repo)

- LangGraph workflow with Theme / New Creative / Campaign Update agents
- RAG over Qdrant + Google Drive ingestion (`rag_ingestion.py`)
- Diagnosis formatter + document creator (resume + full listing)
- OpenAI provider path; Vertex code present but **not active**
- LangSmith tracing hook + JSONL workflow metrics
- CLI entry via `main.py`

**Active constraint:** Campaign Update often relies on **local XLSX** because Google Sheets access is blocked under current GCP/Drive setup.

---

## Phase 1 — Stabilize the Standalone Grooming Loop

**Goal:** Trustworthy local runs evaluators can complete in **10–15 minutes** including manual review.

| Step | Feature | Notes |
|------|---------|--------|
| 1.1 | **Campaign Update local path** | Document and standardize folder layout for Actual/Previous files; clear errors when files missing |
| 1.2 | **OpenAI-only hardening** | Default env to OpenAI; disable Vertex paths in docs/scripts to avoid confusion |
| 1.3 | **Output download UX (CLI)** | Single command or flag to emit both `.txt` artifacts to a known output dir |
| 1.4 | **Grounding + field eval gate** | Treat `EVAL_MIN_*` as release check before evaluator handoff |
| 1.5 | **Golden brief fixtures** | 2–3 anonymized briefs (Theme, New Creative, Campaign Update) for regression |
| 1.6 | **Reduce false negatives** | Prompt/tool tuning so critical issues (offer mismatches, missing assets) are not skipped |
| 1.7 | **Evaluator checklist** | Short “verify before deliver” list embedded in output or README |

**Exit criteria:** ≤5% workflow failure rate on golden set; evaluators prefer output over fully manual notes for at least one task type.

---

## Phase 2 — Quality & Consistency Depth

**Goal:** Better diagnoses and fewer wrong/missed calls—still local, still human-reviewed.

| Step | Feature | Notes |
|------|---------|--------|
| 2.1 | **Campaign Update matching** | Improve Actual vs Previous pairing when only partial overlap |
| 2.2 | **Critique vs rewrite policy** | Agent prompts: default critique; rewrites only when high-confidence and guideline-backed |
| 2.3 | **RAG corpus hygiene** | Dev-run ingestion checklist; validate folder names (Theme, New Creative, Campaign Update, Actual/Previous) |
| 2.4 | **Diagnosis memory loop** | Ensure past diagnoses in Qdrant help consistency (sync from Drive spreadsheets) |
| 2.5 | **Explainability in output** | Each diagnosis cites guideline snippet or retrieved brief id where possible |
| 2.6 | **Latency/cost pass** | Tune `max_tokens`, model choice, batching—stay within 10–15 min total session time |

**Exit criteria:** Evaluators report improved consistency vs manual-only; wrong-diagnosis rate acceptable on spot audits.

---

## Phase 3 — API + React UI (Internal Tool)

**Goal:** UI-adaptable grooming—no CLI required for daily use.

| Step | Feature | Notes |
|------|---------|--------|
| 3.1 | **Thin API wrapper** | FastAPI (or similar): `POST /run`, job id, status, result URLs |
| 3.2 | **API key auth** | Protect endpoints for internal users |
| 3.3 | **React shell** | Upload brief, pick task type, show run progress |
| 3.4 | **Results view** | Tabbed diagnoses + resume + full listing; highlight low-grounding items |
| 3.5 | **Human review UI** | Accept / reject / edit per diagnosis before download |
| 3.6 | **Download `.txt`** | Match current deliverable format for creative handoff |
| 3.7 | **Peak-load ergonomics** | Queue or batch list for month-start volume (simple list first, fancy queue later) |

**Exit criteria:** Primary users (C&C evaluators) can complete a brief end-to-end without editing Python.

---

## Phase 4 — Team Tooling & Ops

**Goal:** Sustainable tool for the grooming team—not a solo dev script.

| Step | Feature | Notes |
|------|---------|--------|
| 4.1 | **GitHub Actions** | PR checks: lint, unit tests, optional golden-brief smoke |
| 4.2 | **Unit test suite** | Parsers, `graph/models.py`, eval gates, tool mocks |
| 4.3 | **Prompt/version tags** | Tag agent YAML versions in metrics JSONL / LangSmith |
| 4.4 | **Run history** | Per-evaluator run log (brief id, duration, pass/fail) for 5% fail tracking |
| 4.5 | **Feedback to Creative** | Optional export template aligned with what Campaign PMs expect |
| 4.6 | **Onboarding doc** | One-pager for evaluators + PMs (not just developers) |

**Exit criteria:** Another developer can run, test, and deploy changes without tribal knowledge.

---

## Phase 5 — Source Access & Retrieval Expansion

**Goal:** Unblock Campaign Update and richer cross-campaign context.

| Step | Feature | Notes |
|------|---------|--------|
| 5.1 | **Fix Google Sheets/Drive access** | Resolve GCP service account / scope limits for Campaign Update sources |
| 5.2 | **Drive-only Campaign Update** | Remove local-file workaround once reliable |
| 5.3 | **Expand campaign retrieval** | Broader similarity search across historical briefs in Qdrant |
| 5.4 | **Dropbox (or equivalent) integration** | Similar campaigns by dealership id—**deferred until access approved** |

**Exit criteria:** Campaign Update runs from shared Drive corpus without manual file staging.

---

## Phase 6 — Deferred / Later

Only after Phases 1–4 are stable for daily evaluators.

| Item | Why deferred |
|------|----------------|
| **Vertex AI full integration** | Not working in current environment; OpenAI is sufficient for now |
| **Google-native vector store** | Qdrant is working; migration adds risk without clear win |
| **Supervised tuning / RLHF** | Prompt + RAG + eval gates should be exhausted first |
| **Multi-tenant / per-dealer KB** | Single shared KB is the current model |
| **Formal compliance program** | None required today |

---

## Priority Stack (when unsure what to build next)

1. **Missed critical issues** and **wrong diagnoses** (mission failure modes)
2. **Campaign Update source access** (active blocker)
3. **10–15 minute evaluator session** (speed + review UI)
4. **React + API** (audience requirement)
5. **Unit tests + CI** (sustainability)
6. Everything in Phase 6

---

## Backlog (nice-to-have)

- Batch mode: many briefs in one queue for month-start peaks
- Side-by-side diff UI for Campaign Update (Actual vs Previous)
- Dealer-specific rule packs in RAG metadata
- Slack/Teams notification when a run completes
- Dashboard from `workflow_metrics.jsonl` (no new APM required initially)

---

## How to Use This Roadmap

- Pick **one step** from the current phase, ship it, run a golden brief + evaluator spot check.
- Do not start Phase 6 while Phase 1 exit criteria are red.
- Update this file when a step ships or a blocker changes (e.g. Vertex or Drive access fixed).
