# Label QA correction implementation plan

> Execute inline with the existing benchmark, recovery and export layers. The user authorised fixing and rerunning only label_qa on main; do not push.

**Goal:** Supply the missing sector-code meanings to all models and replace only the affected category in a new, explicitly sourced report.

**Architecture:** `shve_cases.py` owns the corrected question. `shve_labelqa.py` freezes a category-only seed using the original cases, batches and repeat IDs, invokes existing missing-slot recovery, and creates a new full report with explicit category supersession provenance. Earlier evidence remains untouched.

**Protocol:** 60 primary cases, 14 original repeat cases with two additional answers each, two warm-ups per model; Jev/GPT/Nimble/Laya only. No churn or other task requests. Same installed models and configured endpoints. The full revised report retains original warm-ups and adds the new session warm-ups; costs include only evidence selected into that report.

- [x] Reproduce missing dictionary with a failing request-contract test; fix question.
- [x] Test corrected-case freezing, incomplete-run rejection and category-only replacement with preserved source evidence; include repeated interruption and actual older combined reports without saved run plans.
- [x] Implement a focused runner using existing audit, lock, local-runtime and recovery functions.
- [x] Run full tests, syntax and independent review before hosted inference: 142 tests passed; review fixes verified.
- [ ] Execute the authorised category-only protocol; never retry saved attempts.
- [ ] Audit complete results, unchanged source evidence and unchanged other category answers; export the full report and verify localhost/downloads.
- [ ] Update documentation, make a focused local commit and explain corrected findings.
