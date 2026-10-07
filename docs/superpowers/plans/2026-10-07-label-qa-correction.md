# Label QA correction implementation plan

> Execute inline with the existing benchmark, recovery and export layers. The user authorised fixing and rerunning only label_qa on main; do not push.

**Goal:** Supply the missing sector-code meanings to all models and replace only the affected category in a new, explicitly sourced report.

**Architecture:** `shve_cases.py` owns the corrected question. `shve_labelqa.py` freezes a category-only seed using the original cases, batches and repeat IDs, invokes existing missing-slot recovery, and creates a new full report with explicit category supersession provenance. Earlier evidence remains untouched.

**Protocol:** 60 primary cases, 14 original repeat cases with two additional answers each, two warm-ups per model; Jev/GPT/Nimble/Laya only. No churn or other task requests. Same installed models and configured endpoints. The full revised report retains original warm-ups and adds the new session warm-ups; costs include only evidence selected into that report.

- [x] Reproduce missing dictionary with a failing request-contract test; fix question.
- [x] Test corrected-case freezing, incomplete-run rejection and category-only replacement with preserved source evidence; include repeated interruption and actual older combined reports without saved run plans.
- [x] Implement a focused runner using existing audit, lock, local-runtime and recovery functions.
- [x] Run full tests, syntax and independent review before hosted inference: 142 tests passed; review fixes verified.
- [x] Execute the authorised category-only protocol; never retry saved attempts. Exactly 360 requests / 90 per model completed; one Nimble warm-up timed out during approved restoration of its existing cloud-evicted weight file. No scored-request errors or invalid answers.
- [x] Audit complete results, unchanged source evidence and unchanged other category answers; export the full report and verify localhost/downloads. All source/revised audits clean; 6,256 selected requests, 56 confusion cohorts and eight weighted churn cohorts preserved. All fourteen browser tabs rendered. Actual HTML/CSV/JSON downloads verified.
- [x] Update documentation and make focused local commits without pushing. Corrected evaluation: Jev/GPT 48/48, Nimble 37/48, Laya 20/48. Explain these findings in the completion response.

Verified artifacts: corrected evidence `results/20261007_154819_recovery_5ab997`;
complete revised report `results/20261007_155903_labelqa_revised_2f475b`;
verification log `results/shve-labelqa-preflight/verification.json`.
Original `results/20261007_150410_combined_7c24d7` remains untouched.
The owned server on port 8767 now serves the revised report.
