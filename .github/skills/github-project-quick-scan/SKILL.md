---
name: github-project-quick-scan
description: "Quickly scan a GitHub repository and return a concise, evidence-backed summary of what it does, how to run it, its current signals, and key caveats. Use for fast GitHub project overviews, triage, short comparisons, or deciding whether deeper research is warranted."
argument-hint: "Provide a GitHub URL or owner/repository and the decision you need to make"
user-invocable: true
disable-model-invocation: false
---

# GitHub Project Quick Scan

## When to Use

Use this skill when the user needs a fast first-pass view of a GitHub repository, not a full technical due-diligence report.

## Boundaries

- Keep the scan concise and time-bounded.
- Use repository pages, README files, metadata, releases, and a small sample of source or configuration files as primary evidence.
- Do not run repository code or treat stars, forks, downloads, or contributor counts as proof of quality.
- Do not call a project secure, production-ready, actively maintained, or abandoned without dated evidence.
- Do not invent missing setup steps, capabilities, metrics, or project intent.
- Never reproduce secrets, tokens, personal data, or sensitive configuration.
- Recommend the full `github-project-research` skill when the decision involves adoption, security, licensing, architecture, or significant investment of time.

## Procedure

1. Confirm the canonical repository, stated purpose, default branch, visibility, license, archive status, and research date.
2. Read the README and inspect the most relevant metadata, dependency manifest, entry point, example, test, or configuration file.
3. Summarize the project in one sentence, then identify its main inputs, outputs, users, and runtime or setup assumptions.
4. Check a small set of health signals: latest release or commit date, recent issue or pull-request activity, CI or test evidence, and documentation completeness when visible.
5. Separate observations from interpretations. Record missing or unverified information instead of filling gaps.
6. Answer the user's stated decision with a brief recommendation and one next step that would reduce uncertainty.

## Output Format

### Snapshot
- Repository:
- Purpose:
- Status:
- License:
- Researched:

### What It Does
Two to four sentences describing the project and typical workflow.

### Signals
- Maintenance:
- Documentation and setup:
- Tests and CI:
- Adoption indicators:

Use "not verified" where the repository does not provide enough evidence.

### Fit
One short paragraph answering the user's specific question.

### Caveats
Two to five concrete risks, unknowns, or claims that need verification.

### Next Step
The single most useful follow-up check. Mention the full `github-project-research` skill when a deeper assessment is warranted.

### Sources
Link the primary repository pages used for the scan.
