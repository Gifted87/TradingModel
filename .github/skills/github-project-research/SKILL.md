---
name: github-project-research
description: "Research a GitHub repository or project and produce an evidence-backed report covering purpose, architecture, activity, adoption signals, setup, strengths, risks, and fit for the user's goal. Use when asked to investigate, evaluate, compare, summarize, or diligence a GitHub project."
argument-hint: "Provide a GitHub URL or owner/repository and the research question"
user-invocable: true
disable-model-invocation: false
---

# GitHub Project Research

## When to Use

Use this skill when the user wants to understand, evaluate, compare, summarize, or perform lightweight technical due diligence on a GitHub repository or organization project.

Typical requests include:

- Explain what a GitHub project does and how it is structured.
- Assess whether a repository is suitable for a use case.
- Compare two or more GitHub projects.
- Investigate maintenance, release history, dependencies, licensing, security signals, or adoption.
- Produce a concise research brief for a technical or non-technical audience.

## Research Principles

- Prefer primary sources: the repository, its releases, issues, pull requests, documentation, package metadata, and linked official project sites.
- Label every important statement as observed fact, reasoned inference, or open question.
- Use dates for activity and release claims. Do not describe a project as active, abandoned, popular, secure, or production-ready without supporting evidence.
- Treat stars, forks, downloads, and contributor counts as signals, not proof of quality or adoption.
- Do not expose secrets, tokens, private data, or sensitive configuration discovered during research.
- Do not run untrusted repository code. Read source and configuration files; only execute commands when the user explicitly asks and the action is understood and safe.
- State the research date and note when a source could not be verified.
- Keep conclusions proportional to the evidence. Surface uncertainty instead of filling gaps with plausible guesses.

## Procedure

1. **Define the question**
   - Identify the repository URL or owner/repository.
   - Extract the user's decision: learn, adopt, compare, contribute, invest time, or communicate.
   - Clarify the target audience and desired depth only when they materially change the research. Otherwise use a concise technical brief.

2. **Establish repository identity**
   - Confirm the canonical repository, owner, default branch, visibility, license, and stated project description.
   - Check whether the repository redirects, is archived, or appears to be a fork.
   - Record the research date.

3. **Read the primary materials**
   - Start with the README and documentation.
   - Inspect package or project metadata, dependency manifests, configuration, entry points, examples, tests, and release notes as relevant.
   - Use source code to verify important claims about architecture and behavior; do not infer implementation details from the README alone.

4. **Map the project**
   - Summarize the problem, users, inputs, outputs, main components, and typical workflow.
   - Identify the principal languages, frameworks, runtimes, build system, data stores, external services, and deployment assumptions.
   - Describe only the architecture that can be supported by source, documentation, or configuration evidence.

5. **Assess project health**
   - Review recent commits, releases, issue and pull-request activity, contributor distribution, CI status, and versioning practices when visible.
   - Check documentation quality, test coverage signals, dependency freshness, security policy, and response patterns.
   - Distinguish current observations from historical activity. A single recent commit is not enough to establish sustained maintenance.

6. **Assess fit and risk**
   - Compare the project's stated and observed behavior against the user's requirements.
   - Cover operational complexity, compatibility, license obligations, maturity, bus factor, security posture, performance evidence, extensibility, and migration or lock-in concerns when relevant.
   - Identify what should be tested in a proof of concept before adoption.

7. **Produce the report**
   - Cite or name the source for each material claim using links where available.
   - Separate verified facts, interpretation, and unanswered questions.
   - End with a recommendation tied directly to the user's stated goal, including confidence and the next validation step.

## Decision Rules

- If the repository cannot be accessed, say so and provide a bounded plan for what to inspect next; do not fabricate a summary.
- If documentation and code disagree, report the discrepancy and prefer current executable behavior or configuration for implementation claims.
- If the project is a fork or mirror, research both the repository and the upstream project when that distinction affects maintenance or licensing.
- If the user asks for a comparison, apply the same criteria and time window to every project before drawing conclusions.
- If the user asks for a social post or marketing copy after research, provide the evidence-backed research first, then clearly separate copy from findings.

## Output Format

Use this structure unless the user requests another format:

### Executive Summary
Two to five sentences answering the user's main question.

### Project Identity
- Canonical repository and current status
- License and research date
- Stated purpose

### How It Works
A concise architecture and workflow description, with source links for material claims.

### Evidence And Health
- Releases, commits, issues, pull requests, contributors, CI, tests, and documentation signals that were actually checked
- Dates and limitations for each signal

### Fit For The User's Goal
- Relevant strengths
- Constraints and risks
- Proof-of-concept checks or questions to resolve

### Recommendation
A direct recommendation with confidence level and the reason it is proportionate to the evidence.

### Sources And Open Questions
Primary links, secondary links if used, and claims that still require verification.
