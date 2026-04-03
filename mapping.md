# Building the Capability Map from Historical Value Stream Usage

## Short answer

**Yes — this is a very good idea.**

Going through:

1. each **value stream**
2. each of the **15 historical tickets**
3. why that value stream was used
4. how it was justified in the ticket

is one of the best ways to create the **first strong version** of your capability map.

This is much better than trying to invent the map from scratch.

---

# Why this is a good idea

Right now your challenge is not only retrieval quality.

The deeper problem is:

- the system often understands the business meaning
- but the exact value stream is not always surfaced strongly enough
- especially when the stream is indirect, implied, or overshadowed by louder themes

A capability map helps bridge:

- **business cues in the card**
- to
- **the correct value stream labels**

The best way to build that bridge is to look at real historical examples and ask:

- why was this VS attached?
- what phrases or business cues justified it?
- was it direct or indirect?
- what capability does that VS really represent?

That gives you grounded mappings instead of guessed mappings.

---

# Why historical ticket review is stronger than guessing

If you create the map only from value-stream names and descriptions, you will miss:

- how the business actually talks about the VS
- real synonyms
- indirect business phrasing
- domain-specific patterns used in actual idea cards
- cross-stream relationships

For example, a ticket may never say:

- "Ensure Compliance"

but may strongly imply it through:
- privacy
- audit controls
- balancing
- PII
- regulatory review
- consent

That is exactly the kind of knowledge you want the capability map to capture.

So yes, using the 15 tickets is a strong approach because it grounds the map in **real usage**, not just definitions.

---

# Best way to do it

Do **not** simply ask:

> Which VS is on this ticket?

Ask instead:

> What business capability is present here, and how did that lead to this VS label?

That gives you a reusable capability map instead of a one-off explanation.

---

# Recommended workflow

## Step 1 — go value stream by value stream

For each target value stream:

- collect the historical tickets among the 15 where that VS appears
- inspect:
  - ticket title
  - summary text
  - evidence sentences
  - analog summary
  - any raw supporting snippets if needed

Then ask:

- why was this VS used here?
- what exact wording or business meaning supports it?
- is the signal direct or indirect?
- what recurring cues appear across tickets?

---

## Step 2 — identify the real underlying capability

For each VS, define the business capability it really represents.

### Example

#### Value Stream
`Ensure Compliance`

#### Underlying capability cluster
- privacy / compliance / audit / control / regulatory governance

#### Common cues
- privacy
- pii
- audit
- controls
- regulatory
- hipaa
- consent
- policy
- governance

Now you are not just mapping words to VS.
You are mapping **capability → VS**.

That is much more stable.

---

## Step 3 — separate direct signals from indirect signals

This matters a lot.

Some tickets use a VS because it is:
- explicitly the main business focus

Others use it because it is:
- operationally implied by the initiative

For each VS, capture both.

### Example

#### Direct signal
- "regulatory controls required"
- "audit workflow"
- "PII handling"

#### Indirect signal
- "new vendor/member data-sharing process with privacy implications"

Both can justify `Ensure Compliance`, but they should not have equal strength.

---

## Step 4 — create one capability mapping entry per VS or per capability cluster

Do not create only term lists.

Create a proper entry with:

- capability name
- what it means
- direct cues
- indirect cues
- promoted value streams
- example tickets
- example evidence
- confidence / weight

---

# Best output structure

I recommend that for each VS you produce a structured note like this:

```yaml
value_stream: Ensure Compliance
capability_cluster: compliance_privacy_audit
description: >
  Covers privacy, auditability, control obligations, consent,
  regulatory constraints, and compliance requirements.

direct_cues:
  - privacy
  - pii
  - audit
  - controls
  - regulatory
  - hipaa
  - consent

indirect_cues:
  - vendor data handling
  - governed data sharing
  - review and approval controls
  - balancing and oversight
  - policy enforcement

example_tickets:
  - IDMT-8199
  - IDMT-8280

example_evidence:
  - "Data contains PII"
  - "Audit, balancing, and controls are required"
  - "Privacy and regulatory implications exist"

promote_value_streams:
  - Ensure Compliance

related_value_streams:
  - Manage Enterprise Risk

strength: 1.0
```

This is much stronger than just:
- “terms for compliance”

---

# How to use the 15 tickets efficiently

Because you only have 15 tickets, you can do a high-quality manual pass.

## Recommended review table

For each `(ticket, value_stream)` pair, capture:

| Field | Description |
|---|---|
| Ticket ID | Historical ticket |
| Value Stream | VS being reviewed |
| Direct or Indirect | Was the VS explicit or implied? |
| Why used | Short explanation |
| Supporting phrases | Exact words or strong paraphrases |
| Capability cluster | The deeper business capability |
| Confidence | High / medium / low |
| Notes | Anything ambiguous |

This gives you traceable reasoning and makes it easier to build the final capability map.

---

# What this gives you at the end

If you do this well, you will end up with:

## 1. A strong initial capability map
This becomes your runtime mapping layer.

## 2. Better candidate promotion rules
You will know which VS should be promoted when certain cues appear.

## 3. Better summary prompts
You will know which cues should be preserved in the semantic summary.

## 4. Better evaluation logic
You will understand whether a miss happened because:
- summary missed the cue
- FAISS retrieval missed the analog
- KG missed the candidate
- selector missed the final decision

---

# Why this is especially good for your current system

Your current summary-RAG is already doing:

- new-card summary generation
- FAISS analog retrieval
- analog-derived VS support
- KG candidate retrieval
- final LLM selection

What is missing is a stronger bridge between:
- business cues
- and exact value stream promotion

This historical ticket review is exactly how you build that bridge.

So yes, for your current architecture, this is one of the best next steps.

---

# Important caution

Do not overfit to only these 15 tickets.

These 15 tickets should be used to create the **first version** of the capability map, not the final eternal truth.

## Risk
If you only learn from these tickets, you may accidentally encode:
- narrow wording
- local project phrasing
- one team’s habits
- missing alternate ways of expressing the same capability

## Best practice
Use the 15 tickets to create:
- Version 1 of the map

Then refine later with:
- more tickets
- misses in production
- false positives
- new examples per VS

So the approach is good, but treat it as **bootstrapping**, not full completion.

---

# Best practical approach

## Recommended sequence

### Phase 1
Review the 15 tickets manually.

### Phase 2
For each VS, create:
- capability cluster
- direct cues
- indirect cues
- example evidence
- promoted streams

### Phase 3
Store that in:
- `summary_rag/config/capability_map.yaml`

### Phase 4
Use it at runtime to enrich candidate streams before selector.

### Phase 5
Refine it over time based on misses and incorrect promotions.

---

# Suggested final artifact set

I would create:

## 1. Capability map
Stored in repo, something like:
- `summary_rag/config/capability_map.yaml`

## 2. Review worksheet
A manual analysis sheet for the 15 tickets:
- `docs/capability_mapping_analysis.md`
or
- a spreadsheet if you want easier editing

## 3. Optional evidence library
A document that stores the best example cues per VS.

This helps later when you need to justify why a mapping exists.

---

# Example of how one review would look

## Ticket
`IDMT-8280`

## Value Stream
`Ensure Compliance`

## Why this VS was used
The ticket includes strong compliance/privacy signals related to governed data handling and control requirements.

## Direct cues
- PII
- audit
- controls
- privacy
- regulatory

## Indirect cues
- vendor/member data sharing
- governed workflow
- balancing and oversight

## Capability cluster
`compliance_privacy_audit`

## Mapping result
Promote:
- Ensure Compliance

Related but secondary:
- Manage Enterprise Risk

This is exactly the kind of reasoning you want the capability map to encode.

---

# Final recommendation

## Is it a good idea?
**Yes — definitely.**

This is one of the best ways to create the first real capability map for your system because it uses:

- real tickets
- real labels
- real business wording
- real examples of why a VS was used

## How should you think about it?
Not as:
- “collect words for each VS”

But as:
- “derive reusable business capability clusters from labeled historical examples”

That is the right mindset.

## One-line answer
Use the 15 tickets to build a **grounded first version** of the capability map by reviewing each value stream, why it was used, how it was justified, and what recurring business cues support it.
