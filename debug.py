You are performing Level 3 business capability classification for multiple Value Stream Stages that share the same Theme context.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use Theme Business Needs as the primary shared business evidence.
Use Theme Description as supporting context that can clarify the Business Needs, but do not use it to introduce unsupported business functions.

Classify each supplied Value Stream Stage independently using only:
- the shared Theme Business Needs,
- the shared Theme Description,
- that Stage's governed metadata,
- that Stage's supplied candidate L3 capabilities.

EVIDENCE

Theme Business Needs describes the business outcomes, requirements, activities, responsibilities, and functions that need to be delivered.

Theme Description may clarify the scope and intent of those Business Needs.

Each Stage's name, description, entrance criteria, and exit criteria define that Stage's business-process boundary.

Stage metadata is a boundary and disambiguation mechanism only.
Stage metadata must NOT by itself establish that an L3 capability is required.

Positive evidence for selecting an L3 must originate from Theme Business Needs or from Theme Description when it clearly clarifies those Business Needs.

For each candidate L3:
- capability_id is the exact identifier to return when selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is a supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer business meaning from capability_id.

CLASSIFICATION

For each Stage independently:

1. Identify the concrete business functions required by Theme Business Needs.

2. Use Theme Description to clarify the meaning, scope, or intent of those functions.

3. Use the Stage name, Stage description, entrance criteria, and exit criteria to determine whether each business function falls within this Stage boundary.

4. Evaluate EVERY supplied candidate L3 independently.

Select a candidate when its business function is:

- explicitly required by the Theme context, OR

- strongly semantically entailed by a concrete requirement, outcome, activity, responsibility, or intended business behavior in the Theme context.

Strong semantic entailment means that the stated business requirement would reasonably remain incomplete or uncovered without the business function represented by that capability.

The exact capability name or terminology does not need to appear in the Theme text.

The Theme also does not need to describe the candidate's complete capability definition.
A candidate may be selected when the Theme clearly requires a meaningful business function contained within that capability's semantic definition.

Selecting one capability does not automatically justify selecting another capability.

If multiple distinct business functions are genuinely supported, return all of the supported candidates.

DO NOT SELECT A CANDIDATE ONLY BECAUSE:

- it belongs to the supplied Stage,
- it shares words or terminology with the Theme,
- it is generally related to the Theme,
- it would be useful in the same workflow,
- it commonly supports another selected capability,
- it is a prerequisite,
- it is enabling,
- it is upstream or downstream,
- it is adjacent to a supported capability,
- it is a typical function performed within that Stage.

A capability must have its own concrete semantic support from the Theme context.

Do not use one Stage's metadata or candidates as evidence for another Stage.

FINAL PRECISION CHECK

Before producing the final answer, review EVERY selected candidate again.

For each selected candidate, ask:

"What specific business requirement, outcome, activity, responsibility, or intended behavior in the Theme context requires this capability?"

If there is no concrete answer grounded in Theme Business Needs or Theme Description, REMOVE that candidate.

Do not keep a candidate merely because its capability description appears compatible with the Stage.

When two candidates overlap semantically, keep both only when the Theme context provides distinct business evidence supporting both functions.

Prefer complete coverage of genuinely supported business functions, but do not expand beyond what the Theme actually requires.

OUTPUT RULES

Only return capability_id values supplied for that Stage.

If no candidate is supported for a Stage, return an empty list.

Return exactly one result for every supplied stage_id.

Return JSON only in this exact structure:

{
  "stages": [
    {
      "stage_id": "VSS000123",
      "l3": [
        "CAP00000123",
        "CAP00000456"
      ]
    },
    {
      "stage_id": "VSS000456",
      "l3": []
    }
  ]
}

Do not return reasons, explanations, confidence scores, Markdown, or additional fields.
