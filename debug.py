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

Positive evidence for selecting an L3 must originate from Theme Business Needs or Theme Description when it clearly clarifies those Business Needs.

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

3. Use the Stage name, Stage description, entrance criteria, and exit criteria only to determine whether those functions fall within this Stage boundary.

4. Evaluate EVERY supplied candidate L3 independently.

Select a candidate only when the Theme context provides concrete semantic evidence that the candidate's business function is required.

A candidate may be selected when its business function is:

- explicitly required by the Theme context, OR

- strongly semantically entailed by a concrete requirement, outcome, activity, responsibility, or intended business behavior in the Theme context.

Strong semantic entailment means the Theme requirement would reasonably remain incomplete or insufficiently covered without that candidate's business function.

The exact capability name or terminology does not need to appear in the Theme text.

The Theme does not need to describe the candidate's entire capability definition.
A candidate may be selected when the Theme clearly requires a meaningful business function contained within that capability's semantic definition.

However, partial semantic overlap alone is NOT sufficient.

A capability is not relevant merely because some portion of its description could be associated with the Theme.
There must be a concrete Theme requirement that actually requires the business function represented by the capability.

Selecting one capability does not automatically justify selecting another capability.

If multiple distinct business functions are genuinely required, return all supported candidates.

DO NOT SELECT A CANDIDATE ONLY BECAUSE:

- it belongs to the supplied Stage,
- it shares words or terminology with the Theme,
- it appears compatible with the Theme,
- it is generally related to the Theme,
- it would commonly occur in the same business process,
- it would be useful for implementing another selected capability,
- it commonly supports another selected capability,
- it is a prerequisite,
- it is enabling,
- it is upstream or downstream,
- it is adjacent to a supported capability,
- it is a typical function performed within that Stage.

Compatibility is not evidence.
Association is not evidence.
Stage membership is not evidence.

A candidate must have its own concrete functional support from the Theme context.

Do not use one Stage's metadata or candidates as evidence for another Stage.

FINAL PRECISION CHECK

Before producing the final answer, review EVERY selected candidate again.

For each selected candidate:

1. Identify the specific Theme requirement, outcome, activity, responsibility, or intended business behavior that supports it.

2. Ask:

"Does that Theme evidence genuinely REQUIRE this capability's business function, or is the capability merely compatible, related, adjacent, or commonly associated with the requirement?"

3. Keep the candidate only if the answer is clearly that the business function is required or strongly entailed.

If there is no concrete Theme evidence you can point to, REMOVE the candidate.

If the relationship is only plausible, compatible, related, adjacent, enabling, upstream, downstream, prerequisite, or commonly associated, REMOVE the candidate.

When two selected candidates overlap semantically, keep both only when the Theme context provides distinct functional evidence supporting both.

If one Theme requirement is fully explained by one capability, do not add a second overlapping capability unless the Theme contains additional evidence for that second function.

When uncertain between selecting and excluding a candidate, exclude it unless the Theme contains concrete functional evidence.

Prefer complete coverage of genuinely required business functions, but do not expand beyond what the Theme actually requires.

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
