# Theme & Business Needs Generation — Condense Contract and Prompt Templates

## Purpose

This document defines the final v1 contract for the condense step and the downstream prompt inputs/templates used for Theme Description and Business Needs generation.

The current IDMT ticket does **not** already have a Theme Description or Business Needs artifact. Those are generated after the SME confirms the recommended Value Streams. Therefore, the generation layer must use only the current ticket source material: the idea card when available, the Jira description, and supported attachments.

---

## 1. Condense Step Design

The condense step performs structured extraction in a **single LLM call**. It returns compact summary fields and generation signals together. This avoids running a second extraction pass over the same idea card or attachment text.

### Source priority

1. If an attachment named or tagged like `idea_card.ppt` / `idea_card.pptx` exists, use it as the primary business context.
2. If the idea card is missing, fall back to the Jira ticket description plus the top four supported attachments.
3. Supported attachment priority: PowerPoint, PDF, Word document.

### Why summary alone is not enough

The generated summary is useful for routing, retrieval, Value Stream selection, and general context. However, Theme Description and Business Needs need more detailed signals such as market segments, funding model, dependencies, reporting, rules, training, and operational impacts. These are often lost in a short summary, so they are extracted as structured generation signals during the same condense call.

---

## 2. Final v1 Condense Output Contract

`generationSignals` is an object. Each field is an array of lightweight evidence objects. Arrays are used because the idea card or attachments may contain multiple signals per category.

Separate `sourceSnippets` are **not required in v1**. They can be added later if deeper citation or SME evidence review is required. For now, each generation signal carries lightweight traceability using `source` and `sourceSection`.

```json
{
  "idmtTicketId": "IDMT-####",
  "idmtTicketTitle": "<ticket title>",

  "sourceContext": {
    "sourceMode": "idea_card | fallback",
    "ideaCardFound": true,
    "jiraDescriptionUsed": true,
    "attachmentsUsed": ["idea_card.pptx"]
  },

  "summaryFields": {
    "generatedSummary": "<concise business summary>",
    "businessProblem": "<business pain point / gap>",
    "businessCapability": "<desired capability or outcome>",
    "keyTerms": ["<term 1>", "<term 2>"],
    "stakeholders": ["<stakeholder 1>", "<stakeholder 2>"],
    "systemsAndProducts": ["<system/product 1>", "<system/product 2>"]
  },

  "generationSignals": {
    "marketSegments": [
      {
        "text": "<market / segment signal>",
        "source": "idea_card | jira_description | attachment",
        "sourceSection": "<section name if available>"
      }
    ],
    "fundingModelSignals": [],
    "marketOpportunity": [],
    "businessSolutionObjectives": [],
    "valueProposition": [],
    "estimatedCosts": [],
    "estimatedBenefits": [],
    "dependencies": [],
    "resourcesNeeded": [],
    "digitalExperienceSignals": [],
    "productAvailabilitySignals": [],
    "planSignals": [],
    "networkSignals": [],
    "productPairingSignals": [],
    "businessRules": [],
    "operationalSignals": [],
    "reportingSignals": [],
    "trainingSignals": [],
    "notes": []
  }
}
```

### Condense rules

- Return summary fields and generation signals in one structured extraction call.
- Keep generation signals optional. If a signal is not present in the idea card, Jira description, or supported attachments, return an empty array.
- Do not invent product availability, plans, networks, funding models, product pairing, reporting, training, or business rules.
- Do not add placeholder values such as `TBD` inside extracted fields.
- Keep signal text concise and source-grounded.
- `sourceContext` is kept only to indicate which extraction path was used. It is not meant to be a full audit record.
- Separate `sourceSnippets` are deferred for v2 to reduce output size and condense latency.

---

## 3. Field Usage by Generation Step

### 3.1 Value Stream Selection

Used fields:

```text
idmtTicketId
idmtTicketTitle

summaryFields.generatedSummary
summaryFields.businessProblem
summaryFields.businessCapability
summaryFields.keyTerms
summaryFields.stakeholders
summaryFields.systemsAndProducts

generationSignals.marketOpportunity
generationSignals.businessSolutionObjectives
generationSignals.valueProposition
generationSignals.dependencies
generationSignals.digitalExperienceSignals
```

Purpose:

```text
These fields provide the business intent and context used to retrieve candidate Value Streams from idp_idmt_data and prepare the LLM review pool. Candidate blocks are grouped as Semantic + Historic, Historic-only, and Semantic-only.
```

---

### 3.2 Stage Prediction

Stage prediction runs after the SME confirms the final Value Streams.

Used fields:

```text
idmtTicketId
idmtTicketTitle

approved valueStreamId
approved valueStreamName
valueStreamDescription

summaryFields.generatedSummary
summaryFields.businessProblem
summaryFields.businessCapability
summaryFields.keyTerms
summaryFields.stakeholders
summaryFields.systemsAndProducts

generationSignals.businessSolutionObjectives
generationSignals.dependencies
generationSignals.digitalExperienceSignals
generationSignals.operationalSignals
generationSignals.reportingSignals
generationSignals.businessRules
generationSignals.notes

Cosmos governed stage catalogue:
- stageId
- stageName
- stageDescription
- valueStreamId
```

Expected output:

```json
{
  "selectedStages": [
    {
      "stageId": "<stage id>",
      "stageName": "<approved stage name>",
      "rank": 1,
      "reason": "<why this stage applies>",
      "evidence": "<source-grounded explanation>",
      "validationStatus": "valid | invalid"
    }
  ]
}
```

Rules:

- Select only from the governed Stage list mapped to the approved Value Stream.
- Do not invent new Stage names.
- Use the approved Value Stream description and stage descriptions as the governed catalogue boundary.

---

## 4. Theme Description Generation

Theme Description is generated only after human-in-the-loop Value Stream approval. It runs in parallel with Stage Prediction.

The current ticket does not have a Theme Description yet. The prompt must not depend on an existing Theme Description for the current ticket.

### 4.1 Fields used in Theme Description prompt

```text
idmtTicketId
idmtTicketTitle

approved valueStreamId
approved valueStreamName
valueStreamDescription

summaryFields.generatedSummary
summaryFields.businessProblem
summaryFields.businessCapability
summaryFields.keyTerms
summaryFields.stakeholders
summaryFields.systemsAndProducts

generationSignals.marketSegments
generationSignals.fundingModelSignals
generationSignals.marketOpportunity
generationSignals.businessSolutionObjectives
generationSignals.valueProposition
generationSignals.estimatedBenefits
generationSignals.dependencies
generationSignals.resourcesNeeded
generationSignals.digitalExperienceSignals
generationSignals.productAvailabilitySignals
generationSignals.planSignals
generationSignals.networkSignals
generationSignals.productPairingSignals
generationSignals.operationalSignals
generationSignals.reportingSignals
generationSignals.notes
```

### 4.2 Theme Description prompt template

```text
You are generating a Jira Theme Description for an approved Value Stream.

The Theme Description is a new artifact for the current IDMT ticket. Do not assume an existing Theme Description exists for this ticket.

Use only the provided ticket context, approved Value Stream, summary fields, and extracted generation signals.

Approved Value Stream:
- valueStreamId: {{valueStreamId}}
- valueStreamName: {{valueStreamName}}
- valueStreamDescription: {{valueStreamDescription}}

Ticket Context:
- idmtTicketId: {{idmtTicketId}}
- idmtTicketTitle: {{idmtTicketTitle}}
- generatedSummary: {{generatedSummary}}
- businessProblem: {{businessProblem}}
- businessCapability: {{businessCapability}}
- keyTerms: {{keyTerms}}
- stakeholders: {{stakeholders}}
- systemsAndProducts: {{systemsAndProducts}}

Generation Signals:
- marketSegments: {{marketSegments}}
- fundingModelSignals: {{fundingModelSignals}}
- marketOpportunity: {{marketOpportunity}}
- businessSolutionObjectives: {{businessSolutionObjectives}}
- valueProposition: {{valueProposition}}
- estimatedBenefits: {{estimatedBenefits}}
- dependencies: {{dependencies}}
- resourcesNeeded: {{resourcesNeeded}}
- digitalExperienceSignals: {{digitalExperienceSignals}}
- productAvailabilitySignals: {{productAvailabilitySignals}}
- planSignals: {{planSignals}}
- networkSignals: {{networkSignals}}
- productPairingSignals: {{productPairingSignals}}
- operationalSignals: {{operationalSignals}}
- reportingSignals: {{reportingSignals}}
- notes: {{notes}}

Generation Instructions:
1. Generate a high-level Theme Description for the approved Value Stream.
2. Include optional product availability fields only when source evidence exists.
3. Skip unavailable fields. Do not write TBD for missing fields.
4. Do not invent plans, funding models, networks, product pairing exclusions, dates, or product availability details.
5. Keep this Theme-level. Do not generate detailed stage-level requirements here.
6. Use the approved Value Stream as the boundary for the Theme.
7. Ground the output in the idea card, Jira description, attachments, summary fields, and extracted generation signals.
```

### 4.3 Theme Description output template

```text
Theme Description and Product Availability:

<High-level paragraph explaining the objective of this Theme for the approved Value Stream.
Mention the business context, initiative/product, impacted stakeholders, and expected outcome.
Keep this at Theme level, not detailed requirement level.>


Product Availability:
<Include this section only if product availability signals exist. Skip empty fields.>

Go live: <only if present in source>
Plans: <only if present in source>
Market Segments: <only if present in source>
Funding Model: <only if present in source>
Networks Impacted: <only if present in source>
Product Structure and Pairing Matrix: <only if present in source>
Product Pairing Exclusions: <only if present in source>


<Initiative / Product Name>:

<Short overview of the initiative and major business outcomes/features to be delivered.>


Key Features:
- <feature or outcome supported by source>
- <feature or outcome supported by source>
- <feature or outcome supported by source>


Digital Experience:
<Include only if digital experience signals exist.>
- <member/provider/employer/user digital experience impact>


Integration / Operational Capabilities:
<Include only if source mentions systems, integrations, reporting, operations, vendors, workflows, claims, servicing, or platform impacts.>
- <integration / operational capability>
```

### 4.4 Theme Description structured output

```json
{
  "themeDescription": {
    "themeOverview": "<high-level theme paragraph>",
    "productAvailability": {
      "goLive": "<optional>",
      "plans": ["<optional>"],
      "marketSegments": ["<optional>"],
      "fundingModel": ["<optional>"],
      "networksImpacted": ["<optional>"],
      "productStructureAndPairingMatrix": "<optional>",
      "productPairingExclusions": ["<optional>"]
    },
    "initiativeOverview": "<initiative / product overview>",
    "keyFeatures": ["<feature 1>", "<feature 2>"],
    "digitalExperience": ["<optional>"],
    "integrationOperationalCapabilities": ["<optional>"]
  }
}
```

---

## 5. Business Needs Generation

Business Needs generation runs after Stage Prediction resolves. Business Needs must explain the selected stages in the context of the current ticket.

Business Needs should be detailed, stage-specific, and grouped by selected Value Stage and Business Product Feature.

### 5.1 Fields used in Business Needs prompt

```text
idmtTicketId
idmtTicketTitle

approved valueStreamId
approved valueStreamName
valueStreamDescription

selected stages:
- stageId
- stageName
- stageDescription

summaryFields.generatedSummary
summaryFields.businessProblem
summaryFields.businessCapability
summaryFields.keyTerms
summaryFields.stakeholders
summaryFields.systemsAndProducts

generationSignals.businessSolutionObjectives
generationSignals.dependencies
generationSignals.resourcesNeeded
generationSignals.digitalExperienceSignals
generationSignals.businessRules
generationSignals.operationalSignals
generationSignals.reportingSignals
generationSignals.trainingSignals
generationSignals.notes
```

### 5.2 Business Needs prompt template

```text
You are generating Business Needs for an approved Value Stream and selected Value Stages.

Business Needs are stage-specific. Generate needs only for the selected stages provided in the prompt.

Approved Value Stream:
- valueStreamId: {{valueStreamId}}
- valueStreamName: {{valueStreamName}}
- valueStreamDescription: {{valueStreamDescription}}

Selected Stages:
{{selectedStagesWithDescriptions}}

Ticket Context:
- idmtTicketId: {{idmtTicketId}}
- idmtTicketTitle: {{idmtTicketTitle}}
- generatedSummary: {{generatedSummary}}
- businessProblem: {{businessProblem}}
- businessCapability: {{businessCapability}}
- keyTerms: {{keyTerms}}
- stakeholders: {{stakeholders}}
- systemsAndProducts: {{systemsAndProducts}}

Generation Signals:
- businessSolutionObjectives: {{businessSolutionObjectives}}
- dependencies: {{dependencies}}
- resourcesNeeded: {{resourcesNeeded}}
- digitalExperienceSignals: {{digitalExperienceSignals}}
- businessRules: {{businessRules}}
- operationalSignals: {{operationalSignals}}
- reportingSignals: {{reportingSignals}}
- trainingSignals: {{trainingSignals}}
- notes: {{notes}}

Generation Instructions:
1. Group Business Needs by selected Value Stage.
2. Under each Value Stage, group requirements by Business Product Feature.
3. Use numbered requirements.
4. Include Notes only when supported by source context.
5. Include Dependencies only when supported by source context.
6. Include Business Rules only when supported by source context.
7. Include Operational Training only when training/readiness signals exist.
8. Include Operational Reporting only when reporting, metrics, dashboard, or tracking signals exist.
9. Do not include Assumptions for now.
10. Do not invent unsupported requirements.
11. Every selected stage should have at least one meaningful business need unless there is insufficient source support. If insufficient, flag it for SME review.
```

### 5.3 Business Needs output template

```text
Value Stage: <Selected Stage Name>

Business Product Feature: <Feature / Scope Area>

1. <Detailed business need grounded in the ticket context>
2. <Detailed business need grounded in the ticket context>
3. <Detailed business need grounded in the ticket context>

   Note:
   <Only include if the source has a clarification, exception, validation note, or important detail.>

   Dependency:
   <Only include if the source mentions a system, vendor, team, upstream/downstream dependency, or related capability.>

   Business Rule:
   <Only include if the source mentions a business rule, policy, eligibility rule, routing rule, configuration rule, or operational rule.>


Business Product Feature: <Next Feature / Scope Area>

1. <Detailed business need>
2. <Detailed business need>


Operational Training:
<Include only if training/readiness signals exist.>
1. <training need>
2. <readiness need>


Operational Reporting:
<Include only if reporting/dashboard/metrics/data-tracking signals exist.>
1. <reporting need>
2. <metrics / dashboard / data requirement>
3. <data dependency or reporting rule>
```

### 5.4 Business Needs structured output

```json
{
  "businessNeeds": [
    {
      "stageId": "<stage id>",
      "stageName": "<stage name>",
      "businessProductFeatures": [
        {
          "featureName": "<feature / scope area>",
          "needs": [
            {
              "number": 1,
              "text": "<business need>"
            }
          ],
          "notes": ["<optional note>"],
          "dependencies": ["<optional dependency>"],
          "businessRules": ["<optional business rule>"]
        }
      ],
      "operationalTraining": ["<optional training/readiness need>"],
      "operationalReporting": ["<optional reporting/metrics/data need>"],
      "validationStatus": "valid | needs_review"
    }
  ]
}
```

---

## 6. Final Theme Package

After Value Stream approval, Stage Prediction, Theme Description, Business Needs, L2, and L3 generation resolve, the system packages one Theme per approved Value Stream.

```json
{
  "themePackage": {
    "idmtTicketId": "IDMT-####",
    "themeTitle": "<IDMT ticket title> - <Value Stream name>",

    "valueStreamId": "<approved VS id>",
    "valueStreamName": "<approved VS name>",

    "selectedStages": [],
    "themeDescription": {},
    "businessNeeds": [],
    "l2Capabilities": [],
    "l3Capabilities": [],

    "validationStatus": "valid | needs_review",
    "warnings": []
  }
}
```

Final rule:

```text
The final Theme package is still a recommendation. It must be shown to the SME for review before Jira Theme, Epic, Business Needs, or capability artifacts are created or updated.
```
