from langchain_core.documents import Document
from extract_mentioned_docs import extract_mentioned_docs, SearchMode

LLM_RESPONSE = """
Claim Management is the capability for receiving and resolving payment requests 
for products and services rendered.

Claim Management handles the intake and resolution of payment requests to settle 
financial liability for services provided.

- Claim Orchestration, which coordinates activities and documents required to 
  process intake, management, and routing of individual payment requests.
- Claim Adjudication, which manages resolution of financial liability and payment 
  settlement according to eligible member benefits.
- Claim Support, which assists with communications, inquiries, appeals, and 
  recoveries related to claims.
- Claims Analytics and Reporting, which manages metrics, standards, reporting, 
  and analysis of claims information and processing.
- Payment Integrity Management, which manages measures to guarantee accurate 
  payments and minimize payment errors, fraud, waste, and abuse.
- Claim Acquisition Management, which manages claim receipt, imaging, translation, 
  and supporting documentation to adjudicate a claim.
- Claim Data Management, which stores, maintains, and publishes claim data 
  throughout the claim lifecycle.
- Claim Communication Management, which designs and shares information about a claim.
- Claim Edit Management, which reviews and audits claims to ensure accurate billing 
  and coding.
- Claim Inventory Management, which manages and tracks claim records through all 
  stages to resolve issues and finalize claims timely.
- Claim Appeal Management, which manages challenges to reimbursement decisions 
  related to a claim.
- Blue Exchange supports Claim Acquisition Management, Claim Communication 
  Management, and Claim Data Management.
- HIPAA 27x Availity supports Claim Acquisition Management, Claim Communication 
  Management, and Claim Data Management.
- Integrated Channels Platform supports Claim Acquisition Management, Claim 
  Communication Management, and Claim Data Management.
- NEBO and Real Time Verification Tool support Claim Acquisition Management, 
  Claim Communication Management, and Claim Data Management.
"""


def test_vector_search():
    docs = extract_mentioned_docs(LLM_RESPONSE, mode=SearchMode.VECTOR)

    assert len(docs) > 0 and isinstance(docs[0], Document)

    for doc in docs:
        print(doc.metadata)
        print(doc.metadata.get("entity_name"), "|", doc.page_content[:100])


from impact_analysis.vector_client import IDPAzureSearchRetriever
from langchain_core.documents import Document

retriever = IDPAzureSearchRetriever(
    index_name="idp_kg_data",
    search_type="semantic",
    semantic_configuration_name="default",
)

docs = retriever.invoke(input="Claim Management")

assert len(docs) > 0 and isinstance(docs[0], Document)
print(f"SUCCESS: got {len(docs)} docs")
print(docs[0].metadata)