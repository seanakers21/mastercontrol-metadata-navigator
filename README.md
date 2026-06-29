# MasterControl Quality Knowledge Navigator

A Streamlit proof of concept for searching fictional pharmaceutical quality-event metadata and effective controlled-document metadata using natural-language engineering questions.

The application supports two distinct discovery domains:

- **Historical quality experience:** deviations, variances, events, investigations, CAPAs, equipment failures, recorded root causes, and actions.
- **Controlled guidance:** effective SOPs, work instructions, forms, and validation documents.

Historical records and controlled documents are deliberately presented separately. A historical record is evidence of what was recorded previously; it is not an approved instruction or an automated recurrence determination.

## Important limitations

This repository is a demonstration using fictional data. It is not validated for GxP production use and does not connect to MasterControl.

The prototype must not be used to:

- Make or approve a quality decision.
- Determine recurrence, root cause, reportability, CAPA effectiveness, or product impact.
- Replace an investigation or Quality-unit review.
- Modify, approve, close, or create a regulated record.
- Treat an indexed copy as the authoritative record.

MasterControl is intended to remain the validated system of record. Users must open and verify each authoritative source record before relying on it.

## Current capabilities

- Hybrid semantic and keyword search across fictional historical events.
- Exact asset-ID and keyword matching with engineering synonyms.
- Semantic search and cross-encoder reranking of effective document metadata.
- Filters for site, record type, department, status, and event year.
- Expandable evidence cards with source record IDs and direct links.
- Recorded root-cause, action, and recurrence fields clearly attributed to source metadata.
- Separate potentially related event and applicable controlled-document sections.
- Explicit uncertainty and decision-support warnings.

## Project structure

```text
app.py                  Streamlit interface and document semantic index
data_service.py         File-backed data provider boundary
search_service.py       Event filtering, lexical retrieval, and result fusion
mock_documents.json     Fictional controlled-document metadata
mock_events.json        Fictional quality-event and CAPA metadata
tests/                  Retrieval and filtering tests
```

## Run locally

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The embedding and reranking models may be downloaded the first time the application starts.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Intended MasterControl integration

`data_service.py` is the initial provider boundary. For production, replace the JSON-backed functions with an authenticated read-only adapter while retaining a normalized internal schema.

```text
MasterControl API
  -> service account / user authorization
  -> permission-filtered read-only extraction
  -> schema validation and synchronization audit
  -> versioned search index
  -> source-cited results
  -> authoritative MasterControl record
```

The adapter should capture source record ID, record type, lifecycle status, revision or version, timestamps, site, owning organization, equipment identifiers, approved/closed narrative fields, relationships, source URL, and source-system update timestamp. Field availability must follow the requesting user's authorization; indexing must not become a way to bypass MasterControl permissions.

Recommended synchronization behavior:

1. Use stable MasterControl record identifiers.
2. Perform incremental, read-only synchronization where the API permits it.
3. Reject or quarantine records that fail schema validation.
4. record synchronization time, outcome, source update time, and index version.
5. Remove or restrict indexed access when source permissions or lifecycle states change.
6. Preserve enough lineage to reproduce which indexed metadata supported a displayed result.

API paths, authentication, available fields, rate limits, and permission semantics must be confirmed against the licensed MasterControl product and organizational configuration before implementation.

## Production compliance work

Deployment in a pharmaceutical environment requires review under the organization's quality system. At minimum, plan for:

- Approved intended use and explicitly prohibited uses.
- GxP and electronic-record applicability assessment.
- Risk assessment and validation strategy proportionate to intended use.
- Requirements, configuration specifications, and traceable testing.
- Role-based access aligned with MasterControl.
- Audit logging, retention, security, backup, and recovery controls.
- Search acceptance criteria using an approved representative query set.
- Tests for false positives, false negatives, permissions, stale data, and unavailable sources.
- Version control for code, models, synonym rules, schemas, and indexes.
- Controlled model/index updates, change control, and periodic performance review.
- User training and procedures for source verification and Quality escalation.

The current system generates no free-form AI conclusion. It ranks and displays attributed metadata. Any future generated synthesis should cite supporting record IDs at the claim level, distinguish source facts from inference, expose uncertainty, and remain subject to authorized human review.

## Mock event schema

The fictional event data includes record type and ID, status, dates, site and area, equipment identity, problem statement, failure mode, immediate action, recorded root-cause category and confirmed root cause, corrective/preventive actions, recurrence classification, related records, keywords, and authoritative URL.

Production mappings should prefer approved structured fields. Sensitive personal information and unrestricted narrative must not be indexed unless its necessity, authorization, and controls are explicitly approved.
