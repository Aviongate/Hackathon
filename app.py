import os
import json
import re
import io
import base64
from typing import Any, Dict, List

import streamlit as st
from groq import Groq
import numpy as np

st.set_page_config(page_title="HaqFlow AI/RAG", page_icon="🤝", layout="wide")

# -----------------------------
# 1. Curated, source-linked KB
# -----------------------------
PROGRAMS = [
    {
        "program_id": "BISP_TALEEMI_WAZAIF",
        "name": "Benazir Taleemi Wazaif",
        "category": "Education",
        "official_source": "https://www.bisp.gov.pk/Detail/YzNlY2Q2ZGYtNjIwZS00MjNiLWFhMmEtZGM5NWNkMjZhMjQ3",
        "eligibility": [
            "Child is part of a BISP/Benazir Kafaalat beneficiary household.",
            "Child is enrolled in an eligible educational institution.",
            "Published program conditions and school attendance requirements must be verified."
        ],
        "documents": [
            "B-form/child registration document",
            "School enrollment/admission information",
            "BISP/Kafaalat beneficiary information"
        ],
        "steps": [
            "Confirm household/beneficiary status.",
            "Confirm the child's school enrollment and required attendance.",
            "Follow BISP's published enrollment/application process."
        ],
        "keywords": ["school", "student", "child", "education", "stipend", "b-form", "school fees"]
    },
    {
        "program_id": "BISP_NASHONUMA",
        "name": "Benazir Nashonuma Programme",
        "category": "Nutrition / Social Protection",
        "official_source": "https://www.bisp.gov.pk/Detail/YjAyMjI5ZDQtMTVkOC00YTNlLWE5NjctMjA1NTYwN2JhOTE3",
        "eligibility": [
            "The household must meet the program's published beneficiary conditions.",
            "The program is designed around nutrition support for eligible women and young children.",
            "Final eligibility must be verified through the official program process."
        ],
        "documents": [
            "CNIC of relevant household member",
            "Child/woman identity or registration evidence as applicable",
            "Program-specific verification documents"
        ],
        "steps": [
            "Check whether the household is covered by the relevant BISP process.",
            "Confirm maternal/child eligibility at an authorized Nashonuma facility.",
            "Complete official enrollment/verification requirements."
        ],
        "keywords": ["pregnant", "pregnancy", "mother", "nutrition", "child", "baby", "nashonuma"]
    },
    {
        "program_id": "PUNJAB_SOCIAL_WELFARE",
        "name": "Punjab Social Welfare & Protection Services",
        "category": "Social Support",
        "official_source": "https://punjab.gov.pk/social-welfare-and-protection",
        "eligibility": [
            "Eligibility depends on the specific Punjab social-welfare service.",
            "Geographic and program-specific conditions must be checked for the selected service.",
            "This knowledge-base entry is a navigation route, not an official eligibility decision."
        ],
        "documents": [
            "CNIC or identity evidence where required",
            "Documents supporting the selected service",
            "Any program-specific evidence requested by the responsible authority"
        ],
        "steps": [
            "Identify the relevant Punjab social-welfare service.",
            "Review the official service requirements.",
            "Apply through the official channel and complete verification."
        ],
        "keywords": ["social welfare", "punjab", "support", "assistance", "family", "welfare"]
    },
]

# Simple local lexical RAG: transparent and dependency-light.
# The backend/rules team can replace this with pgvector later.
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def retrieve_programs(query: str, k: int = 3) -> List[Dict[str, Any]]:
    q = set(tokenize(query))
    scored = []
    for p in PROGRAMS:
        corpus = " ".join([
            p["name"], p["category"], " ".join(p["eligibility"]),
            " ".join(p["documents"]), " ".join(p["keywords"])
        ])
        tokens = set(tokenize(corpus))
        score = len(q & tokens)
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for score, p in scored[:k] if score > 0] or [PROGRAMS[0]]

# -----------------------------
# 2. Single Groq LLM wrapper
# -----------------------------
def groq_json(system: str, user: str, image_b64: str | None = None) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    # Optional multimodal document path. If the configured Groq model does not
    # support vision, the UI falls back to text/document metadata.
    if image_b64:
        model = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# -----------------------------
# 3. Situation Intake Agent
# -----------------------------
def intake_agent(text: str) -> Dict[str, Any]:
    system = """You are HaqFlow's Situation Intake Agent.
Return ONLY valid JSON. Never decide official eligibility.
Extract only facts stated by the user. If a field is unknown, use null.
Identify missing fields needed for useful program matching.

Schema:
{
  "household_size": integer|null,
  "province": string|null,
  "district": string|null,
  "income_or_income_change": string|null,
  "life_shock": string|null,
  "children": [{"age": integer|null, "in_school": true|false|null}],
  "pregnancy_or_maternal_need": true|false|null,
  "disability": true|false|null,
  "employment_status": string|null,
  "documents_mentioned": [string],
  "missing_fields": [string],
  "next_question": string|null
}
"""
    return groq_json(system, text)

# -----------------------------
# 4. Deterministic match layer
# -----------------------------
def deterministic_match(situation: Dict[str, Any], programs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matches = []
    text = json.dumps(situation).lower()

    for p in programs:
        reasons = []
        if p["program_id"] == "BISP_TALEEMI_WAZAIF":
            if "school" in text or "student" in text or "education" in text or "child" in text:
                reasons.append("The situation mentions education/school-age children.")
        elif p["program_id"] == "BISP_NASHONUMA":
            if situation.get("pregnancy_or_maternal_need") is True or "pregnan" in text:
                reasons.append("The situation indicates a pregnancy/maternal nutrition need.")
            if "child" in text or "baby" in text:
                reasons.append("The situation mentions a child/young-child need.")
        elif p["program_id"] == "PUNJAB_SOCIAL_WELFARE":
            if situation.get("province") and "punjab" in str(situation["province"]).lower():
                reasons.append("The stated province is Punjab.")
            elif any(x in text for x in ["support", "welfare", "assistance", "income", "job loss"]):
                reasons.append("The situation indicates a general social-support/navigation need.")

        if reasons:
            matches.append({
                "program_id": p["program_id"],
                "name": p["name"],
                "match_status": "appears_to_match",
                "reasons": reasons,
                "required_documents": p["documents"],
                "verification_needed": True,
                "official_source": p["official_source"]
            })
    return matches

# -----------------------------
# 5. Grounded explanation
# -----------------------------
def grounded_explanation(situation: Dict[str, Any], matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence = []
    for m in matches:
        evidence.append({
            "program_id": m["program_id"],
            "name": m["name"],
            "source": m["official_source"],
            "reasons": m["reasons"],
            "required_documents": m["required_documents"]
        })

    system = """You are HaqFlow's Grounded Explanation Agent.
Use ONLY the supplied situation and retrieved evidence.
Do not invent eligibility thresholds, benefits, deadlines, government names, or URLs.
Use cautious language: "appears to match", "based on the published criteria",
"needs verification".
Return JSON:
{
 "summary": string,
 "matches": [
   {
    "program_id": string,
    "why_it_appears_relevant": string,
    "missing_evidence": [string],
    "verification": string,
    "next_steps": [string]
   }
 ],
 "global_warning": string
}
"""
    return groq_json(
        system,
        json.dumps({"situation": situation, "retrieved_evidence": evidence}, indent=2)
    )

# -----------------------------
# 6. Document Agent
# -----------------------------
def analyze_document(uploaded_file, situation: Dict[str, Any]) -> Dict[str, Any]:
    data = uploaded_file.getvalue()
    suffix = uploaded_file.name.lower().split(".")[-1]

    # Images can be interpreted by Groq multimodal models.
    if suffix in {"png", "jpg", "jpeg", "webp"}:
        image_b64 = base64.b64encode(data).decode("utf-8")
        system = """You are HaqFlow's Document Agent.
Interpret the uploaded document image only as evidence.
Never treat uncertain extraction as verified fact.
Return ONLY JSON:
{
 "document_type": string,
 "status": "usable"|"needs_review"|"unreadable",
 "extracted_fields": [{"field": string, "value": string|null, "confidence": number}],
 "missing_or_unclear": [string],
 "overall_confidence": number,
 "verification_note": string
}
Confidence must be between 0 and 1."""
        return groq_json(
            system,
            "Analyze this uploaded document image. Extract only visible information.",
            image_b64=image_b64
        )

    # For PDFs, keep the hackathon dependency-light. Streamlit can identify it,
    # while a user can upload a rendered page/image for multimodal extraction.
    return {
        "document_type": "PDF",
        "status": "needs_review",
        "extracted_fields": [],
        "missing_or_unclear": [
            "PDF text/image extraction is intentionally conservative in this MVP.",
            "For evidence extraction, upload a clear image/screenshot of the relevant page."
        ],
        "overall_confidence": 0.0,
        "verification_note": "Do not treat this PDF as verified evidence until fields are extracted and confirmed."
    }

# -----------------------------
# 7. Streamlit UI
# -----------------------------
st.title("🤝 HaqFlow — AI / RAG / Document Intelligence")
st.caption("Pakistan-first hackathon MVP • LLM proposes • RAG grounds • rules decide • evidence explains")

with st.sidebar:
    st.header("Configuration")
    st.write("Add `GROQ_API_KEY` in Streamlit Cloud → Settings → Secrets.")
    st.write("Optional secret: `GROQ_MODEL`")
    st.write("Optional vision secret: `GROQ_VISION_MODEL`")
    st.divider()
    st.info("Demo data is curated and source-linked. This prototype does not make official eligibility decisions.")

tab1, tab2, tab3 = st.tabs(["1. Situation → RAG", "2. Document Analyzer", "3. Program Knowledge Base"])

with tab1:
    st.subheader("Tell HaqFlow what happened")
    example = "My father lost his daily-wage job. We are six people. My sister is in school and my mother is pregnant. We live in Punjab."
    user_text = st.text_area("Situation", value=example, height=130)

    if st.button("Analyze situation", type="primary"):
        try:
            with st.spinner("Extracting situation..."):
                situation = intake_agent(user_text)
            st.session_state["situation"] = situation

            query = user_text + " " + json.dumps(situation)
            retrieved = retrieve_programs(query)
            st.session_state["retrieved"] = retrieved

            matches = deterministic_match(situation, retrieved)
            st.session_state["matches"] = matches

            if matches:
                with st.spinner("Generating grounded explanation..."):
                    explanation = grounded_explanation(situation, matches)
            else:
                explanation = {
                    "summary": "No deterministic match was found in the small demo knowledge base.",
                    "matches": [],
                    "global_warning": "Expand the curated program set or route the case to human review."
                }
            st.session_state["explanation"] = explanation
        except Exception as e:
            st.error(str(e))

    if "situation" in st.session_state:
        st.markdown("### UserSituation JSON")
        st.json(st.session_state["situation"])

    if "retrieved" in st.session_state:
        st.markdown("### Retrieved verified context")
        for p in st.session_state["retrieved"]:
            with st.expander(f'{p["name"]} — {p["category"]}'):
                st.write("**Program ID:**", p["program_id"])
                st.write("**Eligibility conditions:**")
                for x in p["eligibility"]:
                    st.write("•", x)
                st.write("**Required documents:**")
                for x in p["documents"]:
                    st.write("•", x)
                st.write("**Application steps:**")
                for x in p["steps"]:
                    st.write("•", x)
                st.markdown(f'**Official source:** [{p["official_source"]}]({p["official_source"]})')

    if "explanation" in st.session_state:
        st.markdown("### Grounded explanation")
        e = st.session_state["explanation"]
        st.success(e.get("summary", ""))
        for m in e.get("matches", []):
            st.markdown(f'#### {m.get("program_id")}')
            st.write("**Why it appears relevant:**", m.get("why_it_appears_relevant", ""))
            st.write("**Missing evidence:**")
            for x in m.get("missing_evidence", []):
                st.write("•", x)
            st.write("**Next steps:**")
            for x in m.get("next_steps", []):
                st.write("•", x)
            st.write("**Verification:**", m.get("verification", ""))
        st.warning(e.get("global_warning", ""))

with tab2:
    st.subheader("Upload a sample document")
    st.caption("Use synthetic/demo documents. Clear JPG/PNG images are analyzed by the multimodal Groq model. PDFs are conservatively flagged for review in this simple MVP.")
    uploaded = st.file_uploader("Document", type=["png", "jpg", "jpeg", "webp", "pdf"])

    if uploaded and st.button("Analyze document"):
        situation = st.session_state.get("situation", {})
        try:
            with st.spinner("Analyzing evidence..."):
                result = analyze_document(uploaded, situation)
            st.session_state["document_result"] = result
        except Exception as e:
            st.error(str(e))

    if "document_result" in st.session_state:
        st.markdown("### Document JSON")
        st.json(st.session_state["document_result"])
        if st.session_state["document_result"].get("overall_confidence", 0) < 0.75:
            st.warning("Low-confidence evidence must be reviewed/confirmed; it is not treated as verified fact.")

with tab3:
    st.subheader("Curated program knowledge base")
    st.write(f"{len(PROGRAMS)} source-linked programs are included for the MVP.")
    st.dataframe(
        [
            {
                "Program ID": p["program_id"],
                "Program": p["name"],
                "Category": p["category"],
                "Official Source": p["official_source"]
            }
            for p in PROGRAMS
        ],
        use_container_width=True,
        hide_index=True
    )
    st.info("For the final team merge, the Data/KB member can expand this list to 5–10 verified programs for the selected province and demo scenario.")

st.divider()
st.caption("HaqFlow prototype: matches are navigation signals, not official eligibility determinations.")
