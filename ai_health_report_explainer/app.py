import re
from typing import Dict, List

import streamlit as st
from transformers import pipeline


APP_TITLE = "AI Health Report Explainer"
MODEL_NAME = "google/flan-t5-base"

SECTION_TITLES = {
    "summary": "Plain-language Summary",
    "risk_factors": "Key Risk Factors",
    "next_steps": "Suggested Next Steps",
    "limitations": "Important Warning or Limitation",
    "concern": "Concern Level",
}

AUDIENCE_GUIDANCE = {
    "general public": (
        "Use everyday wording. Avoid technical terms when possible. Focus on the main idea and practical meaning."
    ),
    "patient": (
        "Use supportive patient-centered wording. Focus on what the information may mean for a person "
        "and what they could ask a qualified health professional."
    ),
    "public health student": (
        "Use public health language when helpful. Mention study design, population-level risk, confounding, "
        "association versus causation, and communication implications when relevant."
    ),
}

DETAIL_GUIDANCE = {
    "brief": "Write 1 to 2 short sentences or 2 to 3 short bullets.",
    "detailed": "Write 3 to 5 sentences or 4 to 5 bullets with more explanation.",
}


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )


def clean_input(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3500]


def build_section_prompt(text: str, audience: str, detail_level: str, section_key: str) -> str:
    audience_guidance = AUDIENCE_GUIDANCE[audience]
    detail_guidance = DETAIL_GUIDANCE[detail_level]
    section_tasks = {
        "summary": "Explain the main message in clear language for this reader.",
        "risk_factors": (
            "List the main risk factors, exposures, groups, or behaviors mentioned in the text. "
            "If none are stated, say that they are not clearly stated."
        ),
        "next_steps": (
            "Suggest safe, conservative next steps for understanding the information. "
            "Do not diagnose disease or recommend treatment changes."
        ),
        "limitations": (
            "State the most important limitation, uncertainty, or caution about interpreting the text."
        ),
        "concern": (
            "Choose one concern level: low concern, medium concern, or high concern. "
            "Then give one short reason based only on the text."
        ),
    }

    return f"""
You are an AI assistant that explains medical and public health information.
Do not diagnose disease. Do not recommend medication changes. Do not give emergency medical instructions beyond telling users to seek urgent professional help when appropriate.

Audience: {audience}
Audience-specific style: {audience_guidance}
Detail level: {detail_level}
Detail instruction: {detail_guidance}

Task: {section_tasks[section_key]}

Health text:
{text}
""".strip()


def generate_section(
    generator,
    text: str,
    audience: str,
    detail_level: str,
    section_key: str,
) -> str:
    prompt = build_section_prompt(text, audience, detail_level, section_key)
    max_new_tokens = 170 if detail_level == "detailed" else 90
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        truncation=True,
        do_sample=False,
        num_beams=4,
    )
    return result[0]["generated_text"].strip()


def generate_explanation(text: str, audience: str, detail_level: str) -> Dict[str, str]:
    generator = load_model()
    return {
        section_key: generate_section(generator, text, audience, detail_level, section_key)
        for section_key in SECTION_TITLES
    }


def split_sections(model_output: str) -> Dict[str, str]:
    labels = list(SECTION_TITLES.values())
    pattern = "|".join(re.escape(label) for label in labels)
    matches = list(re.finditer(rf"({pattern})\s*:", model_output, flags=re.IGNORECASE))

    sections = {key: "" for key in SECTION_TITLES}
    if not matches:
        sections["summary"] = model_output.strip()
        return sections

    label_to_key = {value.lower(): key for key, value in SECTION_TITLES.items()}
    for index, match in enumerate(matches):
        label = match.group(1).lower()
        key = label_to_key.get(label)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(model_output)
        if key:
            sections[key] = model_output[start:end].strip(" \n:-")

    return sections


def keyword_fallback(text: str, audience: str = "general public", detail_level: str = "brief") -> Dict[str, str]:
    lower = text.lower()
    high_terms = ["emergency", "severe", "death", "fatal", "stroke", "heart attack", "outbreak"]
    medium_terms = ["risk", "increased", "high blood pressure", "diabetes", "asthma", "pollution"]

    if any(term in lower for term in high_terms):
        concern = "high concern - the text includes potentially serious health outcomes."
    elif any(term in lower for term in medium_terms):
        concern = "medium concern - the text discusses meaningful health risks."
    else:
        concern = "low concern - the text does not clearly describe immediate serious risk."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:2]).strip() or "The text discusses a health-related topic."
    factors = find_risk_terms(lower)

    next_steps = (
        "Use this explanation as a starting point, review the original source, "
        "and discuss personal medical questions with a qualified health professional."
    )
    if audience == "patient":
        next_steps = (
            "Write down questions from this text and discuss them with a qualified health professional, "
            "especially before changing medication, diet, or care plans."
        )
    elif audience == "public health student":
        next_steps = (
            "Review the study design, population, comparison group, measured outcomes, and possible confounders "
            "before drawing public health conclusions."
        )

    if detail_level == "detailed":
        next_steps += (
            " Also check whether the text reports absolute risk, who was included in the data, "
            "and whether the findings apply to the population you care about."
        )

    return {
        "summary": summary,
        "risk_factors": "\n".join(f"- {factor}" for factor in factors) if factors else "Not clearly stated.",
        "next_steps": next_steps,
        "limitations": (
            "This automated explanation may miss details, and the original text may not prove cause and effect."
        ),
        "concern": concern,
    }


def find_risk_terms(lower_text: str) -> List[str]:
    candidates = [
        "smoking",
        "sleep deprivation",
        "high blood pressure",
        "air pollution",
        "diabetes",
        "obesity",
        "age",
        "physical inactivity",
        "poor diet",
        "stress",
        "infection",
        "exposure",
    ]
    return [term for term in candidates if term in lower_text]


def normalize_sections(
    sections: Dict[str, str],
    source_text: str,
    audience: str,
    detail_level: str,
) -> Dict[str, str]:
    fallback = keyword_fallback(source_text, audience, detail_level)
    normalized = {}
    for key in SECTION_TITLES:
        value = sections.get(key, "").strip()
        normalized[key] = value if is_usable_model_text(value) else fallback[key]
    normalized["summary"] = tailor_summary(normalized["summary"], audience, detail_level)
    normalized["risk_factors"] = tailor_risk_factors(normalized["risk_factors"], source_text, audience, detail_level)
    normalized["next_steps"] = tailored_next_steps(audience, detail_level)
    normalized["limitations"] = tailored_limitations(source_text, audience, detail_level)
    normalized["concern"] = tailor_concern(normalized["concern"], audience)
    return normalized


def is_usable_model_text(value: str) -> bool:
    if len(value.strip()) < 12:
        return False
    low_value = value.lower()
    repeated_prompt_phrases = [
        "do not diagnose",
        "do not recommend medication",
        "task:",
        "health text:",
        "audience:",
    ]
    return not any(phrase in low_value for phrase in repeated_prompt_phrases)


def tailored_next_steps(audience: str, detail_level: str) -> str:
    if audience == "patient":
        steps = [
            "Write down the main point and any personal questions it raises.",
            "Discuss personal risk, symptoms, or care decisions with a qualified health professional.",
            "Do not change medication or treatment based only on this text.",
        ]
        if detail_level == "detailed":
            steps.extend(
                [
                    "Ask whether the information applies to your age, medical history, and current risk factors.",
                    "Look for absolute risk numbers, not only words such as higher or lower risk.",
                ]
            )
    elif audience == "public health student":
        steps = [
            "Identify the study design, population, exposure, comparison group, and outcome.",
            "Check whether the text supports causation or only association.",
            "Look for confounders, missing context, and whether results are reported as absolute or relative risk.",
        ]
        if detail_level == "detailed":
            steps.extend(
                [
                    "Consider whether the findings generalize to other populations.",
                    "Think about how the message could be communicated without overstating certainty.",
                ]
            )
    else:
        steps = [
            "Use this as a plain-language explanation, not as medical advice.",
            "Check the original source for details about who was studied and what was measured.",
            "Ask a health professional if the topic affects your own health decisions.",
        ]
        if detail_level == "detailed":
            steps.extend(
                [
                    "Pay attention to whether the text reports cause and effect or only a possible link.",
                    "Compare the finding with trusted sources such as public health agencies or clinicians.",
                ]
            )
    return "\n".join(f"- {step}" for step in steps)


def tailor_summary(summary: str, audience: str, detail_level: str) -> str:
    if audience == "patient":
        prefix = "Patient-focused explanation: "
        addition = (
            "The main personal takeaway is to understand the risk message and bring questions to a clinician, "
            "rather than treating the text as a diagnosis."
        )
    elif audience == "public health student":
        prefix = "Public health interpretation: "
        addition = (
            "The main analytic takeaway is to separate the reported association from causal claims and consider "
            "how the evidence was produced."
        )
    else:
        prefix = "Plain-language explanation: "
        addition = "The main takeaway is what the text says in everyday terms, without adding medical advice."

    if detail_level == "detailed":
        return f"{prefix}{summary}\n\n{addition}"
    return f"{prefix}{summary}"


def tailor_risk_factors(risk_factors: str, source_text: str, audience: str, detail_level: str) -> str:
    lower = source_text.lower()
    detected = find_risk_terms(lower)
    if detected:
        base = "\n".join(f"- {factor}" for factor in detected)
    else:
        base = risk_factors

    if audience == "patient":
        note = (
            "Patient note: these are factors mentioned in the text, not a personalized risk assessment."
        )
    elif audience == "public health student":
        note = (
            "Public health note: treat these as exposures or covariates to evaluate in relation to the outcome."
        )
    else:
        note = "General note: these are the main health-related factors the text appears to discuss."

    if detail_level == "detailed":
        return f"{base}\n\n{note}"
    return f"{base}\n\n{note}"


def tailored_limitations(source_text: str, audience: str, detail_level: str) -> str:
    lower = source_text.lower()
    limitation = "This explanation is generated by AI and may miss details from the original text."
    if "observational" in lower or "associated" in lower or "linked" in lower:
        limitation = "The text appears to describe an association, so it should not be read as proof of cause and effect."
    elif "study" in lower:
        limitation = "The result depends on the study design, population, measurements, and possible sources of bias."

    if audience == "patient":
        limitation += " It also cannot determine your personal diagnosis, risk level, or treatment plan."
    elif audience == "public health student":
        limitation += " For public health interpretation, check confounding, selection bias, measurement quality, and generalizability."
    else:
        limitation += " For personal decisions, it should be checked against trusted medical or public health sources."

    if detail_level == "detailed":
        limitation += " A careful reading should also ask whether the text reports absolute risk, uncertainty, and who the results apply to."

    return limitation


def tailor_concern(concern: str, audience: str) -> str:
    if audience == "patient":
        return f"{concern}\n\nPatient context: this level describes the text, not your personal medical risk."
    if audience == "public health student":
        return f"{concern}\n\nPublic health context: interpret this level alongside study design, population, and uncertainty."
    return f"{concern}\n\nGeneral context: this is a rough reading of the text, not a clinical judgment."


def render_section(title: str, body: str):
    st.subheader(title)
    st.write(body)


st.title(APP_TITLE)
st.caption(
    "A Streamlit app that uses a pretrained Hugging Face language model to explain medical and public health text."
)

st.info(
    "This tool is for education and health communication only. It does not diagnose disease, replace a clinician, "
    "or provide treatment advice."
)

with st.sidebar:
    st.header("Settings")
    audience = st.selectbox(
        "Target reader",
        ["general public", "patient", "public health student"],
    )
    detail_level = st.radio("Output detail", ["brief", "detailed"], horizontal=True)
    st.divider()
    st.write(f"Model: `{MODEL_NAME}`")

example_text = (
    "High blood pressure, also known as hypertension, is a common condition that can increase the risk "
    "of heart disease, stroke, kidney disease, and other long-term health problems. Many people with high "
    "blood pressure do not have obvious symptoms, so regular screening is important. Blood pressure can be "
    "affected by age, family history, diet, physical activity, body weight, stress, alcohol use, smoking, "
    "and access to healthcare.\n\n"
    "Public health guidelines recommend that adults check their blood pressure regularly and work with "
    "healthcare providers to manage elevated readings. Lifestyle changes such as reducing sodium intake, "
    "eating more fruits and vegetables, being physically active, maintaining a healthy weight, limiting "
    "alcohol, and quitting smoking can help lower blood pressure for many people. Some individuals may also "
    "need medication prescribed by a healthcare provider.\n\n"
    "Although high blood pressure is treatable, it is often underdiagnosed or poorly controlled, especially "
    "in communities with limited access to preventive care. Public health programs can help by improving "
    "screening access, supporting affordable primary care, promoting healthy food environments, and educating "
    "communities about prevention and treatment. This information is for educational purposes and should not "
    "replace medical advice from a healthcare professional."
)

user_text = st.text_area(
    "Paste health news, a research abstract, a public health report, or a doctor explanation:",
    value=example_text,
    height=260,
)

analyze = st.button("Analyze Text", type="primary")

if analyze:
    cleaned_text = clean_input(user_text)
    if len(cleaned_text) < 40:
        st.warning("Please enter a longer health-related text before analyzing.")
    else:
        with st.spinner("Analyzing the text with a pretrained language model..."):
            try:
                generated_sections = generate_explanation(cleaned_text, audience, detail_level)
                sections = normalize_sections(generated_sections, cleaned_text, audience, detail_level)
                used_fallback = False
            except Exception as error:
                sections = keyword_fallback(cleaned_text, audience, detail_level)
                used_fallback = True
                st.error(
                    "The Hugging Face model could not be loaded or run in this environment. "
                    "Showing a simple rule-based fallback so the interface remains usable."
                )
                st.caption(f"Technical detail: {error}")

        if used_fallback:
            st.warning(
                "Fallback output is not the main AI model result. For the final demo, run the app where the model can load."
            )

        st.divider()
        st.header("Explanation")
        st.caption(f"Target reader: {audience} | Output detail: {detail_level}")
        render_section(SECTION_TITLES["summary"], sections["summary"])
        render_section(SECTION_TITLES["risk_factors"], sections["risk_factors"])
        render_section(SECTION_TITLES["next_steps"], sections["next_steps"])
        render_section(SECTION_TITLES["limitations"], sections["limitations"])
        render_section(SECTION_TITLES["concern"], sections["concern"])
