import os
import uuid
import time
import asyncio
from dotenv import load_dotenv
import streamlit as st

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY not found! Add it to your .env file.")

# Retry configuration
retry_config = types.HttpRetryOptions(attempts=3,exp_base=7,initial_delay=1,http_status_codes=[500, 429, 503, 504])

# Gemini model initialization

gemini_model = Gemini(api_key=GOOGLE_API_KEY,model_name="gemini-2.5-flash-lite",temperature=0.2,max_output_tokens=1024,retry_options=retry_config)

# Create Agents

intake_agent = LlmAgent(model=gemini_model,name="intake_agent",
    instruction=(
        "You are the Patient Intake Agent. Your job is to gather the patient's symptoms. "
        "Pass them to the Diagnosis Agent for further analysis."),
    output_key="intake_response")

diagnosis_agent = LlmAgent(model=gemini_model,name="diagnosis_agent",
    instruction=(
        "You are the Diagnosis Agent. Based on symptoms from {intake_response}, "
        "identify possible diagnoses and chance percentage. "
        "Create ONLY structured JSON with this format:\n"
        "{ 'diagnoses': [ {'condition': '...', 'chance': '...%'} ] }"),
    output_key="diagnosis_response")

speciality_agent = LlmAgent(model=gemini_model,name="speciality_agent",
    instruction=(
        "You are the Speciality Agent. Based on {diagnosis_response}, "
        "identify relevant medical specialties. "
        "Create ONLY JSON:\n"
        "{ 'specialties': ['specialty1', 'specialty2'] }"),
    output_key="speciality_response")

diagnostic_test_agent = LlmAgent(model=gemini_model,name="diagnostic_test_agent",
    instruction=(
        "You are the Diagnostic Test Agent. Based on {diagnosis_response}, "
        "suggest diagnostic tests. Create ONLY JSON:\n"
        "{ 'tests': ['test1', 'test2'] }"),
    output_key="diagnostic_test_response")

format_agent = LlmAgent(model=gemini_model,name="format_agent",
    instruction=(
        """You are the Format Agent. Based on:\n
        - Symptoms: {intake_response}\n
        - Diagnoses: {diagnosis_response}\n
        - Specialties: {speciality_response}\n
        - Tests: {diagnostic_test_response}\n\n
        Generate a single final, well-formatted medical summary for the user. No need to show any JSON Structure. \n
        Use:\n
        - Bullet lists\n
        - Tables\n
        - Clear sections (Symptoms, Diagnoses, Specialties, Tests)\n\n
        This is the ONLY user-facing output. The earlier agents should remain silent."""),
    output_key="final_response")

# Create Sequential Agent Pipeline

pipeline_agent = SequentialAgent(name="AgentPipeline",sub_agents=[intake_agent,diagnosis_agent,speciality_agent,diagnostic_test_agent,format_agent])



def run_pipeline(user_input: str) -> str:
    """
    Run the SequentialAgent pipeline and extract text from ADK Event objects.
    """
    try:
        runner = InMemoryRunner(agent=pipeline_agent)
        events = asyncio.run(runner.run_debug(user_input))

        output_text = ""

        for event in events:
            content = getattr(event, "content", None)
            if not content:
                continue

            if content.role == "model":
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        output_text += part.text + "\n"

        return output_text.strip() if output_text else "No response generated."

    except Exception as e:
        return f"Error occurred: {e}"



# Streamlit App UI

st.set_page_config(page_title="AI Medical Assistant",page_icon="🩺",layout="wide")

st.title("🩺 AI Medical Assistant")

# Session State

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Session Control")

    if st.session_state.session_id:
        st.success(f"Session: {st.session_state.session_id}")

        if st.button("🔁 Start New Session"):
            st.session_state.session_id = f"session-{int(time.time())}"
            st.session_state.messages = []
    else:
        if st.button("➕ Create Session"):
            st.session_state.session_id = f"session-{int(time.time())}"
            st.session_state.messages = []

    st.divider()


st.subheader("How can I help you today?")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if st.session_state.session_id:

    user_input = st.chat_input("Describe your symptoms...")

    if user_input:
        st.session_state.messages.append({"role": "user","content": user_input})

        with st.chat_message("assistant"):
            st.info("⏳ Analyzing your symptoms, please wait...")

        # Run pipeline
        response_text = run_pipeline(user_input)

        # store assistant reply
        st.session_state.messages.append({"role": "assistant","content": response_text})

        st.rerun()

else:
    st.info("👈 Please create a session to start chatting.")
