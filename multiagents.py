import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents import SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio
import streamlit as st

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#print("GOOGLE_API_KEY:", GOOGLE_API_KEY)

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY not found! Add it to your .env file.")

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[500, 429, 503, 504]
)

# Initialize the Gemini model
gemini_model = Gemini(
    api_key=GOOGLE_API_KEY,
    model_name="gemini-2.5-flash-lite",
    temperature=0.2,
    max_output_tokens=1024,
    retry_options=retry_config
)

# Create the Intake Agent
intake_agent = LlmAgent(
    model=gemini_model,
    name="intake_agent",
    instruction=(
        "You are the Patient Intake Agent. Your job is to gather the patient's symptoms. "
        "Pass them to the Diagnosis Agent for further analysis."
    ),
    output_key="intake_response"
)

# Create the Diagnosis Agent
diagnosis_agent = LlmAgent(
    model=gemini_model,
    name="diagnosis_agent",
    instruction=(
        "You are the Diagnosis Agent. Based on the symptoms provided by the "
        "Patient Intake Agent in {intake_response}, analyze and suggest possible diagnoses Provide "
        "a brief explanation for each diagnosis and chance of each diagnosis in percentage as a table format with rows and columns."
    ),
    output_key="diagnosis_response"
)

# create the speciality Agent
speciality_agent = LlmAgent(
    model=gemini_model,
    name="speciality_agent",
    instruction=(
        "You are the Speciality Agent. Based on the diagnoses provided by the "
        "Diagnosis Agent in {diagnosis_response}, determine the relevant medical specialties "
        "that the patient should consult. Provide the specialties in a comma-separated list."
    ),
    output_key="speciality_response"
)

# create the diagnostic test Agent
diagnostic_test_agent = LlmAgent(
    model=gemini_model,
    name="diagnostic_test_agent",
    instruction=(
        "You are the Diagnostic Test Agent. Based on the diagnoses provided by the "
        "Diagnosis Agent in {diagnosis_response}, suggest appropriate diagnostic tests "
        "that the patient should undergo. Provide the tests in a comma-separated list."
    ),
    output_key="diagnostic_test_response"
)

# Chain the agents
pipeline_agent = SequentialAgent(
    name="AgentPipeline",
    sub_agents=[intake_agent, diagnosis_agent, speciality_agent, diagnostic_test_agent],
)



# Runner
# runner = InMemoryRunner(agent=pipeline_agent)

# async def main(): 
#     try: 
#         response = await runner.run_debug( "I have been experiencing headaches and dizziness for the past week." ) 
#         print("Agent Response:", response.text) 
#     except Exception as e: 
#         print("Agent analysis completed and closed") 

# if __name__ == "__main__": 
#     asyncio.run(main())

def run_pipeline(user_input: str) -> str:
    """
    Run the SequentialAgent pipeline and return concatenated responses.
    """
    try:
        runner = InMemoryRunner(agent=pipeline_agent)
        response = asyncio.run(runner.run_debug(user_input))
    except Exception as e:
        return f"An error occurred: {e}"


st.title("🩺 AI Medical Assistant")
st.write("Enter your symptoms and let the AI suggest possible diagnoses, relevant specialties, and diagnostic tests.")

symptoms_input = st.text_area("Describe your symptoms:", height=150)

if st.button("Submit"):
    if not symptoms_input:
        st.warning("Please describe your symptoms to proceed.")
    else:
        st.info("Running analysis, please wait...")
        user_input = symptoms_input
        # response_text = run_pipeline(user_input)
        st.subheader("AI Analysis")
        st.write(run_pipeline(user_input))
