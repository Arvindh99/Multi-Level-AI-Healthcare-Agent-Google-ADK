import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents import SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY:", GOOGLE_API_KEY)

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
        "You are the Patient Intake Agent. Your job is to gather the patient's "
        "symptoms. Clarify any ambiguous symptoms by asking 2 - 3 follow-up questions. "
        "Once you have a clear understanding of the symptoms, pass them to the "
        "Diagnosis Agent for further analysis."
    ),
    output_key="intake_response"
)

# Create the Diagnosis Agent
diagnosis_agent = LlmAgent(
    model=gemini_model,
    name="diagnosis_agent",
    instruction=(
        "You are the Diagnosis Agent. Based on the symptoms provided by the "
        "Patient Intake Agent in {intake_response}, analyze and suggest possible diagnoses and the single most appropriate medical specialty, and passes the specialty name. Provide "
        "a brief explanation for each diagnosis and chance of each diagnosis in percentage as a table format."
    ),
    output_key="diagnosis_response"
)

#Create the clinic location Agent
location_agent = LlmAgent(
    model=gemini_model,
    name="location_agent",
    instruction=(
        "You are the Clinic Location Agent. Based on the medical specialty provided by the "
        "Diagnosis Agent in {diagnosis_response}, ask the user about their location and suggest the nearest clinic location that specializes in that field. Provide the clinic name, address, and contact information."
    ),
    output_key="location_response"
)

# Chain the agents
pipeline_agent = SequentialAgent(
    name="BlogPipeline",
    sub_agents=[intake_agent, diagnosis_agent,location_agent],
)



# Runner
runner = InMemoryRunner(agent=pipeline_agent)

async def main():
    try:
        response = await runner.run_debug(
            "I have been experiencing headaches and dizziness for the past week."
        )
        print("Agent Response:", response.text)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    asyncio.run(main())
