### Design and Implementation of an LLM-Based Multi-Agent Healthcare Assistant
##### Capstone Project – Kaggle 5-Day Agents Program

#### Project Overview
This AI Agent is an intelligent medical triage and guidance assistant built using Google’s Agent Developer Kit (ADK) and Gemini models. The application takes user symptoms and processes them through a structured multi-agent pipeline to provide Symptom understanding, possible diagnoses with probability estimates, recommended medical specialties, suggested diagnostic tests and a final structured medical summary.

The project includes two versions:
- multiagents.py → Runs the agent pipeline fully in a terminal
- app.py → Interactive Streamlit UI for public usage

This project was developed as part of the Kaggle 5-Day Agents Learning Program: https://www.kaggle.com/learn-guide/5-day-agents

#### Problem Statement
People often lack immediate medical guidance and may not know:
- What their symptoms indicate
- What type of doctor they should consult
- Whether they need diagnostic tests
- How severe their condition might be

Google's ADK multi-agent architecture helps streamline and automate this assessment without providing a medical diagnosis.

***This tool aims to assist, not replace, healthcare professionals.***

#### Architecture
User Input → Patient Intake Agent 
           → Diagnosis Agent 
           → Speciality Agent 
           → Diagnostic Test Agent 
           → Format Agent → Final Output

#### How It Works
Each agent receives structured context from previous agents, processes it with Gemini, and outputs clean data.
Only the Format Agent produces the final message for the end-user.
Intermediate outputs are hidden in the Streamlit UI for a clean user experience.

#### Demo screenshots
<img width="1730" height="868" alt="image" src="https://github.com/user-attachments/assets/0f34990b-0226-421a-9eb3-41f3a707a812" />


#### Disclaimer
This application does NOT provide medical advice.
It is for educational and research purposes only.
Always consult a qualified healthcare provider for medical concerns.

