import os
import sys
import json
from pydantic import BaseModel, Field
from typing import List, Literal

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_community import GoogleSearchAPIWrapper
except ImportError:
    print("Error: Required LangChain or Google libraries not found.")
    print("Please install them using: pip install langchain langchain-google-genai langchain-google-community pydantic google-api-python-client")
    sys.exit(1)

# --- Configuration ---------------------------------------------------
# IMPORTANT: Set your API keys as environment variables before running.
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
# os.environ["GOOGLE_CSE_ID"] = "YOUR_GOOGLE_CUSTOM_SEARCH_ENGINE_ID"
# os.environ["GOOGLE_API_KEY_SEARCH"] = "YOUR_GOOGLE_SEARCH_API_KEY"
# By default prefer Gemini 2.5 Pro so you don't need to set it on the CLI.
# You can still override by setting the GEMINI_MODEL env var.
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-pro")
# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------------------------------------

# === 1. Pydantic Models (Define our desired JSON structures) ===

class ExtractedJDInfo(BaseModel):
    """Structured information extracted from a Job Description."""
    technical_skills: List[str] = Field(..., description="Specific technologies, languages, or frameworks mentioned (e.g., 'Python', 'React', 'TensorFlow').")
    conceptual_topics: List[str] = Field(..., description="Broader concepts and knowledge areas (e.g., 'Data Structures & Algorithms', 'System Design', 'Machine Learning', 'Backend Development').")
    responsibilities: List[str] = Field(..., description="Key responsibilities of the role (e.g., 'Design and build scalable APIs', 'Manage data pipelines').")

class InterviewRound(BaseModel):
    """A single, structured interview round."""
    round_name: str = Field(..., description="The name of the interview round (e.g., 'Online Assessment', 'Technical Phone Screen', 'System Design', 'Behavioral').")
    topics: List[str] = Field(..., description="Key topics to be tested in this round.")

class FinalRoadmap(BaseModel):
    """The final, structured preparation roadmap."""
    company: str
    role: str
    difficulty: Literal["Easy", "Medium", "Hard", "Very Hard"]
    interview_rounds: List[InterviewRound]
    recommended_preparation_order: List[str] = Field(..., description="The logical order to study topics (e.g., '1. DSA Foundations', '2. Core CS', '3. System Design').")


# === 2. Agent 1: The Analyst (JD Parser) ===

def init_gemini_llm(preferred_models=None, temperature=0):
    """Try to initialize a Gemini-compatible LLM from a list of candidate model names.
    Returns the first successfully created LLM or None.
    """
    # Allow user to override the model via environment variable for compatibility
    env_model = os.environ.get("GEMINI_MODEL")
    if preferred_models is None:
        preferred_models = [
            # Prefer newer 2.5 models (pro and flash). User can also set GEMINI_MODEL to override.
            "gemini-2.5-pro",
            "gemini-flash-2.5",
            "gemini-2.5",
            "gemini-1.5-pro-latest",
            "gemini-1.5",
            "gemini-1.0",
            "gemini-1.5-mini",
            "chat-bison",
        ]
    if env_model:
        # put user-specified model first
        preferred_models.insert(0, env_model)
    last_exc = None
    for m in preferred_models:
        try:
            return ChatGoogleGenerativeAI(model=m, temperature=temperature, google_api_key=os.environ.get("GOOGLE_API_KEY"))
        except Exception as e:
            last_exc = e
    print(f"Error initializing Gemini LLM. Tried models: {preferred_models}. Last error: {last_exc}")
    return None


def get_jd_parser_agent():
    """
    Creates an agent (a LangChain chain) that parses a JD into a
    structured Pydantic object.
    """
    llm = init_gemini_llm(temperature=0)
    if llm is None:
        print("Please ensure GOOGLE_API_KEY is set correctly and that at least one Gemini model is available.")
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert AI assistant and senior technical recruiter. "
         "Your task is to parse the given job description (JD) and role "
         "to extract key skills and map them to broader topics. "
         "Focus on what a candidate needs to know. "
         "For example, 'REST APIs' maps to 'Backend Development', "
         "'React' maps to 'Frontend Development', and 'Data Structures' maps to 'DSA'."),
        ("human", "Role: {role}\n\nJob Description:\n{jd}")
    ])

    # Use .with_structured_output to force reliable JSON/Pydantic output
    parser_chain = prompt | llm.with_structured_output(ExtractedJDInfo)
    return parser_chain


# === 3. Agent 2: The Scout (Interview Researcher) ===

def get_interview_researcher_agent():
    """
    Creates an agent that can use Google Search to find information
    about a company's interview process.
    """
    # 3.1. Initialize the Google Search wrapper
    search_wrapper = GoogleSearchAPIWrapper(
        google_api_key=os.environ.get("GOOGLE_API_KEY_SEARCH"),
        google_cse_id=os.environ.get("GOOGLE_CSE_ID")
    )

    # 3.2. Create a simple researcher object that uses the search wrapper
    class SimpleResearcher:
        def __init__(self, search):
            self.search = search

        def invoke(self, inputs: dict):
            company = inputs.get("company", "")
            role = inputs.get("role", "")
            topics = inputs.get("topics", "")
            query = f"{company} {role} interview process {topics}".strip()
            try:
                results = self.search.run(query)
            except Exception as e:
                return {"output": f"Search failed: {e}"}

            # Return raw search results as the researcher's output. This keeps the
            # researcher simple and avoids relying on LangChain's agent runtime API,
            # which has changed across versions.
            return {"output": results}

    return SimpleResearcher(search_wrapper)


# === 4. Agent 3: The Strategist (Roadmap Builder) ===

def get_roadmap_builder_agent():
    """
    Creates an agent that synthesizes JD info and research
    into the final, structured roadmap.
    """
    llm = init_gemini_llm(temperature=0.2)
    if llm is None:
        print("Please ensure GOOGLE_API_KEY is set correctly and that at least one Gemini model is available.")
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert AI career coach, 'The Strategist'. "
         "Your goal is to create a definitive, structured preparation roadmap. "
         "You will be given: "
         "1. The original company, role, and JD. "
         "2. The structured skills and topics extracted from the JD ('JD Analysis'). "
         "3. A research summary of the typical interview process ('Interview Research'). "
         "Your task is to synthesize all this information: "
         "- Create a final list of interview_rounds. "
         "- Logically assign the conceptual_topics from the JD Analysis to the most appropriate round. "
         "- Determine a final difficulty (Easy, Medium, Hard, Very Hard) based on BOTH the company's reputation AND the skills in the JD. "
         "- Create a logical recommended_preparation_order. "
         "Be precise and generate the final roadmap."),
        ("human",
         "--- Original Input ---\n"
         "Company: {company}\n"
         "Role: {role}\n"
         "JD: {jd}\n\n"
         "--- 1. JD Analysis (from Analyst Agent) ---\n"
         "{jd_analysis}\n\n"
         "--- 2. Interview Research (from Scout Agent) ---\n"
         "{interview_research}\n\n"
         "--- Task ---\n"
         "Generate the final, structured JSON roadmap.")
    ])

    builder_chain = prompt | llm.with_structured_output(FinalRoadmap)
    return builder_chain


# === 5. Main Orchestration Logic ===

def check_env_vars():
    """Checks if all required environment variables are set."""
    required_vars = ["GOOGLE_API_KEY", "GOOGLE_CSE_ID", "GOOGLE_API_KEY_SEARCH"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("="*50)
        print("❌ ERROR: Missing Environment Variables")
        print(f"The following environment variables are not set: {', '.join(missing_vars)}")
        print("\nPlease set them in your environment before running the script:")
        print("  export GOOGLE_API_KEY=\"your_gemini_api_key\"")
        print("  export GOOGLE_CSE_ID=\"your_cse_id\"")
        print("  export GOOGLE_API_KEY_SEARCH=\"your_google_search_api_key\"")
        print("="*50)
        return False
    return True

def save_json(data, filename):
    """Saves Pydantic model data to a JSON file."""
    # Resolve path into OUTPUT_DIR if a relative filename was provided
    if not os.path.isabs(filename):
        filename = os.path.join(OUTPUT_DIR, filename)

    # Normalize data: if it's a Pydantic model, use model_dump(); if dict, use as-is; otherwise wrap
    try:
        if hasattr(data, 'model_dump'):
            payload = data.model_dump()
        elif isinstance(data, dict):
            payload = data
        else:
            # Fallback: try to serialize directly (e.g., string)
            payload = {"value": data}
    except Exception:
        payload = {"value": str(data)}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\n✅ Output saved to {filename}")

def run_research_agent(company: str, role: str, jd: str, output_filename: str):
    """
    Runs the full multi-agent workflow.
    """
    print(f"🚀 Starting research for {role} at {company}...")

    # 1. Instantiate agents
    jd_parser = get_jd_parser_agent()
    researcher = get_interview_researcher_agent()
    builder = get_roadmap_builder_agent()

    if not all([jd_parser, researcher, builder]):
        print("Failed to initialize one or more agents. Exiting.")
        return

    # --- Agent 1: Analyst ---
    print("\n[Phase 1/3] 🕵 Analyst is parsing the JD...")
    # Prepare base filename for final output (do not save intermediate JD analysis file)
    base = os.path.splitext(os.path.basename(output_filename))[0]
    try:
        jd_analysis: ExtractedJDInfo = jd_parser.invoke({
            "role": role,
            "jd": jd
        })
        print("...JD Analysis Complete.")
        print(f"   - Skills: {jd_analysis.technical_skills}")
        print(f"   - Topics: {jd_analysis.conceptual_topics}")
    except Exception as e:
        print(f"Error during JD Parsing (Phase 1): {e}")
        print("\nHint: The Gemini model you are using may not be available for your API version.\n" \
              "Set the environment variable GEMINI_MODEL to a supported model name (for example 'gemini-1.0' or a model listed by your Google account).\n" \
              "You can also check available models via the Google Generative API or your cloud console.")
        return

    # --- Agent 2: Scout ---
    print("\n[Phase 2/3] 🧭 Scout is researching the interview process...")
    try:
        research_summary = researcher.invoke({
            "company": company,
            "role": role,
            "topics": ", ".join(jd_analysis.conceptual_topics)
        })
        print("...Research Complete.")
    except Exception as e:
        print(f"Error during Interview Research (Phase 2): {e}")
        print("This often happens if search API keys (GOOGLE_API_KEY_SEARCH, GOOGLE_CSE_ID) are invalid or have no quota.")
        return

    # --- Agent 3: Strategist ---
    print("\n[Phase 3/3] 🧠 Strategist is building the final roadmap...")
    try:
        final_roadmap: FinalRoadmap = builder.invoke({
            "company": company,
            "role": role,
            "jd": jd,
            "jd_analysis": jd_analysis.model_dump_json(indent=2),
            "interview_research": research_summary['output']
        })
        print("...Roadmap Generated.")
    except Exception as e:
        print(f"Error during Roadmap Building (Phase 3): {e}")
        return

    # --- Output ---
    print("\n--- 🏁 FINAL ROADMAP ---")
    print(final_roadmap.model_dump_json(indent=2))

    # Save to file
    save_json(final_roadmap, f"{base}_roadmap.json")
    return final_roadmap


# === 6. NEW Interactive Mode ===

def interactive_mode():
    """
    Prompts the user for inputs to run the research agent.
    """
    print("Welcome to the GenAI Job Prep Roadmap Agent! 🤖")
    print("Let's build your preparation plan.")
    
    company = input("\n1. Enter the Company Name: ").strip()
    role = input("2. Enter the Role (e.g., Software Engineer 1): ").strip()
    
    print("\n3. Paste the Job Description below.")
    print("   Type 'END' (all caps) on a new line when you are finished.")
    
    jd_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        jd_lines.append(line)
    
    jd = "\n".join(jd_lines)
    
    if not company or not role or not jd:
        print("\n❌ Error: Company Name, Role, and Job Description cannot be empty. Please try again.")
        return

    # Create a safe filename
    safe_company = company.lower().replace(' ', '_').replace('.', '')
    safe_role = role.lower().split('(')[0].strip().replace(' ', '').replace('/', '')
    output_filename = f"{safe_company}_{safe_role}_roadmap.json"

    # Run the agent with the user's inputs
    run_research_agent(company, role, jd, output_filename)


# === 7. Main Execution ===

if __name__ == "__main__":
    
    if not check_env_vars():
        sys.exit(1) # Stop if API keys are missing

    # Run the new interactive mode
    interactive_mode()