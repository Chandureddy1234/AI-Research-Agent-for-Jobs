# GenAI Preparation Roadmap Agent

This project contains a sophisticated, multi-agent AI system that generates a personalized job preparation roadmap based on a company name, role, and job description.

The agent leverages a "Mixture of Experts" (MoE) pattern, specifically an "Analyst-Scout-Strategist" workflow, to break down the complex task into specialized sub-tasks. This ensures a more reliable, accurate, and creative output than a single, monolithic agent.

---

## 🚀 Features

-   *Multi-Agent Workflow:* Uses a three-agent pipeline for robust reasoning and task separation.
-   *Deep JD Parsing:* The Analyst agent extracts not just keywords but conceptual topics (e.g., "scalability," "DSA," "Backend Development").
-   *Web-Enabled Research:* The Scout agent dynamically searches the web using the Google Search API to find the latest information on a company's interview process.
-   *Structured JSON Output:* Uses Pydantic models to guarantee a valid, machine-readable JSON output every time, eliminating parsing errors.
-   *Technology:* Built with Python, LangChain, and the Google Gemini API.

---

## 🤖 The Multi-Agent Flow: Analyst-Scout-Strategist

The agent's reasoning is distributed across three specialized "experts":

### 1. Agent 1: The Analyst (JD Parser)

* *Responsibility:* To perform a deep read of the Job Description (JD).
* *How it Works:* This agent uses a Gemini model with LangChain's **.with_structured_output()** function. This forces the LLM's output to conform perfectly to a Pydantic class (ExtractedJDInfo), which includes lists of technical_skills, conceptual_topics, and responsibilities.

### 2. Agent 2: The Scout (Interview Researcher)

* *Responsibility:* To gather external intelligence on the company's interview process.
* *How it Works:* This is a *Tool-Using Agent*. It is given access to a Google Search tool. It formulates queries (e.g., "Netflix Senior ML Engineer interview rounds") to find information from blogs, Glassdoor, and forums. It then synthesizes this information into a summary of likely rounds.

### 3. Agent 3: The Strategist (Roadmap Builder)

* *Responsibility:* To synthesize all collected data into the final, actionable roadmap.
* *How it Works:* This agent receives the structured data from the Analyst and the research summary from the Scout. It performs the final, critical reasoning step:
    1.  *Map Topics to Rounds:* It logically assigns topics (from the JD) to the interview rounds (from the research).
    2.  *Determine Order:* It generates a recommended_preparation_order based on foundational principles.
    3.  *Finalize Difficulty:* It sets a final difficulty by combining the company's reputation and the JD's specific demands.

---

## 🛠 Setup & Usage

### 1. Installation

First, install the required Python libraries using the requirements.txt file:

```bash
pip install -r requirements.txt

Small helper script that parses a job description (JD), researches interview practices, and generates a final structured preparation roadmap using Google Generative AI (Gemini) and Google Custom Search.

### What it does
- Parses the JD to extract technical skills and conceptual topics.
- Performs a quick web search to gather interview process information.
- Synthesizes a final, structured roadmap (JSON).

Note: By design this repository now only saves the final roadmap JSON to the `output/` folder. Intermediate `*_jd_analysis.json` and `*_research.json` files are no longer written.

### Requirements
- Python 3.10+ (this project used Python 3.12 in development)
- A virtual environment is recommended.
- Required environment variables:
  - `GOOGLE_API_KEY` — Gemini / Generative API key
  - `GOOGLE_API_KEY_SEARCH` — Google Custom Search API key
  - `GOOGLE_CSE_ID` — Google Custom Search Engine ID
  - Optional: `GEMINI_MODEL` — override the default model (defaults to `gemini-2.5-pro` in the script)

### Install
Create and activate a venv, then install dependencies (example):

```bash
python -m venv .venv
source .venv/Scripts/activate   # on Windows (bash)
pip install -r requirements.txt  # or pip install langchain langchain-google-genai langchain-google-community pydantic google-api-python-client
```

### Run (interactive)

Run the script and follow the prompts to enter company, role and paste the JD (type `END` on its own line when finished):

```bash
python research_agent.py
```

When the agent completes, the final roadmap file will be written to the `output/` folder with a filename like:

```
output/{safe_company}_{safe_role}_roadmap_roadmap.json
```

If you'd prefer a different filename format (for example to remove the duplicate `_roadmap`), edit the `interactive_mode()` output filename generation in `research_agent.py`.

### Troubleshooting
- If the script exits immediately, check that the environment variables listed above are set in your shell.
- If a Gemini model name is not found or you receive quota errors, try setting `GEMINI_MODEL` to a model available for your account, or check your Google Cloud quota/billing.
# GenAI Research Agent for Jobs

Repository: https://github.com/Chandureddy1234/AI-Research-Agent-for-Jobs

This project provides a small multi-agent pipeline that parses job descriptions (JDs), researches company interview practices, and generates a final, structured preparation roadmap (JSON) using Google Generative AI (Gemini) and Google Custom Search.

## Quick features

- Multi-agent pipeline: Analyst (JD parser), Scout (web researcher), Strategist (roadmap builder)
- Structured JSON output via Pydantic models
- Uses LangChain + Google Gemini + Google Custom Search

---

## Files in this repository

- `research_agent.py` — main script (interactive). Runs the three-agent pipeline and saves the final roadmap to `output/`.
- `README.md` — this file.
- `requirements.txt` — Python dependencies (created for GitHub).
- `output/` — directory where the final roadmap JSON files are written (created locally; typically not committed with large artifacts).

## Requirements

- Python 3.10+ (tested with 3.12)
- A virtual environment is recommended
- Environment variables (required):
  - `GOOGLE_API_KEY` — Gemini / Generative API key
  - `GOOGLE_API_KEY_SEARCH` — Google Custom Search API key
  - `GOOGLE_CSE_ID` — Google Custom Search Engine ID
  - Optional: `GEMINI_MODEL` — override the default model (defaults to `gemini-2.5-pro` in the script)

## Install & run (local)

Create and activate a venv, install dependencies, then run interactively:

```bash
python -m venv .venv
source .venv/Scripts/activate   # on Windows using Git Bash / bash
pip install -r requirements.txt
python research_agent.py
```

Follow the prompts to enter company, role, and paste the JD (type `END` on a new line when finished). The final roadmap will be saved to the `output/` folder.

## Output

Only the final roadmap JSON is saved by default. Example filename format:

```
output/{safe_company}_{safe_role}_roadmap_roadmap.json
```

If you want intermediate files (`*_jd_analysis.json` and `*_research.json`) to be saved, I can add a simple env var or CLI flag to enable them.

## Troubleshooting

- If the script exits immediately, confirm required environment variables are set.
- If a Gemini model is unavailable or you hit quota errors, set `GEMINI_MODEL` to a model available on your account or check Google Cloud quotas.

---

If you'd like, I can: add a cleaner default filename (remove duplicate `_roadmap`), add a CLI flag to enable intermediates, or create a `requirements.txt` with pinned versions. Tell me which you'd prefer.
