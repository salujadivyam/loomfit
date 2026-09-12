#FastAPI+sentence-transformers+regex+Ollama
import io
import re
import csv
import uuid
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pdfplumber
import docx  # python-docx
from sentence_transformers import SentenceTransformer, util

#title 
app = FastAPI(title="Loomfit Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
print("Loading embedding model...")
embedmodel=SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

ollama_url="http://localhost:11434/api/generate"
ollamamodel="qwen2.5:3b-instruct"

# thread pool for blocking work (embeddings, parsing, ollama http calls) so the
# asyncio event loop never stalls while one request is being processed
executor=ThreadPoolExecutor(max_workers=4)

# sessions keyed by a session_id, instead of one shared global.
# each entry: {"rankings": [...]}
sessions:dict={}

# a reasonably broad skill vocabulary used for keyword extraction.
# JD skills are matched against this list, plus any capitalized/tech-looking
# tokens pulled directly out of the JD text, so it is not limited to this list
skillvocab=[
    "python", "java", "javascript", "typescript", "react", "angular", "vue",
    "node.js", "node", "express", "django", "flask", "fastapi", "spring",
    "html", "css", "tailwind", "bootstrap", "sql", "mysql", "postgresql",
    "mongodb", "redis", "docker", "kubernetes", "aws", "azure", "gcp",
    "git", "github", "ci/cd", "jenkins", "terraform", "linux", "bash",
    "pandas", "numpy", "scikit-learn", "sklearn", "pytorch", "tensorflow",
    "machine learning", "deep learning", "nlp", "computer vision",
    "rest api", "graphql", "microservices", "kotlin", "swift", "android",
    "ios", "figma", "selenium", "cypress", "jira", "excel", "tableau",
    "power bi", "c++", "c#", "golang", "rust", "spark", "hadoop",
    "elasticsearch", "kafka", "rabbitmq", "oauth", "jwt",
]

# sorted longest-first so multi-word / longer skills (e.g. "node.js") are
# checked before short substrings (e.g. "node") get a chance to misfire
skillvocabsort=sorted(skillvocab, key=len, reverse=True)


#skill matching with boundary aware
def build_skill_pattern(skill: str) -> re.Pattern:
    """
    Build a regex that matches a skill as a whole token, not a substring.
    Handles skills with punctuation (c++, c#, node.js, ci/cd) by using
    lookaround boundaries instead of \\b, since \\b does not work reliably
    around non-word characters.
    """
    escaped = re.escape(skill.lower())

    # boundary = start of string/non-alphanumeric on each side
    pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
    return re.compile(pattern)


skillpattern={skill: build_skill_pattern(skill) for skill in skillvocabsort}


def skill_present(text_lower: str, skill: str) -> bool:
    pattern = skillpattern.get(skill) or build_skill_pattern(skill)
    return pattern.search(text_lower) is not None

#parsing helper
def extract_text(filename: str, raw: bytes) -> str:
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            text = []
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        elif name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        else:  # txt or unknown, best-effort decode
            return raw.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Failed to parse {filename}: {e}")
        return ""


def split_sections(resume_text: str) -> dict:
    """
    Lightweight section splitter. Looks for common headers and buckets
    the text under experience / skills / education / other. Resumes with no
    clear headers fall entirely under 'other', which still gets scored.
    """
    lines=resume_text.split("\n")
    sections={"experience": [], "skills": [], "education": [], "other": []}
    current="other"

    header_map={
        "experience": ["experience", "work experience", "employment", "professional experience"],
        "skills": ["skills", "technical skills", "core competencies"],
        "education": ["education", "academic background"],
    }

    for line in lines:
        stripped=line.strip().lower()
        matched_header=None
        for section, headers in header_map.items():
            if any(stripped==h or stripped.startswith(h) for h in headers) and len(stripped)< 40:
                matched_header=section
                break
        if matched_header:
            current =matched_header
            continue
        sections[current].append(line)

    return {k: "\n".join(v) for k, v in sections.items()}


def extract_jd_skills(jd_text: str) -> List[str]:
    jd_lower =jd_text.lower()
    found= [skill for skill in skillvocabsort if skill_present(jd_lower, skill)]

    # also grab capitalized short tokens (likely tool/tech names) not in vocab
    extra= re.findall(r"\b[A-Z][a-zA-Z0-9+.#]{1,20}\b", jd_text)
    stop= {"the", "and", "you", "our", "job", "role", "required", "good"}
    for token in extra:
        t= token.lower()
        if t not in found and len(t)> 1 and t not in stop:
            found.append(t)

    # dedupe, keep order
    seen= set()
    result= []
    for s in found:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def keyword_score(resume_text: str, sections: dict, jd_skills: List[str]):
    resume_lower= resume_text.lower()
    exp_lower= sections["experience"].lower()

    matched, missing= [], []
    weighted_hits= 0.0

    for skill in jd_skills:
        # skills not in our known vocab (freeform tokens pulled from the JD)
        # still get boundary-safe matching via build_skill_pattern

        in_resume= skill_present(resume_lower, skill)
        if not in_resume:
            missing.append(skill)
            continue
        matched.append(skill)
        # skill mentioned inside experience section counts more than a bare list
        if skill_present(exp_lower, skill):
            weighted_hits+= 1.0
        else:
            weighted_hits +=0.6

    denom =max(len(jd_skills), 1)
    score= min(100, round((weighted_hits/denom)*100))
    return score, matched, missing


#semantic scoring, with chunking, fixes long resume truncation
def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping character chunks. MiniLM's real limit is in
    tokens (~256), but chunking by characters with overlap is a cheap, model
    agnostic way to keep each chunk comfortably under that limit while not
    cutting a relevant paragraph in half at a boundary.
    """
    text =text.strip()
    if len(text)<= max_chars:
        return [text] if text else []

    chunks =[]
    start =0
    while start <len(text):
        end =start+ max_chars
        chunks.append(text[start:end])
        start =end-overlap
    return chunks


def _semantic_score_sync(jd_text: str, resume_text: str) -> int:
    if not resume_text.strip():
        return 0

    resume_chunks= chunk_text(resume_text)
    if not resume_chunks:
        return 0

    emb_jd= embedmodel.encode(jd_text, convert_to_tensor=True)
    emb_chunks=embedmodel.encode(resume_chunks, convert_to_tensor=True)

    sims=util.cos_sim(emb_jd, emb_chunks)[0]  # similarity to each chunk
    best_sim=float(sims.max())

    # cosine sim is roughly -1..1, in practice 0..0.8 for real text; rescale
    score=max(0, min(100, round(best_sim * 125)))
    return score


async def semantic_score(jd_text: str, resume_text: str) -> int:
    loop=asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _semantic_score_sync, jd_text, resume_text)

#explanation
def template_explanation(name: str, matched: List[str], missing: List[str]) -> str:
    matched_str = ", ".join(matched[:6]) if matched else "no direct skill matches"
    missing_str = ", ".join(missing[:5]) if missing else "no major gaps"
    return (
        f"{name} matched on {matched_str}. "
        f"Missing or unconfirmed: {missing_str}."
    )


def _call_ollama_sync(prompt: str) -> str:
    try:
        resp = requests.post(
            ollama_url,
            json={"model": ollamamodel, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return ""


async def call_ollama(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _call_ollama_sync, prompt)


async def llm_explanation(name: str, matched: List[str], missing: List[str], score: int) -> str:
    prompt = (
        f"You are explaining a resume ranking result to a recruiter in 2-3 short sentences. "
        f"Candidate: {name}. Overall score: {score}/100. "
        f"Matched skills: {', '.join(matched) if matched else 'none'}. "
        f"Missing skills: {', '.join(missing) if missing else 'none'}. "
        f"Write a plain, professional explanation of why this candidate ranked here. "
        f"Do not repeat the raw skill lists verbatim, summarize naturally."
    )
    result = await call_ollama(prompt)
    if result:
        return result
    return template_explanation(name, matched, missing)


#routes
@app.post("/upload")
async def upload(jd: str = Form(...), jd_role: str = Form(""), resumes: List[UploadFile] = File(...)):
    jd_skills = extract_jd_skills(jd)
    candidates = []
    loop = asyncio.get_event_loop()

    for f in resumes:
        raw = await f.read()
        # parsing is cheap for txt but pdfplumber/docx can be slow on big files,
        # push it off the event loop too
        text = await loop.run_in_executor(executor, extract_text, f.filename, raw)
        sections = split_sections(text)

        k_score, matched, missing = keyword_score(text, sections, jd_skills)
        s_score = await semantic_score(jd, text)
        final_score = round(0.6 * s_score + 0.4 * k_score)

        candidate_name = f.filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()

        candidates.append({
            "name": candidate_name,
            "job": jd_role or "custom",
            "score": final_score,
            "semantic": s_score,
            "keyword": k_score,
            "matched": matched,
            "missing": missing,
            "explanation": "",
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    # only generate LLM explanations for top 3, template for the rest (keeps it fast)
    for i, c in enumerate(candidates):
        if i < 3:
            c["explanation"] = await llm_explanation(c["name"], c["matched"], c["missing"], c["score"])
        else:
            c["explanation"] = template_explanation(c["name"], c["matched"], c["missing"])

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"rankings": candidates}

    return {"session_id": session_id, "rankings": candidates}


@app.get("/rankings")
async def get_rankings(session_id: str = ""):
    session = sessions.get(session_id)
    if session is None:
        return {"rankings": []}
    return {"rankings": session["rankings"]}


@app.get("/export-csv")
async def export_csv(session_id: str = ""):
    session = sessions.get(session_id, {"rankings": []})
    rankings = session["rankings"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rank", "Name", "Job", "Score", "Semantic", "Keyword", "Matched", "Missing", "Explanation"])
    for i, c in enumerate(rankings):
        writer.writerow([
            i + 1, c["name"], c["job"], c["score"], c["semantic"], c["keyword"],
            "; ".join(c["matched"]), "; ".join(c["missing"]), c["explanation"],
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=loomfit_rankings.csv"},
    )


@app.post("/chat")
async def chat(payload: dict):
    question = payload.get("question", "")
    session_id = payload.get("session_id", "")
    session = sessions.get(session_id)

    if not session or not session["rankings"]:
        return {"answer": "Run a ranking first, then ask me about the results."}

    rankings = session["rankings"]
    context_lines = []
    for i, c in enumerate(rankings[:10]):
        context_lines.append(
            f"{i+1}. {c['name']} - score {c['score']}/100 "
            f"(semantic {c['semantic']}, keyword {c['keyword']}). "
            f"Matched: {', '.join(c['matched']) or 'none'}. "
            f"Missing: {', '.join(c['missing']) or 'none'}."
        )
    context = "\n".join(context_lines)

    prompt = (
        f"You are a recruiting assistant. Here is the current candidate ranking:\n\n"
        f"{context}\n\n"
        f"Recruiter question: {question}\n\n"
        f"Answer in 2-4 sentences using only the data above. Do not invent skills or scores."
    )
    answer = await call_ollama(prompt)
    if not answer:
        answer = "Could not reach the local Ollama model. Check that it is running on port 11434."
    return {"answer": answer}


@app.get("/")
async def root():
    return {"status": "Loomfit backend running"}
