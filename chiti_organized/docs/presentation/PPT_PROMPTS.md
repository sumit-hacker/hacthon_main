# PPT Prompts (Saras RAG + CMS)

Use these prompts to generate a clean hackathon PPT quickly (Google Slides / PowerPoint / Canva / Gamma).

## Prompt 0 — Generate full deck (recommended)
Copy-paste:

"""
Create a 10–12 slide hackathon pitch deck.

Project 1: Saras (Team Chitti RAG)
- Saras is an education-focused assistant for admin-side work: organizing, filtering, and managing bulk student/academic data.
- Built during Virsat Hackathon at Patna Women’s College (Patna, Bihar) on 30 Jan, 2:57pm.
- Team: BCA 2nd-year students (College of Commerce, Arts and Science): Sumit, Gautam, Sachin, Aditya, Anshu.
- Key feature: Auto-RAG (AI chooses when to use retrieval). Trivial greetings should not force RAG.
- Local stack: FastAPI backend + React frontend + TinyLlama (local LLM) + Sentence-Transformers embeddings + FAISS vector store.
- Offline/private usage: bootable USB / Codespaces / VM / local PC.

Project 2: Complaint Management System (CMS) website
- A multi-step complaint form with categories, attachments, and OTP verification checkbox.
- Show how Saras can support admin workflows: summarizing complaints, tagging, routing, and answering policy questions.

Deck requirements:
- Each slide: title + 3–6 bullets.
- Add speaker notes per slide (2–4 sentences).
- Include 1 architecture diagram slide (ASCII/box format is OK).
- Include 1 demo flow slide with steps: Warmup → Build Index → Chat (Auto/On/Off).
- Keep language simple and judge-friendly.
"""

## Prompt 1 — Slide-by-slide (if you build manually)

### Slide 1: Title
Prompt:
"Write a title slide for a hackathon PPT with project name Saras (Team Chitti RAG) + Complaint Management System (CMS). Include team names and event/place/time."

### Slide 2: Problem Statement
Prompt:
"Explain the problem college admins face: large student/academic documents + scattered complaints + slow manual processing. 5 bullets."

### Slide 3: Our Solution
Prompt:
"Explain Saras + CMS together as a workflow: collect complaint → retrieve policy/docs → generate response/summary → route to correct department. 5 bullets."

### Slide 4: Saras Key Features
Prompt:
"List Saras features: Auto-RAG, local/private mode, sources/citations, admin-focused responses, fast retrieval with FAISS."

### Slide 5: CMS Key Features
Prompt:
"Describe CMS complaint form features: multi-step flow, user types (student/teacher/staff), complaint category/subcategory, attachments, optional OTP checkbox, review/submit."

### Slide 6: Architecture (Diagram)
Prompt:
"Draw a simple architecture: React UI → FastAPI → (Auto-RAG router) → TinyLlama + (Embeddings + FAISS) → Answer + Sources. Add a box for CMS website and how it can send complaint text to Saras." 

### Slide 7: RAG vs No-RAG (Why Auto-RAG)
Prompt:
"Explain why not every query needs RAG. Mention greetings like 'hi' should be normal chat. Mention confidence gating by similarity score and index presence."

### Slide 8: Demo Flow
Prompt:
"Create a 6-step demo flow with expected outputs (warmup, build index, ask 2 questions, show sources, switch RAG On/Off)."

### Slide 9: Challenges & Fixes
Prompt:
"Write 4–6 bullets: chunking loop fix, offline embeddings caching, local-only TinyLlama for Codespaces, avoid forcing RAG for trivial messages, stable endpoints (/api)."

### Slide 10: Results / Impact
Prompt:
"Write results: faster document-grounded answers, admin time saving, consistent responses, scalable for large datasets."

### Slide 11: Future Scope
Prompt:
"List future features: complaint auto-tagging, routing rules, dashboards, role-based access, audit logs, reranking, multilingual support." 

### Slide 12: Thank You
Prompt:
"Write a closing slide with a short one-liner and demo URLs placeholders."

## Prompt 2 — Design prompt (for a PPT generator)
Copy-paste:

"""
Design style:
- Theme: modern academic + government portal
- Colors: navy + white + accent saffron/teal
- Fonts: clean sans-serif for body; bold for headings
- Use icons for: documents, search, shield/privacy, complaint, workflow
- Add placeholders for screenshots:
  - Saras web UI
  - CMS complaint form
"""
