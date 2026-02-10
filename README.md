# ABB Multi-Agent RAG API

A multi-agent Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **CrewAI**, **IBM watsonx**, and **Elasticsearch** for answering and assisting with ABB low-voltage AC drive queries.

---

## Overview

This API intelligently routes user queries into two categories:

1. **General Product Information** → Answered using RAG (documentation-based retrieval)
2. **Application-Specific Support** → Either provides expert recommendations or requests clarification

---

## How It Works

### 1️ Supervisor Agent
Classifies the query into:
- `rag_agent`
- `expert_support_agent`

---

### 2️ RAG Agent
Used for:
- Product overviews
- Specifications
- Technical explanations

Pipeline:
- Clean query
- Perform Elasticsearch vector search
- Generate structured response

---

### 3️ Expert Support Agent
Used for:
- Drive selection
- Sizing
- ROI / energy savings
- Application-based recommendations

Flow:
- If technical specs are provided → Generate expert response
- If insufficient details → Return clarification template

---

##  Tech Stack

- FastAPI
- CrewAI
- IBM watsonx (LLaMA 3 70B Instruct)
- Elasticsearch (KNN vector search)
- NLTK
