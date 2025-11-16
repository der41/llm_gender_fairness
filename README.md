# Profession Description Bias Analysis

This project analyzes how a Large Language Model (LLM) — specifically Qwen 1.7B running locally via Ollama — describes individuals in a given profession when asked using multiple prompt variations. The goal is to evaluate potential **gender bias** and **stereotypical framing** using NLP-based explainability methods.

The pipeline evaluates fairness and explainability by examining:
- How descriptive language varies across prompts
- How frequently gendered or neutral descriptions appear
- What adjectives dominate across generated samples
- Semantic differences between gendered vs. non-gendered descriptions (TBD)
- Independent judgment via LLM-as-a-Judge (TBD)

---

## 📦 Pipeline Structure

**Generation phase (`gen_sentence.py`):**
1. Generates open-ended descriptions of a given `PROFESSION` using multiple prompt variants.
2. Extracts **adjectives** in the responses → used to study dominant traits and build a word cloud.
3. Detects **gender references** → counts occurrences of male / female / non-gender descriptions.

**Embedding analysis (`embeddings.py`)** *(TBD)*:
4. Computes description similarity between gendered and non-gendered responses to produce a disparity score.

**Bias evaluation (`judge_llm.ipynb`)**:
5. Applies LLM-as-a-Judge to compare sentences and evaluate potential stereotypical or biased framing.

**Integration (`main.py`)** *(TBD)*:
6. Aggregates metrics and visualizations to enhance explainability (XAI) around the likelihood of bias in generated language.

---

## 🔧 Extending the Pipeline

This pipeline can be applied to other fairness evaluations by modifying:
- `SYSTEM` prompt
- `QUESTIONS`
- `PROFESSION`
- Gender detection keywords

---

## 🛠 Installation

### 1) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```
---

## Ensure that Ollama is running locally
```bash
ollama pull qwen3:1.7b
```