# Bahçe-sihir — Continuation Prompt

Paste everything below into a new conversation to continue work on this project.

---

I'm building **Bahçe-sihir**, a RAG-based Q&A assistant for Bahçeşehir University (BAU) students, in `/Users/khalid/Desktop/bahce-sihir`. It's based on a capstone proposal (`~/Downloads/Bahce-sihir_Capstone_Proposal.pdf`) but I've told my AI coding assistant explicitly the proposal is **not strict** — deviate freely when there's a good reason, but confirm with me first and explain why.

## Team / context
AI Engineering capstone at BAU, Faculty of Engineering and Natural Sciences. Team: Ali Hwala, Zubair Binshihon, Abdullah Yamin (me). Advisor: Ahmet Sayar.

## Current status: Phases 1-4 done, frontend built and working, Docker in progress

### Tech stack actually in use (with deviations from the proposal, and why)

- **Backend**: Python 3.11, FastAPI, `uvicorn`
- **LLM**: **Gemini 3.5-flash-lite** via `langchain-google-genai` (NOT gpt-4o-mini from the proposal, NOT gemini-2.5-flash, NOT gemini-flash-latest — see "Gemini model history" below for why)
- **Dense retrieval**: multilingual `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embeddings, but **NOT FAISS** — plain NumPy cosine similarity (`embeddings.npy` + `embeddings @ query_embedding`). FAISS was dropped because it conflicts with `torch`'s bundled OpenMP at the C library level and reproducibly segfaults the process when both are used together (confirmed by isolating every combination). At this corpus size (~1750 chunks, 384-dim), `IndexFlatIP` was already exact search anyway, so NumPy gives identical results with zero crash risk and, as a bonus, unblocked full multi-threaded reranking (the crash workaround had been silently forcing single-threaded execution, making reranking ~5x slower).
- **Sparse retrieval**: `rank_bm25`, with a **simple Turkish-aware tokenizer** (`src/text_utils.py`) — Turkish-correct lowercasing (İ/I handling) + regex tokenization + a small stopword list, **no stemming**. I tried `zeyrek` (morphological analyzer) first; it was ~475ms/chunk (too slow) AND picked wrong lemmas with no disambiguation (e.g. "kapsam"→"kapmak", a real word but wrong meaning) since it just returns the first unranked candidate parse. Wrong stems corrupting BM25 matches was worse than no stemming at all.
- **Reranking**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual) — NOT the proposal's `cross-encoder/ms-marco-MiniLM-L-6-v2` (English-only), since the corpus is 100% Turkish.
- **Frontend**: React + Vite (JavaScript, not TS), plain CSS (no Tailwind/UI kit) — see "Frontend" section below.
- **No Docker yet** — files are written but the image hasn't been built/tested (see "In progress" below).

### Gemini model history (why it's `gemini-3.5-flash-lite` specifically)
Started on `gemini-2.5-flash` per original plan → deprecated for new API keys → switched to `gemini-flash-latest` → that resolved to `gemini-3.6-flash`, which hit its free-tier cap of **20 requests/day** almost immediately during dev/testing (429 `ResourceExhausted`). Rate limits are scoped **per project per model**, so I switched to a *different* model with its own untouched quota: `gemini-3.5-flash-lite`, free tier **15 RPM / 1,500 RPD** (confirmed via web search) — 750 full pipeline queries/day at 2 LLM calls/query. It's also ~5x cheaper per token if billing is ever enabled, and turned out noticeably *faster* per call too. If quota errors come back, check which exact model name is in the 429 error before assuming the whole key is exhausted.

### Corpus / ingestion
65 official BAU regulation documents (64 PDF + 1 DOCX) in 4 categories (`yönergeler`, `usul ve esaslar`, `kurum yönetmelikleri`, `uygulama-program esasları`), in `data/raw/<category>/`. One PDF was a scanned image with zero extractable text (`ARAŞTIRMA VE YAYIN ETİĞİ YÖNERGESİ.pdf`) — rather than fight OCR tooling (Homebrew's `tesseract` install was taking hours compiling from source on this old Intel Mac), I read the page images directly with vision and hand-transcribed the text into a sibling `.txt` file next to the PDF; `src/ingestion/loaders.py` prefers a `.txt` override when one exists next to a PDF/DOCX. Known gap: the transcription has a small hole (item "g" missing from one list, and no visible "Madde 3" heading) that appears to be a defect in the original scan itself, not something fixable without a cleaner source copy.

**Chunking** (`src/ingestion/chunking.py`) is article-aware: splits primarily on `MADDE N -` boundaries (Turkish legal article markers), with chapter (`BÖLÜM`) headings tracked and prepended as context. For long articles, it further splits on numbered sub-clauses `(N)` at line starts — this was a deliberate fix after discovering a long article bundling 6 unrelated sub-rules into one chunk diluted its embedding enough to lose retrieval for a specific fact (the "max 4 courses in summer school" rule, buried as clause (9) in an article mostly about unrelated things). Confirmed this fixed the specific failure case; don't revert to blind character-count splitting for oversized articles.

Rebuild the index with:
```bash
source .venv/bin/activate
SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())") python3 -m src.ingestion.build_index
```
This regenerates `data/processed/{chunks.json,embeddings.npy}`. **Important**: rebuild BM25 separately (`python3 -m scripts.build_bm25_only`) if you ever see it crash when run in the same process as the embedding step — this used to reproducibly segfault before FAISS was removed; it's probably fine now but hasn't been stress-tested since.

### Retrieval pipeline (`src/retrieval/pipeline.py`)
1. **Query expansion** (`src/generation/query_expansion.py`): Gemini generates 3 Turkish + 3 English reformulations of the query (always both languages regardless of input language — confirmed decision, since BM25 is lexical-only and the corpus is 100% Turkish, so an English query needs Turkish variants to reach it via BM25 at all).
2. **Hybrid retrieval**: dense (batched — all query variants encoded in ONE `model.encode()` call, not N separate calls; calling `.encode()` in a loop was observed to degrade subsequent reranker throughput, not just add per-call overhead) + BM25, per variant.
3. **Reciprocal Rank Fusion** merges all ranked lists.
4. **Reranking**: top **6** fused candidates (`rerank_candidate_pool_size` in `src/config.py`) go through the cross-encoder. This was tuned carefully — tested 5/6/7/8/10 candidates; 5 was rejected because it equals `rerank_top_k`, meaning the reranker would have zero actual candidates to discard (pointless). 6 preserves at least one spare candidate while landing under the proposal's 7s latency target.
5. **`min_reranker_score = -100.0`** (effectively no filtering) — **do not set this back to 0.0**. Cross-encoder raw scores are a relative ranking signal, not a calibrated probability; a 0.0 threshold was found (via the evaluation harness) to silently discard ALL candidates for some legitimate queries, including cases where the correct document was ranked #1 but scored negative (e.g. -2.46). This was a real correctness bug, not a tuning nitpick — fixing it took Precision@5 from 0.313 to 0.707+ on the benchmark set.
6. **Generation** (`src/generation/generator.py` + `prompts.py`): Gemini answers using ONLY the provided context, must cite source doc + article for every claim, must say so if context is insufficient (don't guess), and **must respond in the same language as the query** — this instruction is repeated right before the "Answer:" marker in the prompt (not just stated once earlier), because with 100% Turkish context, the model was defaulting to Turkish even for English queries until the instruction was repeated at the point closest to generation.

### Latency
Went from ~30-37s/query originally down to **~6-10s** (varies by query) via: dropping FAISS (unblocked reranker threading), batching the embedding calls, switching to gemini-3.5-flash-lite (much faster per-call than the flash-lite predecessor), and tuning the rerank pool size. The proposal's <7s target is roughly met but not guaranteed every time — reranking on this machine's 2 physical CPU cores is still the dominant cost.

### Evaluation (Phase 4) — `scripts/` directory
- `benchmark_queries.json`: 15 hand-verified queries (5 factual, 4 procedural, 3 explanatory, 3 comparative), each with expected source document(s) and article(s), verified against actual corpus content.
- `evaluate_retrieval.py`: Precision@5 = 0.72 (document-level) / 0.147 (strict article-level), MRR = 0.947 / 0.550. Document-level is the metric that matters for the proposal's ≥0.80 target — the strict article-level number is naturally much lower since hand-picking a single "correct" article per query is often too narrow (multiple articles can legitimately answer one question).
- `ablation_study.py`: naive dense-only baseline (P@5=0.627, MRR=0.806) → + hybrid+expansion (0.707/0.922) → + reranking (0.707/0.950). Reranking's real contribution is *ordering*, not recall — it doesn't increase how many relevant docs land in top-5, it moves them higher.
- `evaluate_generation.py` + manual grading: **93.3% answer accuracy** (9 correct + 5 partially correct out of 15, matching proposal's ≥80% target), **0% hallucination rate** observed (well under the <10% target) — the two weakest cases had the system honestly decline to answer rather than fabricate, which is the correct safe-failure behavior.
- **Not measured**: Response Coherence (Likert) — needs a real 5-10 student panel per the proposal's own methodology, can't be done by an AI agent.
- Gemini's free tier is 15 RPM as well as 1,500 RPD — batch evaluation scripts need `time.sleep()` pacing between LLM-calling queries or you'll hit a 429 mid-run.

### Frontend (`frontend/`, React + Vite)
Built after a long back-and-forth on the design — final direction: **no hologram, no photo background** (both were tried and explicitly rejected). Current design: navy (`--navy-deep: #0a1a35`) + muted gold (`--gold: #cda36a`) palette, serif display face (`ui-serif, Georgia`) for the title/header, clean sans for body/UI. Sequence: title card ("Bahçeşehir Üniversitesi / University Assistant") → crest scales down + a single light-sweep effect crosses the screen → chat panel resolves into view (header → messages → input arrive in a quick stagger), then holds. Pure CSS `@keyframes` timeline (9s total), no JS-driven phase state — simpler and avoids animation/JS timing drift.
- `src/components/TitleCard.jsx` + `.css` — intro
- `src/components/ChatPanel.jsx` + `.css` — **fully functional real chat**, not a mockup: real message state, POST to backend `/query`, renders answer + source citation chips, loading dots while waiting, Enter-to-send
- `src/api.js` — fetch wrapper, `VITE_API_BASE` env var (defaults to `http://localhost:8000`)
- Backend needed CORS middleware added (`src/api/main.py` + `cors_allow_origins` in `src/config.py`, defaults to Vite's `localhost:5173`) for the frontend to call it — already done.
- **Verified working end-to-end** in the browser: real query → real backend → real Gemini answer with citations, rendered correctly.

To run both:
```bash
# Terminal 1 (backend)
source .venv/bin/activate && bash scripts/run_api.sh
# Terminal 2 (frontend)
cd frontend && npm run dev
```
Then open `http://localhost:5173`.

## Environment gotchas on this machine (Intel Mac, old Xcode CLT)
- **No precompiled Homebrew bottles for many formulas** — `tesseract`, `node`, `qemu`'s dependency chain (`swig`, `llvm`) all tried building from source, taking many minutes to hours. For Node specifically, I gave up on Homebrew and installed the **official prebuilt binary directly** from nodejs.org (`node-v24.19.0-darwin-x64.tar.gz`, extracted to `/usr/local/lib/nodejs/`, symlinked into `/usr/local/bin/`) — much faster, no compilation. Consider this approach first for any future tool that Homebrew wants to build from source here.
- **Never run two `brew install` commands concurrently** — they can deadlock on shared dependency locks (hit this exact issue: `node` and `qemu` both needed `ninja`, causing a lock conflict).
- `SSL_CERT_FILE` needs to be set to certifi's bundle for some Python network calls (model downloads, NLTK data) on this machine's Python — `SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")`.

## In progress / not done
1. **Docker**: `Dockerfile`, `.dockerignore`, `docker-compose.yml` are written (bundles `data/processed/` + pre-downloads model weights at build time so the container works offline afterward; `GEMINI_API_KEY` passed via `--env-file .env` or docker-compose's `env_file`, never baked into the image). NOT yet built or tested — was blocked on installing `colima` + `qemu` (Docker needs a Linux VM on macOS; `colima` is the lightweight CLI-only choice since Docker Desktop needs GUI interaction I can't automate). `docker` and `colima` themselves installed fine (precompiled bottles), but `qemu` was mid-install (building `swig`/`llvm` from source) when this conversation needed to end. **Next step**: check if `qemu` finished (`which qemu-img`), then `colima start`, then `docker build -t bahce-sihir .` and test with `docker run --env-file .env -p 8000:8000 bahce-sihir`.
2. Docker for the frontend hasn't been considered yet — only the backend is containerized so far.
3. **GitHub repo**: not created — proposal references a placeholder URL.
4. **No automated test suite** — `tests/` is an empty scaffold; proposal names pytest but nothing's written yet.
5. User mentioned the project **might get a full production deployment later** — not scoped, but worth keeping in mind (e.g. Gemini's free-tier rate limits would become a real constraint under genuine multi-student concurrent load; don't hard-code single-request-at-a-time assumptions unnecessarily).
6. The user wants the whole system to be able to **run locally and be shared via Docker** (explicit requirement) — this is the main reason Docker matters here, not just nice-to-have.

## Where to find more detail
This project has AI assistant memory files at `~/.claude/projects/-Users-khalid-Desktop-bahce-sihir/memory/` (`capstone_proposal.md`, `design_decisions.md`) that a fresh session in this same project directory should auto-load — they contain the same information as above in more detail, plus reasoning trails for each decision. If those aren't loading for some reason, this prompt should be self-sufficient.
