# Adobe India Hackathon 2025 - Round 1B

## Approach
Processes PDFs to extract and rank sections/subsections relevant to a persona and job-to-be-done:
- **PDF Parsing**: `PyMuPDF` extracts text, detecting headings via font size, bold formatting, and keywords (e.g., "budget", "nightlife").
- **Relevance Scoring**: `SentenceTransformer` (`paraphrase-multilingual-MiniLM-L12-v2`) scores content using cosine similarity, prioritizing group-oriented, budget-friendly content.
- **Optimization**: `multiprocessing` ensures <60-second processing.
- **Multilingual Support**: Handles non-English PDFs for bonus points.

## Libraries
- PyMuPDF==1.23.26
- sentence-transformers==2.7.0
- numpy==1.26.4
- spacy==3.7.4
- langdetect==1.0.9

## Build and Run
1. Ensure Docker is installed.
2. Provide `input.json` and PDFs in a local `input_pdfs/` directory.
3. Build:
   ```bash
   docker build -t adobe-hackathon .