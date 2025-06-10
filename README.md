ISA Result Analysis

This repository contains a Flask application and analysis pipeline for processing ISA (In-Semester Assessment) score CSVs and a question-paper TXT to generate detailed analytics, visualizations, and AI-driven teaching insights via TinyLlama. The actual 4 GB TinyLlama weights are not included in this repository; see instructions below to download and place them locally.

PPT: https://docs.google.com/presentation/d/1doyihSiRKpnsu_P5z5kKuOToS3M9friI/edit?usp=drive_link&ouid=110483044580318446821&rtpof=true&sd=true

────────────────────────────────────────────────────────────
CONTENTS
────────────────────────────────────────────────────────────
1. Project Overview
2. Folder Structure
3. Prerequisites
4. Setup Instructions
   4.1. Clone the Repository
   4.2. Create and Activate a Python Virtual Environment
   4.3. Install Dependencies
   4.4. Download TinyLlama Weights
   4.5. Verify Folder Layout
   4.6. Start the Flask Application
5. Usage
   5.1. Single-Division Analysis
   5.2. Multiple-Division Analysis
6. File Descriptions
7. Contact / Support

────────────────────────────────────────────────────────────
1. PROJECT OVERVIEW
────────────────────────────────────────────────────────────
The “ISA Result Analysis” project automates the ingestion of raw student‐score CSVs (one or multiple divisions) along with the corresponding question‐paper TXT. It performs the following:
  •    Parses the question paper to extract question metadata (question number, max marks, CO, Bloom level, PI code).
  •    Reads per‐student marks CSV(s), computes question‐level averages, percentage attainment, attempted percentages, and performance classifications.
  •    Calculates highest-scorer vs. lowest-scorer metrics (HS/LS) and generates multiple charts: bar plots, line plots, violin plots, radar charts, performance‐band distributions, and Bloom’s pie charts.
  •    Optionally uses a TinyLlama (∼4 GB) model to generate AI‐powered “Possible Reasons / Suggested Actions” for questions with very low attainment (< 30%).
  •    Builds a comprehensive HTML + PDF report and also exports a colored Excel table for deeper review.
  •    Provides a simple Flask web interface for uploading files and displaying results.

────────────────────────────────────────────────────────────
2. FOLDER STRUCTURE
────────────────────────────────────────────────────────────
When you clone this repository and follow setup, you should have:

    ISA-Result-Analysis/
    ├── app.py
    ├── analysis_pipeline.py
    ├── requirements.txt
    ├── README.txt          ← (this file)
    ├── templates/
    │    ├── index.html
    │    ├── results.html
    │    └── multi_results.html
    ├── static/
    │    ├── css/
    │    │    └── styles.css
    │    └── (other assets, e.g. favicon, icons)
    ├── models/
    │    └── tinyllama/     ← This folder is empty until you download TinyLlama weights
    └── .gitignore

Notes:
  •    `models/tinyllama/` is currently empty. You will place the downloaded `model.safetensors` and associated files here.
  •    `static/uploads/` and `static/outputs/` will be auto‐created at runtime for user‐uploaded files and generated charts/reports.
  •    `templates/` contains the three HTML templates for the Flask web UI.

────────────────────────────────────────────────────────────
3. PREREQUISITES
────────────────────────────────────────────────────────────
  •    Python 3.8 or newer
  •    pip (Python package installer)
  •    (Recommended) A virtual‐environment tool, e.g. `venv`
  •    (Optional, for GenAI insights) A CUDA‐capable GPU if you want faster TinyLlama inference (CPU is supported but slower).
  •    Internet access to download TinyLlama from Google Drive.

────────────────────────────────────────────────────────────
4. SETUP INSTRUCTIONS
────────────────────────────────────────────────────────────

4.1 CLONE THE REPOSITORY
-------------------------
Open a terminal and run:
    git clone https://github.com/ShaileshPawale2004/ISA-Result-Analysis.git
    cd ISA-Result-Analysis

You should now be inside the project root directory:
    /path/to/ISA-Result-Analysis

4.2 CREATE AND ACTIVATE A PYTHON VIRTUAL ENVIRONMENT
----------------------------------------------------
(It’s best practice to isolate dependencies.)

On Windows (PowerShell):
    python -m venv venv
    .\venv\Scripts\Activate.ps1

On macOS/Linux:
    python3 -m venv venv
    source venv/bin/activate

You should see your prompt change, indicating the `venv` is active.

4.3 INSTALL DEPENDENCIES
------------------------
With the virtual environment active, run:
    pip install -r requirements.txt

This installs:
  •    flask
  •    numpy
  •    pandas
  •    matplotlib
  •    seaborn
  •    torch
  •    transformers
  •    tabulate
  •    openpyxl
  •    werkzeug
  •    zipfile36
  •    xhtml2pdf

4.4 DOWNLOAD TINYLLAMA WEIGHTS
------------------------------
The TinyLlama (~4 GB) model files are needed only if you want AI‐powered insights. Otherwise, the app still runs but will skip generating AI suggestions.

1. Go to the Google Drive folder:
   https://drive.google.com/drive/folders/1Zj9sNcSo1b5gDDW8hnOZ7zfj-eHEn7wF?usp=sharing

2. Download all files in that folder. You will get:
   - config.json
   - generation_config.json
   - model.safetensors    (≈ 4 GB)
   - special_tokens_map.json
   - tokenizer.json
   - tokenizer.model
   - tokenizer_config.json

3. In your local clone of this repo, create the `models/tinyllama/` directory if it doesn’t already exist:
   On Windows:
       mkdir models\tinyllama
   On macOS/Linux:
       mkdir -p models/tinyllama

4. Move (or copy) the downloaded files into `models/tinyllama/`. After this step, you should have:
    ISA-Result-Analysis/
    ├── models/
    │    └── tinyllama/
    │         ├── config.json
    │         ├── generation_config.json
    │         ├── model.safetensors
    │         ├── special_tokens_map.json
    │         ├── tokenizer.json
    │         ├── tokenizer.model
    │         └── tokenizer_config.json

5. Verify that `models/tinyllama/model.safetensors` is present. The app will automatically detect this folder and load TinyLlama when you run analyses.

4.5 VERIFY FOLDER LAYOUT
------------------------
At this point, your local repo should look like:

    ISA-Result-Analysis/
    ├── app.py
    ├── analysis_pipeline.py
    ├── requirements.txt
    ├── README.txt
    ├── templates/
    │    ├── index.html
    │    ├── results.html
    │    └── multi_results.html
    ├── static/
    │    └── css/
    │         └── styles.css
    ├── models/
    │    └── tinyllama/
    │         ├── config.json
    │         ├── generation_config.json
    │         ├── model.safetensors
    │         ├── special_tokens_map.json
    │         ├── tokenizer.json
    │         ├── tokenizer.model
    │         └── tokenizer_config.json
    └── .gitignore

Everything under `models/tinyllama/` is ignored by Git (because of `.gitignore`), and nothing under `models/` is tracked unless explicitly added (which you shouldn’t do).

4.6 START THE FLASK APPLICATION
-------------------------------
With your virtual environment still active, run:
    python app.py

You should see output like:
    * Serving Flask app "app"
    * Environment: production
    * Debug mode: on
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

Open your browser and navigate to:

    http://127.0.0.1:5000/

You will see the ISA Result Analysis homepage, where you can upload CSV/TXT files.

────────────────────────────────────────────────────────────
5. USAGE
────────────────────────────────────────────────────────────

5.1 SINGLE-DIVISION ANALYSIS
----------------------------
1. From the homepage, under “Single Division Analysis,” click “Choose File” next to “Marks CSV File.” Browse and select your CSV (e.g., `C_M1.csv`).
2. Click “Choose File” next to “Question Paper TXT File.” Select the corresponding question-paper TXT (e.g., `Computer_Networks-1.txt`).
3. Click “Analyze Single Division.”  
   • The server will process the files, generate tables and plots, and then render `results.html`.  
   • You will see:
     - A colored table (question-wise summary) in an embedded iframe.
     - AI‐generated insights (if TinyLlama is present and any question’s attainment < 30%).
     - Multiple chart images (HS-LS difference, violin plot, radar chart, performance band, Bloom’s pie).
     - A “Student Performance” table with combined totals and letter grades.
     - Buttons/links to download the full PDF report, the colored analysis HTML, and the student-scores CSV.

5.2 MULTIPLE-DIVISION ANALYSIS
------------------------------
1. From the homepage, under “Multiple Division Analysis,” click “Choose File” next to “ZIP File (Containing CSVs).” Select your ZIP (e.g. `all_div.zip` containing `A_M1.csv`, `B_M1.csv`, etc.).
2. Click “Choose File” next to “Question Paper TXT File.” Select the same question-paper TXT (e.g., `Computer_Networks-1.txt`).
3. Click “Analyze Multiple Divisions.”  
   • The server will unzip all CSVs, process each division, and render `multi_results.html`.  
   • You will see:
     - A division‐summary table (class strength, max/min/avg, Bloom-level averages).
     - Six chart images: average marks bar plot, box plot of marks distribution, correlation heatmap, line plot of average per question, aggregate performance‐band distribution, aggregate Bloom’s pie chart.
     - (Optional) You can uncomment and enable per‐division HS/LS tables and cross‐division correlation tables in the template if desired.

────────────────────────────────────────────────────────────
6. FILE DESCRIPTIONS
────────────────────────────────────────────────────────────

•   `app.py`  
    – The Flask server. Hosts three routes:  
      1. `/` (GET) → Renders `templates/index.html`.  
      2. `/upload` (POST) → Handles single‐division CSV + TXT uploads, calls `process_single_division`, and renders `results.html`.  
      3. `/upload_multiple` (POST) → Handles ZIP + TXT uploads, calls `process_multiple_divisions`, and renders `multi_results.html`.  
      4. `/download/<filename>` → Serves any file from `static/outputs/` as an attachment for download.  
    – On startup, attempts to load TinyLlama from `models/tinyllama/`. If not found, `model=None` and AI insights are skipped.

•   `analysis_pipeline.py`  
    – Core data processing and plotting logic for single and multiple divisions.  
    – Functions:  
      1. `parse_question_paper(qp_txt_path)`  
      2. `extract_subject_name(qp_txt_path)`  
      3. `generate_insight(...)` (uses TinyLlama if present)  
      4. `process_single_division(marks_csv_path, qp_txt_path, output_dir, model, tokenizer, device)`  
      5. `process_multiple_divisions(zip_path, qp_txt_path, output_dir)`  
    – Saves CSV summaries, chart PNGs, HTML tables, and a PDF report (using xhtml2pdf) under `static/outputs/`.

•   `requirements.txt`  
    – Lists all Python dependencies. Use `pip install -r requirements.txt` to install.

•   `templates/index.html`  
    – Homepage with two upload forms (single division and multiple divisions). Uses Bootstrap for styling.

•   `templates/results.html`  
    – Renders single‐division outputs: colored question‐wise table (iframe), AI insights, chart images, and a combined student‐scores table.

•   `templates/multi_results.html`  
    – Renders multi‐division outputs: division‐summary table and chart images. Optional sections (commented out) for per‐division HS/LS tables and correlation matrices.

•   `static/css/styles.css`  
    – Custom styling for templates (navbar gradients, card hover effects, etc.). Adjust as needed.

•   `models/tinyllama/`  
    – **Ignored by Git.** Place your TinyLlama model files here (downloaded from Google Drive).  
    – Files to include: `config.json`, `generation_config.json`, `model.safetensors`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`.

•   `.gitignore`  
    – Excludes: `models/tinyllama/`, Python bytecode, virtual‐envs, `static/uploads/`, `static/outputs/`, OS artifacts.

────────────────────────────────────────────────────────────
7. CONTACT / SUPPORT
────────────────────────────────────────────────────────────
If you encounter any issues or have questions, please open an issue on the GitHub repository:
    https://github.com/ShaileshPawale2004/ISA-Result-Analysis/issues

Happy analyzing!
