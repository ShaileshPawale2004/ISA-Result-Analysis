from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
import torch
import urllib.parse
from transformers import AutoTokenizer, AutoModelForCausalLM
from analysis_pipeline import process_single_division, process_multiple_divisions

app = Flask(__name__)

# Add URL encode filter for templates
@app.template_filter('urlencode')
def urlencode_filter(s):
    return urllib.parse.quote(str(s))

# Configure paths
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model and tokenizer (if available)
try:
    model_path = "models/tinyllama"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Successfully loaded the model and tokenizer")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model, tokenizer, device = None, None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'marks_file' not in request.files or 'qp_file' not in request.files:
        return 'No file uploaded', 400

    marks_file = request.files['marks_file']
    qp_file = request.files['qp_file']

    if marks_file.filename == '' or qp_file.filename == '':
        return 'No file selected', 400

    # Save uploaded files
    marks_path = os.path.join(UPLOAD_FOLDER, secure_filename(marks_file.filename))
    qp_path = os.path.join(UPLOAD_FOLDER, secure_filename(qp_file.filename))
    marks_file.save(marks_path)
    qp_file.save(qp_path)    # Process the files and generate analysis
    results = process_single_division(marks_path, qp_path, OUTPUT_FOLDER, model=model, tokenizer=tokenizer, device=device)

    # Extract subject name
    with open(qp_path, 'r', encoding='utf-8') as f:
        subject_name = f.readline().strip()
        if subject_name.lower().startswith('subject:'):
            subject_name = subject_name[8:].strip()
        else:
            subject_name = os.path.splitext(os.path.basename(qp_path))[0]
            
    # Extract division name from marks file (e.g., C_M1.csv -> C)
    division_name = os.path.basename(marks_path).split('_')[0]

    # Convert plot paths to relative paths for template and ensure forward slashes
    relative_plots = []
    for p in results['plots']:
        # Get the filename portion and ensure forward slashes
        filename = os.path.basename(p).replace('\\', '/')
        # Add to the list without the static/ prefix since url_for will add it
    relative_plots.append(f"outputs/{filename}")    

    # If model is not available, add a warning message
    if model is None:
        print("Warning: Model not available, GenAI insights will not be generated")
        for idx in results['summary'].index:
            results['summary'].at[idx, 'GenAI Insight'] = "⚠️ Model not available. Unable to generate insights."

    return render_template('results.html',
                         summary=results['summary'],
                         hs_ls_table=results['hs_ls_table'],
                         grouped_hs_ls=results['grouped_hs_ls'],
                         combined_scores=results['combined_scores'],
                         plots=relative_plots,
                         report_path=results['report_path'],
                         colored_table_path='outputs/colored_analysis_table.html',
                         excel_table_path='outputs/colored_analysis_table.xlsx',
                         subject_name=subject_name,
                         division_name=division_name)

@app.route('/upload_multiple', methods=['POST'])
def upload_multiple():
    if 'zip_file' not in request.files or 'qp_file' not in request.files:
        return 'No file uploaded', 400

    zip_file = request.files['zip_file']
    qp_file = request.files['qp_file']

    if zip_file.filename == '' or qp_file.filename == '':
        return 'No file selected', 400

    # Save uploaded files
    zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(zip_file.filename))
    qp_path = os.path.join(UPLOAD_FOLDER, secure_filename(qp_file.filename))
    zip_file.save(zip_path)
    qp_file.save(qp_path)    # Process multiple divisions
    results = process_multiple_divisions(zip_path, qp_path, OUTPUT_FOLDER)

    # Extract subject name
    with open(qp_path, 'r', encoding='utf-8') as f:
        subject_name = f.readline().strip()
        if subject_name.lower().startswith('subject:'):
            subject_name = subject_name[8:].strip()
        else:
            subject_name = os.path.splitext(os.path.basename(qp_path))[0]    # Convert plot paths to relative paths for template and ensure forward slashes
    relative_plots = []
    for p in results['plots']:
        # Get the filename portion and ensure forward slashes
        filename = os.path.basename(p).replace('\\', '/')
        # Add to the list without the static/ prefix since url_for will add it
        relative_plots.append(f"outputs/{filename}")
    
    return render_template('multi_results.html',
                         summary_df=results['summary_df'],
                         hs_ls_by_division=results['hs_ls_by_division'],
                         stats_by_division=results['stats_by_division'],
                         correlation_across_divisions=results['correlation_across_divisions'],
                         plots=relative_plots,
                         report_path=results['report_path'],
                         subject_name=subject_name)

@app.route('/download/<filename>')
def download(filename):
    """Handle downloads for reports and other files."""
    # Ensure the filename uses forward slashes
    safe_filename = filename.replace('\\', '/')
    return send_file(os.path.join(OUTPUT_FOLDER, safe_filename),
                    as_attachment=True,
                    download_name=safe_filename)

if __name__ == '__main__':
    # Configure Flask to ignore uploads directory when watching for changes
    extra_files = []
    for root, dirs, files in os.walk('.'):
        if 'uploads' in dirs:
            dirs.remove('uploads')  # Skip the uploads directory
        for filename in files:
            extra_files.append(os.path.join(root, filename))
    
    app.run(debug=True, use_reloader=False, extra_files=extra_files)  # Disable auto-reloader