import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import defaultdict
from tabulate import tabulate
import zipfile
import tempfile
from matplotlib.colors import ListedColormap

def extract_subject_name(qp_txt_path):
    """Extract subject name from the question paper text file."""
    try:
        with open(qp_txt_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line.lower().startswith('subject:'):
                return first_line[8:].strip()  # Remove "subject:" and any whitespace
            return os.path.basename(qp_txt_path).split('.')[0]  # Fallback to filename
    except Exception as e:
        print(f"Error reading subject name: {e}")
        return os.path.basename(qp_txt_path).split('.')[0]  # Fallback to filename

# Set matplotlib to non-interactive backend to avoid display issues
plt.switch_backend('Agg')

def generate_insight(question, bloom, avg, maxm, att,Act_attemp, tokenizer, model, device):
    """Generate teaching insights using LLM."""
    prompt = f"""
    You are an experienced educational expert and teaching consultant.

    Analyze the following assessment result data carefully and generate actionable teaching insights.

    Question:
    \"\"\"{question}\"\"\"

    Bloom's Taxonomy Level: {bloom}
    Average Marks Achieved: {avg:.1f} out of {maxm}
    Attainment Percentage (among attempted students): {att:.1f}%
    Actual Attainment Percentage (including all students): {Act_attemp:.1f}%

    TASK:
    1. Identify and explain the **possible reasons why students performed at this level**.
    2. Describe **why these reasons are likely impacting student performance for this specific question**.
    3. Suggest **specific, practical, and actionable teaching strategies** that the instructor can apply to improve student understanding and performance in future assessments on this topic.
    4. Recommendations must be **directly linked to the question content and the observed data**.

    Ensure your response is:
    - Structured in clear sections: **Possible Reasons, Explanation, Suggested Actions**
    - Insightful and practical
    - Written in a professional, constructive tone

    Focus on why students may have struggled **with this specific question**, what could have been challenging at the given Bloom level, and how the teacher can enhance learning outcomes effectively.

    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.replace(prompt.strip(), "").strip()

def parse_question_paper(qp_txt_path):
    """Parse question paper TXT file and return metadata DataFrame."""
    with open(qp_txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(
        r"(?P<QNo>[1-9][a-z])\s*[\n:]?"
        r"(?P<Question>.*?)\s+"
        r"(?P<Marks>\d+)\s+"
        r"(?P<CO>\d+)\s+"
        r"(?P<BloomLevel>L\d+)\s+"
        r"(?P<PO>\d+)\s+"
        r"(?P<PICode>\d+\.\d+\.\d+)",
        re.DOTALL
    )
    matches = pattern.findall(text)
    df = pd.DataFrame(matches, columns=["Q.No", "Question", "Max Marks", "CO", "Bloom Level", "PO", "PI Code"])
    df["Q.No"] = df["Q.No"].str.lower()
    return df

def process_single_division(marks_csv_path, qp_txt_path, output_dir, model=None, tokenizer=None, device=None):
    """Process a single division's marks and question paper."""
    # Extract subject name
    subject_name = extract_subject_name(qp_txt_path)
      # Parse question paper and load marks
    qp_metadata = parse_question_paper(qp_txt_path)
    marks_df = pd.read_csv(marks_csv_path, dtype={'Roll.No': str})  # Keep Roll.No as string
    marks_only = marks_df.iloc[:, 1:].replace('-', pd.NA).apply(pd.to_numeric, errors='coerce')

    # Filter metadata and marks
    qp_metadata = qp_metadata[qp_metadata["Q.No"].isin(marks_only.columns)]
    question_labels = qp_metadata["Q.No"].tolist()
    marks_filtered = marks_only[question_labels]

    # Create summary DataFrame
    summary = pd.DataFrame(index=question_labels)
    summary["PI"] = qp_metadata.set_index("Q.No")["PI Code"]
    summary["Bloom"] = qp_metadata.set_index("Q.No")["Bloom Level"]
    summary["CO"] = qp_metadata.set_index("Q.No")["CO"].astype(int)
    summary["Max Marks"] = qp_metadata.set_index("Q.No")["Max Marks"].astype(int)
    summary["Avg Marks"] = marks_filtered.mean()
    summary["Attempted %"] = (marks_filtered.count() / len(marks_filtered)) * 100
    summary["Attainment %"] = (summary["Avg Marks"] / summary["Max Marks"]) * 100
    summary["Actual Attainment"] = (marks_filtered.sum() / (len(marks_filtered) * summary["Max Marks"])) * 100
    summary["Performance"] = summary["Attainment %"].apply(lambda x: "Excellent" if x >= 75 else ("Average" if x >= 60 else "Needs Improvement"))

    # Calculate percentage of low scorers for each question
    low_scorers = {}
    for q in question_labels:
        max_mark = summary.loc[q, "Max Marks"]
        scores = marks_filtered[q].dropna()
        low_threshold = 0.4 * max_mark  # 40% of max marks
        low_count = len(scores[scores <= low_threshold])
        low_scorers[q] = (low_count / len(scores)) * 100 if len(scores) > 0 else 0
    
    summary["% Low Scorers"] = pd.Series(low_scorers)

    # Generate AI insights if model is provided
    if model is not None and tokenizer is not None and device is not None:
        summary["GenAI Insight"] = ""
        for q in summary.index:
            question = qp_metadata.loc[qp_metadata["Q.No"] == q, "Question"].iloc[0]
            bloom = summary.loc[q, "Bloom"]
            avg = summary.loc[q, "Avg Marks"]
            maxm = summary.loc[q, "Max Marks"]
            att = summary.loc[q, "Actual Attainment"]
            Act_attemp = summary.loc[q, "Attempted %"]

            if att < 30:
                try:
                    insight = generate_insight(question, bloom, avg, maxm, att,Act_attemp, tokenizer, model, device)
                except:
                    insight = "⚠️ Generation failed"
            else:
                insight = "✅ Students did okay. No intervention needed."

            summary.at[q, "GenAI Insight"] = insight

    # Calculate HS and LS scores
    hs_scores, ls_scores = {}, {}
    for q in question_labels:
        max_m = summary.loc[q, "Max Marks"]
        scores = marks_filtered[q].dropna()
        non_zero_scores = scores[scores > 0]
        hs = scores.max() if not scores.empty else 0
        ls = non_zero_scores.min() if not non_zero_scores.empty else 0
        hs_scores[q] = round((hs / max_m) * 100, 1)
        ls_scores[q] = round((ls / max_m) * 100, 1)

    hs_ls_table = pd.DataFrame({
        "% HS": hs_scores,
        "% LS": ls_scores
    })
    hs_ls_table["%Difference (HS - LS)"] = hs_ls_table["% HS"] - hs_ls_table["% LS"]

    # Generate plots
    plots = []

    # 1. Bar Plot for HS and LS    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = range(len(hs_ls_table.index))
    plt.bar(index, hs_ls_table['% HS'], bar_width, label='Highest Score %', color='green')
    plt.bar([i + bar_width for i in index], hs_ls_table['% LS'], bar_width, label='Lowest Score %', color='red')
    plt.xlabel('Question')
    plt.ylabel('Score Percentage')
    plt.title('Highest and Lowest Scores per Question', pad=20)
    plt.xticks([i + bar_width/2 for i in index], hs_ls_table.index, rotation=45)
    plt.legend()
    plt.tight_layout(pad=3.0)  # Add more padding between plots
    hs_ls_bar_path = os.path.join(output_dir, 'hs_ls_bar_plot.png')
    plt.savefig(hs_ls_bar_path)
    plt.close()
    plots.append(hs_ls_bar_path)    
    
    # 2. Line Plot for Highest and Lowest Marks    
    plt.figure(figsize=(12, 8))
    # Convert percentages back to actual marks
    highest_marks = {q: (hs_ls_table.loc[q, '% HS'] * summary.loc[q, 'Max Marks']) / 100 for q in hs_ls_table.index}
    lowest_marks = {q: (hs_ls_table.loc[q, '% LS'] * summary.loc[q, 'Max Marks']) / 100 for q in hs_ls_table.index}
    
    plt.plot(hs_ls_table.index, list(highest_marks.values()), marker='o', color='green', label='Highest Marks', linewidth=2)
    plt.plot(hs_ls_table.index, list(lowest_marks.values()), marker='o', color='red', label='Lowest Marks', linewidth=2)
    plt.xlabel('Question')
    plt.ylabel('Marks')
    plt.title('Highest and Lowest Marks per Question', pad=20)
    plt.xticks(rotation=45)
    plt.grid(True)
    # Position legend in top right corner with slight offset
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)  # Make space for legend
    plt.tight_layout(pad=3.0)  # Add padding between plots
    hs_ls_diff_path = os.path.join(output_dir, 'hs_ls_diff_line_plot.png')
    plt.savefig(hs_ls_diff_path)
    plt.close()
    plots.append(hs_ls_diff_path)

    # 3. Grouped Bar Plot by Main Question
    grouped_hs_ls = pd.DataFrame(index=['Q1', 'Q2', 'Q3'])
    for group in ['Q1', 'Q2', 'Q3']:
        sub_questions = [q for q in hs_ls_table.index if q.startswith(group[1])]
        grouped_hs_ls.loc[group, '% HS'] = hs_ls_table.loc[sub_questions, '% HS'].mean()
        grouped_hs_ls.loc[group, '% LS'] = hs_ls_table.loc[sub_questions, '% LS'].mean()    
        plt.figure(figsize=(12, 8))
    bar_positions = np.arange(len(grouped_hs_ls.index))
    plt.bar(bar_positions, grouped_hs_ls['% HS'], bar_width, label='Highest Score %', color='green')
    plt.bar(bar_positions + bar_width, grouped_hs_ls['% LS'], bar_width, label='Lowest Score %', color='red')
    plt.xlabel('Main Question')
    plt.ylabel('Score Percentage')
    plt.title('Highest and Lowest Scores by Main Question', pad=20)
    plt.xticks(bar_positions + bar_width/2, grouped_hs_ls.index)
    plt.legend()
    plt.tight_layout(pad=3.0)  # Add more padding between plots
    main_question_bar_path = os.path.join(output_dir, 'hs_ls_main_question_bar_plot.png')
    plt.savefig(main_question_bar_path)
    plt.close()
    plots.append(main_question_bar_path)

    # 4. Violin Plot with LS points    plt.figure(figsize=(12, 8))
    sns.violinplot(data=marks_filtered, cut=0)
    plt.xlabel('Question')
    plt.ylabel('Marks')
    plt.title('Student Score Distribution for Each Question', pad=20)
    for i, q in enumerate(marks_filtered.columns):
        ls = marks_filtered[q][marks_filtered[q] > 0].min() if (marks_filtered[q] > 0).any() else 0
        plt.scatter([i], [ls], color='red', s=50, label='Lowest Score' if i == 0 else "")
    plt.legend()
    plt.xticks(range(len(marks_filtered.columns)), marks_filtered.columns, rotation=45)
    plt.tight_layout(pad=3.0)  # Add more padding between plots
    violin_path = os.path.join(output_dir, 'violin_plot.png')
    plt.savefig(violin_path)
    plt.close()
    plots.append(violin_path)

    # 5. Radar Chart
    labels = hs_ls_table.index.tolist()
    hs_values = hs_ls_table['% HS'].values.tolist()
    ls_values = hs_ls_table['% LS'].values.tolist()
    
    # Add the first value at the end to close the radar chart
    hs_values += hs_values[:1]
    ls_values += ls_values[:1]
    labels += labels[:1]
    num_vars = len(labels) - 1
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Create a larger figure with more space for the legend
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, hs_values, linewidth=2, linestyle='solid', label='Highest Score (%)', color='green')
    ax.fill(angles, hs_values, 'green', alpha=0.25)
    ax.plot(angles, ls_values, linewidth=2, linestyle='solid', label='Lowest Score (%)', color='red')
    ax.fill(angles, ls_values, 'red', alpha=0.25)
    
    # Configure axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
      # Add title with increased padding and line break
    plt.title('Comparison of Highest and Lowest Score Percentages per Question', size=14, pad=30)
    
    # Place legend further to the right to avoid overlap
    ax.legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
    
    # Adjust layout to accommodate legend
    plt.subplots_adjust(right=0.7)
    
    radar_path = os.path.join(output_dir, 'radar_chart_hs_ls.png')
    plt.savefig(radar_path)
    plt.close()
    plots.append(radar_path)

    # Save summary files
    summary.to_csv(os.path.join(output_dir, 'question_summary.csv'), index=True)
    hs_ls_table.to_csv(os.path.join(output_dir, 'hs_ls_summary.csv'), index=True)
    grouped_hs_ls.to_csv(os.path.join(output_dir, 'grouped_hs_ls_summary.csv'), index=True)

    # Calculate student scores
    question_groups = defaultdict(list)
    for q in summary.index:
        question_groups[f"Q{q[0]}"].append(q)
    
    combined_scores = pd.DataFrame(index=marks_df.index)
    for group, subparts in question_groups.items():
        combined_scores[group] = marks_filtered[subparts].sum(axis=1, skipna=True)
    combined_scores["Total /40"] = combined_scores.apply(lambda row: sum(sorted(row, reverse=True)[:2]), axis=1)
    combined_scores["Total /20"] = round((combined_scores["Total /40"] / 40) * 20, 2)
    combined_scores.insert(0, "Roll No", marks_df["Roll.No"])  # Use the Roll.No column directly
    
    def assign_grade(score):
        if score >= 17:
            return "S"
        elif score >= 12:
            return "A"
        elif score >= 8:
            return "B"
        elif score >= 5:
            return "C"
        else:
            return "D"
    
    combined_scores["Grade"] = combined_scores["Total /20"].apply(assign_grade)
    combined_scores.to_csv(os.path.join(output_dir, 'student_scores.csv'), index=False)

    # Performance Band Distribution
    all_scores = marks_filtered.sum(axis=1)
    band_chart_path, band_counts = generate_performance_band_distribution(all_scores, output_dir)
    plots.append(band_chart_path)

    # Bloom's Level Analysis
    bloom_chart_path, bloom_grouped = generate_bloom_level_analysis(summary, "Single", output_dir)
    plots.append(bloom_chart_path)

    # Add interpretation for Bloom's level
    level_pct = (bloom_grouped['bloom_total_marks'] / bloom_grouped['bloom_total_marks'].sum() * 100).round(1)
    l1 = level_pct.get('L1', 0)
    l2 = level_pct.get('L2', 0)
    l3 = level_pct.get('L3', 0)

    interpretation = f"""
🧠 Bloom’s Level Performance Interpretation

- L1 (Recall): {l1}%
- L2 (Understanding/Application): {l2}%
- L3 (Analysis/Evaluation): {l3}%

This distribution shows that:
"""
    if l3 > max(l1, l2):
        interpretation += "Students are scoring highest in L3, indicating strong performance in analytical and evaluative questions."
    elif l2 > max(l1, l3):
        interpretation += "Most marks are scored in L2, suggesting assessments are focused on understanding and application."
    elif l1 > max(l2, l3):
        interpretation += "Majority of scores lie in L1, pointing to a focus on recall-level questions."
    else:
        interpretation += "Performance is fairly distributed across cognitive levels."

    interpretation += "\n\n> These insights can help refine assessment balance and teaching focus."
    summary["Bloom Interpretation"] = interpretation    # Generate PDF report
    report_title = f"Single Division Analysis Report - {subject_name}"
    additional_data = {
        "Subject": subject_name,
        "High vs Low Scorers Analysis": hs_ls_table,
        "Main Question Analysis": grouped_hs_ls,
        "Student Performance": combined_scores
    }
    
    if model is not None:
        additional_data["AI-Generated Teaching Insights"] = "<br>".join([
            f"<p><strong>Question {q}:</strong> {summary.loc[q, 'GenAI Insight']}</p>"
            for q in summary.index
            if summary.loc[q, 'GenAI Insight'] != "✅ Students did okay. No intervention needed."
        ])

    html_content = generate_report_html(
        report_title,
        summary,
        [os.path.join(output_dir, os.path.basename(p)) for p in plots],
        additional_data
    )
    
    report_path = os.path.join(output_dir, 'single_division_report.pdf')
    generate_pdf_report(report_title, html_content, report_path)
    
    # Generate additional chart and table for single division
    single_division_chart_path, single_division_table_path = generate_single_division_chart_and_table(summary, output_dir)
    plots.append(single_division_chart_path)
    plots.append(single_division_table_path)

    # Generate colored analysis table
    colored_table_path, excel_table_path = generate_single_division_table(summary, qp_metadata, output_dir)

    return {
        'summary': summary,
        'hs_ls_table': hs_ls_table,
        'grouped_hs_ls': grouped_hs_ls,
        'combined_scores': combined_scores,
        'plots': plots,
        'report_path': os.path.basename(report_path),
        'colored_table_path': colored_table_path,
        'excel_table_path': excel_table_path
    }

def generate_performance_band_distribution(all_scores, output_dir):
    """Generate performance band distribution graph with actual marks ranges and color gradient."""
    max_score = 40  # Adjust as needed    # Define marks ranges and labels for better granularity
    bins = [0, 8, 16, 24, 32, 40]  # Actual marks ranges
    labels = ["0-8", "9-16", "17-24", "25-32", "33-40"]
    
    # Color gradient from red (poor) to green (excellent)
    colors = ['#ff4d4d', '#ff9933', '#ffdb4d', '#99cc33', '#4dff4d']  # Red to green gradient
    performance_levels = ['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
    
    performance_band = pd.cut(all_scores, bins=bins, labels=labels, right=True, include_lowest=True)
    band_counts = performance_band.value_counts().sort_index()

    # Create a DataFrame for plotting with additional range information
    plot_data = pd.DataFrame({
        'Band': band_counts.index,
        'Count': band_counts.values,
        'Color': colors[:len(band_counts)],
        'Level': performance_levels[:len(band_counts)],
        'Range': labels  # Add the range labels explicitly
    })
    plt.figure(figsize=(12, 7))
    # Create bar plot with positions explicitly defined
    x_positions = np.arange(len(plot_data))
    bars = plt.bar(x_positions, plot_data['Count'], color=plot_data['Color'])
    
    # Add student count above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    # Customize the plot
    plt.title("Performance Distribution Across Marks Ranges", fontsize=14, pad=20)
    plt.xlabel("Marks Range", fontsize=12, labelpad=15)
    plt.ylabel("Number of Students", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Set x-axis labels to the range values directly
    plt.xticks(x_positions, labels, rotation=0)
    
    # Adjust layout
    plt.margins(y=0.15)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.15)
    
    # Add a custom legend for performance levels
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=level) 
                      for color, level in zip(plot_data['Color'], plot_data['Level'])]
    plt.legend(handles=legend_elements, title="Performance Levels", 
              loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    band_chart_path = os.path.join(output_dir, "performance_band_distribution.png")
    plt.savefig(band_chart_path, bbox_inches='tight', dpi=300)
    plt.close()

    return band_chart_path, band_counts

def generate_bloom_level_analysis(summary, division_name, output_dir):
    """Generate Bloom's level analysis and pie chart."""    
    if "L1 Average" in summary.columns:
        # For multiple divisions summary
        # Convert N/A to 0 for numeric calculations
        l1_values = pd.to_numeric(summary["L1 Average"].replace("N/A", 0))
        l2_values = pd.to_numeric(summary["L2 Average"].replace("N/A", 0))
        l3_values = pd.to_numeric(summary["L3 Average"].replace("N/A", 0))
        
        bloom_data = pd.DataFrame({
            'L1': [l1_values.mean()],
            'L2': [l2_values.mean()],
            'L3': [l3_values.mean()]
        }).T
        bloom_grouped = pd.DataFrame({
            'bloom_total_marks': bloom_data[0],
            'bloom_avg_per_question': bloom_data[0],
            'bloom_question_count': pd.Series(len(summary), index=bloom_data.index)
        })
    else:
        # For single division summary
        bloom_grouped = summary.groupby("Bloom").agg(
            bloom_total_marks=('Avg Marks', 'sum'),
            bloom_avg_per_question=('Avg Marks', 'mean'),
            bloom_question_count=('Avg Marks', 'count')
        ).round(2)

    plt.figure(figsize=(6, 6))
    plt.pie(
        bloom_grouped['bloom_total_marks'],
        labels=bloom_grouped.index,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title(f"Bloom’s Level Distribution (L1, L2, L3)")
    plt.axis('equal')
    plt.tight_layout()
    bloom_chart_path = os.path.join(output_dir, f"bloom_div_{division_name}.png")
    plt.savefig(bloom_chart_path)
    plt.close()

    return bloom_chart_path, bloom_grouped

def generate_report_html(title, summary_df, plots, additional_data=None):
    """Generate HTML content for the report."""
    subject_name = additional_data.get("Subject", "") if additional_data else ""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            .subject-header {{ 
                text-align: center; 
                font-size: 1.5em; 
                color: #34495e; 
                margin: 20px 0; 
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f5f6fa; }}
            .visual-analysis {{ 
                display: flex; 
                flex-direction: column; 
                gap: 50px;
                margin: 40px 0;
            }}
            .plot-container {{ 
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .plot-container h3 {{ 
                color: #2c3e50;
                text-align: center;
                margin-bottom: 20px;
                font-size: 1.2em;
            }}
            .plot-container img {{ 
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
            .insight {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #4e73df; margin: 20px 0; }}
            .warning {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; }}
            .success {{ background-color: #d4edda; padding: 15px; border-left: 4px solid #28a745; }}
        </style>
    </head>    <body>
        <h1>{title}</h1>
        <div class="subject-header">Subject: {subject_name}</div>
    """

    # Add summary table
    html += "<h2>Summary Analysis</h2>"
    html += summary_df.to_html(classes='table', index=True)    # Add plots in vertical layout
    html += "<h2>Visual Analysis</h2>"
    html += '<div class="plots-vertical-layout">'
    for plot in plots:
        plot_name = os.path.basename(plot).replace('_', ' ').replace('.png', '').title()
        html += f"""
        <div class="plot-container">
            <h3>{plot_name}</h3>
            <img src="{plot}" alt="{plot_name}" style="max-width: 100%; height: auto;">
        </div>
        """
    html += '</div>'

    # Add additional data if provided
    if additional_data:
        for section_title, content in additional_data.items():
            html += f"<h2>{section_title}</h2>"
            if isinstance(content, pd.DataFrame):
                html += content.to_html(classes='table', index=True)
            else:
                html += f"<div class='insight'>{content}</div>"

    html += """
    </body>
    </html>
    """
    return html

def generate_pdf_report(title, html_content, output_path):
    """Convert HTML to PDF and save it."""
    from xhtml2pdf import pisa

    with open(output_path, "w+b") as result_file:
        pisa.CreatePDF(html_content, dest=result_file)
        return output_path

def process_multiple_divisions(zip_path, qp_txt_path, output_dir):
    """Process multiple divisions from a ZIP file and return aggregated results."""
    # Extract subject name
    subject_name = extract_subject_name(qp_txt_path)
    
    # Parse question paper metadata to get Bloom Levels
    qp_metadata = parse_question_paper(qp_txt_path)
    bloom_levels = qp_metadata.set_index("Q.No")["Bloom Level"]
    
    # Get max marks from qp_metadata instead of hardcoding
    max_marks = qp_metadata.set_index("Q.No")["Max Marks"].astype(int).to_dict()
    summary = pd.DataFrame({
        'Max Marks': max_marks,
        'BloomLevel': bloom_levels
    })
    
    # Create a mapping of questions to their Bloom levels
    bloom_level_map = bloom_levels.to_dict()
    
    question_labels = list(max_marks.keys())
    division_stats = {}
    summary_table = []
    hs_ls_by_division = {}
    stats_by_division = {}
    question_avg_by_division = {}
    all_scores_list = []  # New list to store all scores

    # Process each division's data
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for filename in os.listdir(temp_dir):
            if filename.endswith(".csv"):
                division = filename.split("_")[0].upper()
                file_path = os.path.join(temp_dir, filename)
                df = pd.read_csv(file_path)
                df = df.drop_duplicates(subset=["Roll.No", "USN"])
                question_cols = [col for col in df.columns if col in question_labels]
                df_marks = df[question_cols].replace("-", pd.NA).apply(pd.to_numeric, errors='coerce')
                df_marks = df_marks.dropna(how='all')
                total_per_student = df_marks.sum(axis=1)
                
                # Store division stats and total scores
                division_stats[division] = total_per_student
                all_scores_list.append(total_per_student)

                # Calculate averages by Bloom Level
                l1_qs = [q for q, level in bloom_level_map.items() if level == 'L1']
                l2_qs = [q for q, level in bloom_level_map.items() if level == 'L2']
                l3_qs = [q for q, level in bloom_level_map.items() if level == 'L3']                
                l1_avg = df_marks[l1_qs].mean().sum() if l1_qs else 0
                l2_avg = df_marks[l2_qs].mean().sum() if l2_qs else 0
                l3_avg = df_marks[l3_qs].mean().sum() if l3_qs else 0                # Store numerical values for calculations, display "N/A" for presentation
                l1_display = "N/A" if l1_avg == 0 else round(l1_avg, 2)
                l2_display = "N/A" if l2_avg == 0 else round(l2_avg, 2)
                l3_display = "N/A" if l3_avg == 0 else round(l3_avg, 2)
                
                summary_table.append({
                    "Division": division,
                    "Class Strength": len(total_per_student),
                    "Max Score": round(total_per_student.max(), 2),
                    "Min Score": round(total_per_student.min(), 2),
                    "Average": round(total_per_student.mean(), 2),
                    "Standard Deviation": round(total_per_student.std(), 2),
                    "L1 Average": l1_display,
                    "L2 Average": l2_display,
                    "L3 Average": l3_display
                })

                # Process HS/LS scores and other statistics
                high_scores, low_scores, hs_ls_diff, stats = {}, {}, {}, {}
                for q in question_cols:
                    max_mark = summary.loc[q, 'Max Marks']
                    q_scores = df_marks[q].dropna()
                    hs = q_scores.max() if not q_scores.empty else 0
                    ls = q_scores[q_scores > 0].min() if not q_scores.empty else 0
                    hs_pct = (hs / max_mark) * 100 if max_mark > 0 else 0
                    ls_pct = (ls / max_mark) * 100 if max_mark > 0 else 0
                    high_scores[q] = round(hs_pct, 1)
                    low_scores[q] = round(ls_pct, 1)
                    hs_ls_diff[q] = round(hs_pct - ls_pct, 1)
                    stats[q] = {
                        'Min': q_scores.min() if not q_scores.empty else 0,
                        'Max': q_scores.max() if not q_scores.empty else 0,
                        'Avg': q_scores.mean() if not q_scores.empty else 0,
                        'Std Dev': q_scores.std() if not q_scores.empty and len(q_scores) > 1 else 0
                    }

                hs_ls_by_division[division] = pd.DataFrame({
                    "% HS": high_scores,
                    "% LS": low_scores,
                    "%Difference (HS - LS)": hs_ls_diff
                }).round(1)
                stats_by_division[division] = pd.DataFrame(stats).T.round(1)
                question_avg_by_division[division] = df_marks.mean().round(1)    # Prepare summary DataFrame 
    summary_df = pd.DataFrame(summary_table).set_index("Division").sort_index()    # Generate Plots
    plots = []    
    
    # 1. Bar Plot for Average Total Marks
    # Sort dataframe by average marks and reset index to get Division as column
    summary_df_sorted = summary_df.sort_values('Average').reset_index()

    plt.figure(figsize=(12, 6))

    # Use a diverging palette mapped to sorted values
    palette = sns.color_palette("RdYlGn", len(summary_df_sorted))    
    bars = plt.bar(
        summary_df_sorted['Division'],  # Use the Division column directly
        summary_df_sorted['Average'],
        color=palette
    )

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.title("Average Total Marks per Division (Sorted by Performance)")
    plt.ylabel("Marks")
    plt.xlabel("Division")
    plt.grid(axis='y', alpha=0.3)

    # Add a horizontal line for overall average
    overall_avg = summary_df['Average'].mean()
    plt.axhline(overall_avg, color='blue', linestyle='--', linewidth=1.5, label=f'Overall Average: {overall_avg:.1f}')
    plt.legend()

    plt.tight_layout()
    avg_marks_path = 'outputs/avg_marks_division_bar.png'
    plt.savefig(os.path.join(output_dir, os.path.basename(avg_marks_path)), dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(avg_marks_path)    
    
    # 2. Box Plot for Marks Distribution
    all_data = pd.DataFrame(division_stats)
    
    # Calculate average marks for each division to determine colors
    div_averages = {div: data.mean() for div, data in division_stats.items()}
    sorted_divs = sorted(div_averages.keys(), key=lambda x: div_averages[x])
    
    # Create color palette from red to green based on performance
    colors = sns.color_palette("RdYlGn", len(sorted_divs))
    # Create a mapping of division to color
    color_map = dict(zip(sorted_divs, colors))
    
    plt.figure(figsize=(12, 6))
    # Create box plot with custom colors
    boxplot = sns.boxplot(data=all_data, palette=color_map)
    
    plt.title("Distribution of Total Marks by Division")
    plt.ylabel("Marks")
    plt.xlabel("Division")
    plt.grid(True, alpha=0.3)
    
    # Add division averages as text above each box
    for i, div in enumerate(all_data.columns):
        avg = div_averages[div]
        plt.text(i, all_data[div].max() + 1, f'Avg: {avg:.1f}', 
                horizontalalignment='center', fontsize=9)

    plt.tight_layout()
    box_plot_path = 'outputs/marks_distribution_box.png'
    plt.savefig(os.path.join(output_dir, os.path.basename(box_plot_path)), 
                dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(box_plot_path)    
    
    # 3. Heatmaps for Cross-Division Correlation
    # Question-wise correlation heatmap
    combined_avg_data = pd.DataFrame(question_avg_by_division)
    correlation_across_divisions = combined_avg_data.corr().round(2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_across_divisions, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Cross-Division Correlation Heatmap (Question-wise Scores)")
    heatmap_path = 'outputs/correlation_heatmap.png'
    plt.savefig(os.path.join(output_dir, os.path.basename(heatmap_path)), dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(heatmap_path)    
    
    # # Bloom's level correlation heatmap
    # bloom_df = pd.DataFrame(summary_table).set_index('Division')[['L1 Average', 'L2 Average', 'L3 Average']]
    
    # # Debug print to verify data
    # print("Bloom's Level Data:")
    # print(bloom_df)
    
    # # Ensure we have numeric data
    # bloom_df = bloom_df.apply(pd.to_numeric, errors='coerce')
    
    # # Only create correlation if we have valid data
    # if not bloom_df.empty and not bloom_df.isna().all().all():
    #     bloom_correlation = bloom_df.corr().round(2)  # Note: removed .T since we want division correlations
        
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(bloom_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    #     plt.title("Correlation Heatmap of Divisions Based on Bloom's Level Averages")
    #     plt.tight_layout()
    #     heatmap_bloom_path = 'outputs/correlation_heatmap_bloom_levels.png'
    #     plt.savefig(os.path.join(output_dir, os.path.basename(heatmap_bloom_path)), dpi=300, bbox_inches='tight')
    #     plt.close()
    #     plots.append(heatmap_bloom_path)
    # else:
    #     print("Warning: No valid data for Bloom's level correlation heatmap")

    # 4. Line Plot for Per-Question Average Scores
    combined_avg_data.T.plot(figsize=(12, 6), marker='o')
    plt.title("Average Marks per Question Across Divisions")
    plt.ylabel("Marks")
    plt.xlabel("Question")
    plt.legend(title="Division")
    plt.grid(True)
    avg_scores_path = 'outputs/avg_scores_question_line.png'
    plt.savefig(os.path.join(output_dir, os.path.basename(avg_scores_path)))
    plt.close()
    plots.append(avg_scores_path)

    # Save Summary
    summary_df.to_excel(os.path.join(output_dir, 'ISA_Division_Wise_Summary.xlsx'), index=False)

    # Performance Band Distribution for all divisions
    all_scores = pd.concat(all_scores_list, ignore_index=True)  # Use the collected scores list
    band_chart_path, band_counts = generate_performance_band_distribution(all_scores, output_dir)
    plots.append(band_chart_path)

    # Bloom's Level Analysis for all divisions
    bloom_chart_path, bloom_grouped = generate_bloom_level_analysis(summary_df, "All Divisions", output_dir)
    plots.append(bloom_chart_path)    # Generate PDF report
    report_title = f"Multiple Division Analysis Report - {subject_name}"
    additional_data = {
        "Subject": subject_name,
        "Division-wise High vs Low Scorers": hs_ls_by_division,
        "Statistical Metrics by Division": stats_by_division,
        "Cross-Division Correlation": correlation_across_divisions
    }

    html_content = generate_report_html(
        report_title,
        summary_df,
        [os.path.join(output_dir, os.path.basename(p)) for p in plots],
        additional_data
    )
    
    report_path = os.path.join(output_dir, 'multiple_division_report.pdf')
    generate_pdf_report(report_title, html_content, report_path)
    
    return {
        'summary_df': summary_df,
        'hs_ls_by_division': hs_ls_by_division,
        'stats_by_division': stats_by_division,
        'correlation_across_divisions': correlation_across_divisions,
        'plots': plots,
        'report_path': os.path.basename(report_path)
    }

def generate_single_division_chart_and_table(data, output_dir):
    """Generate a bar chart and color-filled table for single division analysis."""
    # Generate the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(data.index, data['Attempted %'], color=['green' if x >= 70 else 'yellow' if x >= 40 else 'red' for x in data['Attempted %']])
    ax.set_xlabel('Question Number')
    ax.set_ylabel('Attempted %')
    ax.set_title('Question-wise Attempt Percentage')
    plt.xticks(rotation=45)    # Add more space at the bottom and use high DPI for better quality
    plt.tight_layout(pad=3.0)
    chart_path = os.path.join(output_dir, 'single_division_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Style the table
    def color_cells(val):
        if pd.isna(val):
            return ''
        att_val = float(val)
        if att_val >= 70:
            return 'background-color: #90EE90'  # Light green
        elif att_val >= 40:
            return 'background-color: #FFFFE0'  # Light yellow
        else:
            return 'background-color: #FFB6C1'  # Light red

    styled_table = data.style.map(color_cells, subset=['Attempted %'])
    table_path = os.path.join(output_dir, 'single_division_table.html')
    styled_table.to_html(table_path, index=False)

    return chart_path, table_path

def generate_single_division_table(summary_df, qp_metadata, output_dir):
    """Generate a table with emoji indicators for single division analysis."""
    # Map the internal column names to the ones we want in the output
    table_data = pd.DataFrame({
        'Q.NO': summary_df.index,
        'Attempt %': summary_df['Attempted %'].round(2),
        'Attempt Status': [''] * len(summary_df),
        'Attainment %': summary_df['Attainment %'].round(2),
        'Attainment Status': [''] * len(summary_df),
        'Analysis': summary_df['Performance'].map({'Excellent': 'Excellent', 'Average': 'Average', 'Needs Improvement': 'Needs Improvement-L2'}),
        'Question Paper Analysis': qp_metadata.set_index('Q.No')['Question']
    })

    # Add Status indicators with emojis for both attempt and attainment
    for idx in table_data.index:
        # Handle Attempt Status
        att_val = table_data.loc[idx, 'Attempt %']
          # Set Status for Attempt %
        if att_val >= 70:
            table_data.loc[idx, 'Attempt Status'] = '<span style="font-size: 32px;">🟢</span>'  # Good attempt rate
        elif att_val >= 40:
            table_data.loc[idx, 'Attempt Status'] = '<span style="font-size: 32px;">🟡</span>'  # Moderate attempt rate
        else:
            table_data.loc[idx, 'Attempt Status'] = '<span style="font-size: 32px;">🔴</span>'  # Low attempt rate
            
        # Set Status for Attainment %
        attainment_val = table_data.loc[idx, 'Attainment %']
        if attainment_val >= 70:
            table_data.loc[idx, 'Attainment Status'] = '<span style="font-size: 32px;">🟢</span>'  # High attainment
        elif attainment_val >= 40:
            table_data.loc[idx, 'Attainment Status'] = '<span style="font-size: 32px;">🟡</span>'  # Moderate attainment
        else:
            table_data.loc[idx, 'Attainment Status'] = '<span style="font-size: 32px;">🔴</span>'

    # Custom CSS for the table with consistent font sizes
    html_content = '''
    <style>
        :root {
            --table-font-family: Arial, sans-serif;
            --base-font-size: 14px;
            --header-font-size: 16px;
            --emoji-font-size: 32px;
            --numeric-font-size: 15px;
            --question-font-size: 13px;
            --analysis-font-size: 14px;
        }
        
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            font-family: var(--table-font-family) !important;
        }
        
        /* Base styles for all cells */
        th, td { 
            border: 1px solid #333;
            padding: 12px 8px; 
            text-align: center;
            vertical-align: middle;
        }
        
        /* Header specific styles */
        th { 
            background-color: #4a90e2;
            color: white;
            font-weight: bold;
            font-size: var(--header-font-size) !important;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        
        /* Question number column */
        td:nth-child(1) {
            font-size: var(--base-font-size) !important;
            font-weight: bold;
        }
        
        /* Numeric value columns */
        td:nth-child(2),
        td:nth-child(4) {
            font-size: var(--numeric-font-size) !important;
            font-weight: 500;
        }
        
        /* Status emoji columns */
        td:nth-child(3),
        td:nth-child(5) {
            font-size: var(--emoji-font-size) !important;
            line-height: 1;
            width: 60px;
            padding: 8px;
        }
        
        /* Analysis column */
        td:nth-child(6) {
            font-size: var(--analysis-font-size) !important;
            font-weight: 500;
        }
        
        /* Question text column */
        td:nth-child(7) {
            font-size: var(--question-font-size) !important;
            text-align: left;
            padding-left: 15px;
            line-height: 1.4;
        }
        
        /* Last column */
        td:nth-child(8) {
            font-size: var(--base-font-size) !important;
        }
    </style>
    '''

    # Generate HTML with proper headers and styling
    html_output = f'''    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        {html_content}
    </head>
    <body>
        <table class="analysis-table">
            <thead>
                <tr>
                    <th style="font-size: var(--header-font-size) !important;">Q.NO</th>
                    <th style="font-size: var(--header-font-size) !important;">Attempt %</th>
                    <th style="font-size: var(--header-font-size) !important; width: 100px;">Attempt Status</th>
                    <th style="font-size: var(--header-font-size) !important;">Attainment %</th>
                    <th style="font-size: var(--header-font-size) !important; width: 100px;">Attainment Status</th>
                    <th style="font-size: var(--header-font-size) !important;">Analysis</th>
                    <th style="font-size: var(--header-font-size) !important;">Question Paper Analysis</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    # Add table rows with specific styling for each cell
    for idx in table_data.index:
        html_output += f'''
            <tr>
                <td style="font-size: var(--base-font-size) !important; font-weight: bold;">{table_data.loc[idx, 'Q.NO']}</td>                <td style="font-size: var(--numeric-font-size) !important; font-weight: 500;">{table_data.loc[idx, 'Attempt %']:.2f}</td>
                <td>{table_data.loc[idx, 'Attempt Status']}</td>
                <td style="font-size: var(--numeric-font-size) !important; font-weight: 500;">{table_data.loc[idx, 'Attainment %']:.2f}</td>
                <td>{table_data.loc[idx, 'Attainment Status']}</td>
                <td style="font-size: var(--analysis-font-size) !important; font-weight: 500;">{table_data.loc[idx, 'Analysis']}</td>
                <td style="font-size: var(--question-font-size) !important; text-align: left; padding-left: 15px; line-height: 1.4;">{table_data.loc[idx, 'Question Paper Analysis']}</td>
            </tr>
        '''
    
    html_output += '''
            </tbody>
        </table>
    </body>
    </html>
    '''

    # Save to HTML
    table_path = os.path.join(output_dir, 'colored_analysis_table.html')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(html_output)

    # Save to Excel (emojis will be preserved as is)
    excel_path = os.path.join(output_dir, 'colored_analysis_table.xlsx')
    table_data.to_excel(excel_path, index=False, engine='openpyxl')

    return table_path, excel_path