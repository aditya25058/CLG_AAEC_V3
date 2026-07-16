#!/usr/bin/env python3
import os
import re
import markdown
import shutil
from xhtml2pdf import pisa

MD_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/neuron_sparsity_proof_report.md"
WORKSPACE_PDF_PATH = "/home/palakm/MoEServingSim/neuron_sparsity_proof_report.pdf"
BRAIN_PDF_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/neuron_sparsity_proof_report.pdf"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AAEC v3: MoE Neuron Column Sparsity and Quality Proof Report</title>
  <style>
    @page {
      size: A4;
      margin: 1.8cm 1.5cm 1.8cm 1.5cm;
    }
    
    body {
      font-family: 'Georgia', 'Times New Roman', Times, serif;
      color: #334155;
      line-height: 1.6;
      background-color: #ffffff;
      font-size: 10pt;
    }
    
    h1, h2, h3, h4 {
      font-family: 'Helvetica', 'Arial', sans-serif;
      color: #0f172a;
      font-weight: bold;
      margin-top: 1.5em;
      margin-bottom: 0.5em;
    }
    
    h1 {
      font-size: 18pt;
      border-bottom: 2px solid #cbd5e1;
      padding-bottom: 12px;
      margin-top: 0;
      color: #1e293b;
      text-align: center;
    }
    
    h2 {
      font-size: 13pt;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 6px;
      color: #0f172a;
      margin-top: 1.8em;
    }
    
    h3 {
      font-size: 11pt;
      color: #334155;
      margin-top: 1.2em;
    }
    
    p {
      margin-bottom: 1em;
      text-align: justify;
    }
    
    blockquote {
      background-color: #f8fafc;
      border-left: 4px solid #cbd5e1;
      padding: 8px 12px;
      margin: 15px 0;
      color: #475569;
      font-style: italic;
    }
    
    /* Lists */
    ul, ol {
      margin-bottom: 1em;
      padding-left: 20px;
    }
    
    li {
      margin-bottom: 0.4em;
    }
    
    /* Code block styling */
    pre {
      background-color: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 10px;
      margin: 15px 0;
      font-family: 'Courier New', Courier, monospace;
      font-size: 8.5pt;
    }
    
    code {
      font-family: 'Courier New', Courier, monospace;
      font-size: 8.5pt;
      background-color: #f1f5f9;
      padding: 2px 4px;
      border-radius: 3px;
    }
    
    /* Table styling */
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 15px 0;
      font-size: 7.5pt;
      page-break-inside: avoid;
    }
    
    th, td {
      border: 1px solid #cbd5e1;
      padding: 6px 8px;
      text-align: left;
    }
    
    th {
      background-color: #f1f5f9;
      font-weight: bold;
      color: #0f172a;
    }
    
    tr {
      page-break-inside: avoid;
    }
    
    tr:nth-child(even) {
      background-color: #f8fafc;
    }
    
    img {
      width: 4.2in;
      height: 2.8in;
      display: block;
      margin: 15px auto;
      border: 1px solid #cbd5e1;
    }
    
    .page-break {
      page-break-before: always;
    }
    
    .math-block {
      font-family: 'Georgia', 'Times New Roman', serif;
      font-size: 11pt;
      text-align: center;
      margin: 15px 0;
      padding: 8px 12px;
      background-color: #f8fafc;
      border-left: 3px solid #10b981;
      color: #0f172a;
    }
  </style>
</head>
<body>
  {content}
</body>
</html>
"""

def add_page_breaks(text):
    # Add page breaks before sections 2, 3, 4, 5
    pattern = r'(?=\n##\s+[2345]\s+)'
    return re.sub(pattern, '\n<div class="page-break"></div>\n', text)

def replace_math_formulas(text):
    # Display Equations
    text = text.replace(r'$$\mathbf{y} = \mathbf{W}_{\text{down}} \left( \text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}}^{\top}) \odot (\mathbf{x}\mathbf{W}_{\text{up}}^{\top}) \right)$$',
                        '<div class="math-block"><b>y</b> = <b>W</b><sub>down</sub> · [ SiLU(<b>x</b> · <b>W</b><sub>gate</sub><sup>T</sup>) &odot; (<b>x</b> · <b>W</b><sub>up</sub><sup>T</sup>) ]</div>')
    text = text.replace(r'$$\mathbf{a} = \text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}}^{\top}) \odot (\mathbf{x}\mathbf{W}_{\text{up}}^{\top})$$',
                        '<div class="math-block"><b>a</b> = SiLU(<b>x</b> · <b>W</b><sub>gate</sub><sup>T</sup>) &odot; (<b>x</b> · <b>W</b><sub>up</sub><sup>T</sup>)</div>')
    text = text.replace(r'$$\|\mathbf{y} - \mathbf{y}_{\text{sliced}}\|_2 \le \|\mathbf{W}_{\text{down}}\|_F \cdot \|\mathbf{a}_{\text{dropped}}\|_2$$',
                        '<div class="math-block">||<b>y</b> - <b>y</b><sub>sliced</sub>||<sub>2</sub> &le; ||<b>W</b><sub>down</sub>||<sub>F</sub> · ||<b>a</b><sub>dropped</sub>||<sub>2</sub></div>')
    
    # Inline formulas
    text = text.replace(r'$50\%$', '<b>50%</b>')
    text = text.replace(r'$75\text{ MB}$', '<b>75 MB</b>')
    text = text.replace(r'$6.4\text{ MB}$', '<b>6.4 MB</b>')
    text = text.replace(r'$208$', '<b>208</b>')
    text = text.replace(r'$$115.5\text{ columns} \approx 3.5\text{ MB}$$', '<b>115.5 columns &asymp; 3.5 MB</b>')
    text = text.replace(r'$115.5\text{ columns} \approx 3.5\text{ MB}$', '<b>115.5 columns &asymp; 3.5 MB</b>')
    
    # Subscript styles
    text = text.replace(r'$\mathbf{W}_g$', '<b>W<sub>g</sub></b>')
    text = text.replace(r'$\mathbf{h}_{pre}$', '<b>h<sub>pre</sub></b>')
    text = text.replace(r'$\mathbf{s}_{pre}$', '<b>s<sub>pre</sub></b>')
    text = text.replace(r'$\mathbf{W}_g^\top$', '<b>W<sub>g</sub><sup>T</sup></b>')
    text = text.replace(r'$\mathbf{s}_{post}$', '<b>s<sub>post</sub></b>')
    text = text.replace(r'$\mathbf{h}_{post}$', '<b>h<sub>post</sub></b>')
    text = text.replace(r'$\mathbf{y}$', '<b>y</b>')
    text = text.replace(r'$\mathbf{W}_{\text{down}}$', '<b>W<sub>down</sub></b>')
    text = text.replace(r'$\mathbf{x}\mathbf{W}_{\text{gate}}^{\top}$', '<b>x · W<sub>gate</sub><sup>T</sup></b>')
    text = text.replace(r'$\mathbf{x}\mathbf{W}_{\text{up}}^{\top}$', '<b>x · W<sub>up</sub><sup>T</sup></b>')
    text = text.replace(r'$\mathbf{a}$', '<b>a</b>')
    text = text.replace(r'$\|\mathbf{y} - \mathbf{y}_{\text{sliced}}\|_2$', '<b>||y - y<sub>sliced</sub>||<sub>2</sub></b>')
    text = text.replace(r'$\|\mathbf{W}_{\text{down}}\|_F$', '<b>||W<sub>down</sub>||<sub>F</sub></b>')
    text = text.replace(r'$\|\mathbf{a}_{\text{dropped}}\|_2$', '<b>||a<sub>dropped</sub>||<sub>2</sub></b>')
    
    # Other symbols
    text = text.replace(r'$\eta$', '<b>&eta;</b>')
    text = text.replace(r'$\sum a_i^2$', '<b>&Sigma; a<sub>i</sub><sup>2</sup></b>')
    text = text.replace(r'$90\%$', '<b>90%</b>')
    text = text.replace(r'$98.86\%$', '<b>98.86%</b>')
    text = text.replace(r'$99.53\%$', '<b>99.53%</b>')
    text = text.replace(r'$50\%$', '<b>50%</b>')
    text = text.replace(r'$90.16\%$', '<b>90.16%</b>')
    text = text.replace(r'$84.84\%$', '<b>84.84%</b>')
    
    # Cleanup trailing LaTeX structures
    text = text.replace(r'\mathbf', '<b>W</b>')
    
    return text

def main():
    print(f"Reading markdown: {MD_PATH}")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    print("Preprocessing content...")
    md_processed = replace_math_formulas(md_content)
    md_processed = add_page_breaks(md_processed)
    
    print("Converting Markdown to HTML...")
    html_body = markdown.markdown(md_processed, extensions=['tables', 'fenced_code'])
    full_html = HTML_TEMPLATE.replace("{content}", html_body)
    
    # Save a temporary HTML copy for debugging if needed
    debug_html_path = "/home/palakm/MoEServingSim/scratch/debug_sparsity_report.html"
    with open(debug_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Compiling PDF to {BRAIN_PDF_PATH}...")
    with open(BRAIN_PDF_PATH, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(full_html, dest=result_file)
        
    if not pisa_status.err:
        print("PDF compiled successfully!")
        shutil.copy(BRAIN_PDF_PATH, WORKSPACE_PDF_PATH)
        print(f"Copied to workspace root: {WORKSPACE_PDF_PATH}")
    else:
        print(f"Error compiling PDF: {pisa_status.err}")

if __name__ == "__main__":
    main()
