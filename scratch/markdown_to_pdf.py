import os
import re
import markdown
from xhtml2pdf import pisa

# Pure HTML/CSS representation for the flowchart in Phase 1 (xhtml2pdf safe)
FLOWCHART_HTML = """
<table class="flowchart-table">
  <tr>
    <td class="node-cell"><div class="node phase1">Phase 1: SAB Scheduling</div></td>
    <td class="arrow-cell">➔</td>
    <td class="node-cell"><div class="node phase2">Phase 2: Routing &amp; HDFG</div></td>
    <td class="arrow-cell">
      <div class="arrow-text red-text">➔</div>
      <div class="label red-text">Negative Results</div>
    </td>
    <td class="node-cell"><div class="node phase3">Phase 3: EPEG Core</div></td>
    <td class="arrow-cell">
      <div class="arrow-text green-text">➔</div>
      <div class="label green-text">Positive Contribution</div>
    </td>
    <td class="node-cell"><div class="node phase4">Phase 4: EPEG-SLA</div></td>
    <td class="arrow-cell">➔</td>
    <td class="node-cell"><div class="node phase5">Phase 5: CAPS &amp; EPEG-Slice</div></td>
  </tr>
</table>
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Full Project Report: MoE Serving Co-Design Research</title>
  <style>
    @page {{
      size: A4;
      margin: 1.8cm 1.5cm 1.8cm 1.5cm;
    }}
    
    body {{
      font-family: 'Helvetica', 'Arial', sans-serif;
      color: #334155;
      line-height: 1.5;
      background-color: #ffffff;
      font-size: 10pt;
    }}
    
    h1, h2, h3, h4 {{
      color: #0f172a;
      font-weight: bold;
      margin-top: 1.5em;
      margin-bottom: 0.5em;
    }}
    
    h1 {{
      font-size: 24pt;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 12px;
      margin-top: 0;
      color: #1e293b;
      text-align: center;
    }}
    
    h2 {{
      font-size: 15pt;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 6px;
      color: #0f172a;
      margin-top: 2em;
    }}
    
    h3 {{
      font-size: 12pt;
      color: #334155;
    }}
    
    h4 {{
      font-size: 10pt;
      color: #475569;
    }}
    
    p {{
      margin-bottom: 1em;
    }}
    
    blockquote {{
      background-color: #f8fafc;
      border-left: 4px solid #cbd5e1;
      padding: 10px 14px;
      margin: 15px 0;
      color: #475569;
      font-style: italic;
    }}
    
    /* Alert styling */
    .alert {{
      padding: 10px 14px;
      border-left: 4px solid;
      margin: 15px 0;
    }}
    .alert-note {{
      border-color: #3b82f6;
      background-color: #eff6ff;
      color: #1e40af;
    }}
    .alert-tip {{
      border-color: #10b981;
      background-color: #ecfdf5;
      color: #065f46;
    }}
    .alert-important {{
      border-color: #8b5cf6;
      background-color: #f5f3ff;
      color: #5b21b6;
    }}
    .alert-warning {{
      border-color: #f59e0b;
      background-color: #fffbeb;
      color: #92400e;
    }}
    .alert-caution {{
      border-color: #ef4444;
      background-color: #fef2f2;
      color: #991b1b;
    }}
    .alert p {{
      margin: 0;
    }}
    
    /* Table styling */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 9pt;
    }}
    
    th, td {{
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
    }}
    
    th {{
      background-color: #1e293b;
      color: #ffffff;
      font-weight: bold;
    }}
    
    tr:nth-child(even) {{
      background-color: #f8fafc;
    }}
    
    /* Code styling */
    code {{
      font-family: 'Courier', monospace;
      background-color: #f1f5f9;
      padding: 2px 4px;
      font-size: 8.5pt;
      color: #0f172a;
    }}
    
    pre {{
      background-color: #0f172a;
      padding: 12px;
      margin: 15px 0;
    }}
    
    pre code {{
      background-color: transparent;
      color: #f8fafc;
      padding: 0;
      font-size: 8.5pt;
    }}
    
    /* Flowchart Table */
    table.flowchart-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 25px 0;
    }}
    table.flowchart-table td {{
      padding: 0;
      border: none;
      background-color: transparent;
      vertical-align: middle;
      text-align: center;
    }}
    table.flowchart-table td.node-cell {{
      width: 16%;
    }}
    table.flowchart-table td.arrow-cell {{
      width: 5%;
      color: #64748b;
    }}
    .node {{
      color: #ffffff;
      font-weight: bold;
      font-size: 8pt;
      padding: 10px 4px;
      text-align: center;
    }}
    .phase1 {{ background-color: #94a3b8; }}
    .phase2 {{ background-color: #f87171; }}
    .phase3 {{ background-color: #34d399; }}
    .phase4 {{ background-color: #60a5fa; }}
    .phase5 {{ background-color: #a78bfa; }}
    .arrow-text {{
      font-size: 12pt;
      font-weight: bold;
      line-height: 1;
    }}
    .label {{
      font-size: 6.5pt;
      font-weight: bold;
      margin-top: 2px;
    }}
    .red-text {{ color: #ef4444; }}
    .green-text {{ color: #10b981; }}
    
    /* Image Grid Table for Carousel */
    table.image-grid-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }}
    table.image-grid-table td {{
      width: 50%;
      padding: 10px;
      text-align: center;
      vertical-align: top;
      border: 1px solid #e2e8f0;
      background-color: #f8fafc;
    }}
    img.grid-img {{
      width: 250px;
      height: 160px;
    }}
    .grid-caption {{
      font-size: 8pt;
      font-weight: bold;
      color: #475569;
      margin-top: 5px;
    }}
    
    /* Standard Markdown Images */
    img {{
      width: 320px;
      height: 200px;
      display: block;
      margin: 15px auto;
    }}
    
    hr {{
      border: 0;
      border-top: 1px solid #e2e8f0;
      margin: 25px 0;
    }}
    
    .page-break {{
      page-break-before: always;
    }}
  </style>
</head>
<body>
  {content}
</body>
</html>
"""

def add_page_breaks(text):
    headings_to_break = [
        "## Phase 1:",
        "## Phase 2:",
        "## Phase 3:",
        "## Phase 4:",
        "## Phase 5:",
        "## Project Statistics"
    ]
    for heading in headings_to_break:
        text = text.replace(heading, f'<div class="page-break"></div>\n\n{heading}')
    return text

def process_carousel(text):
    pattern = r'````carousel\s*(.*?)\s*````'
    
    def replacer(match):
        content = match.group(1)
        slides = content.split('<!-- slide -->')
        
        parsed_slides = []
        for slide in slides:
            slide = slide.strip()
            if not slide:
                continue
            img_match = re.match(r'!\[(.*?)\]\((.*?)\)', slide)
            if img_match:
                caption = img_match.group(1)
                path = img_match.group(2)
                parsed_slides.append((caption, path))
                
        # Now construct a 2-column table
        html_out = ['<table class="image-grid-table">']
        for i in range(0, len(parsed_slides), 2):
            html_out.append('  <tr>')
            # Column 1
            cap1, path1 = parsed_slides[i]
            html_out.append(f'''    <td>
      <img src="{path1}" class="grid-img" alt="{cap1}">
      <div class="grid-caption">{cap1}</div>
    </td>''')
            # Column 2
            if i + 1 < len(parsed_slides):
                cap2, path2 = parsed_slides[i+1]
                html_out.append(f'''    <td>
      <img src="{path2}" class="grid-img" alt="{cap2}">
      <div class="grid-caption">{cap2}</div>
    </td>''')
            else:
                html_out.append('    <td></td>')
            html_out.append('  </tr>')
        html_out.append('</table>')
        return '\n'.join(html_out)

    return re.sub(pattern, replacer, text, flags=re.DOTALL)

def replace_math_and_chars(text):
    text = text.replace(r'$\lambda_c$', 'λ<sub>c</sub>')
    text = text.replace(r'$k=2,4,8$', 'k = 2, 4, 8')
    text = text.replace(r'$k$', 'k')
    return text

def main():
    md_path = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/full_project_report.md"
    html_path = "/home/palakm/MoEServingSim/scratch/full_project_report.html"
    pdf_path = "/home/palakm/MoEServingSim/full_project_report.pdf"
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # 1. Preprocess math characters
    md_content = replace_math_and_chars(md_content)
    
    # 2. Add page breaks before sections
    md_content = add_page_breaks(md_content)
    
    # 3. Preprocess carousels into grids
    md_content = process_carousel(md_content)
    
    # 4. Convert markdown to HTML
    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    # 5. Process blockquotes into styled alerts
    alert_pattern = r'<blockquote>\s*<p>\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(?:\n|<br\s*/?>)?'
    html_body = re.sub(
        alert_pattern,
        lambda m: f'<blockquote class="alert alert-{m.group(1).lower()}"><p>',
        html_body,
        flags=re.IGNORECASE
    )
    
    # 6. Replace the mermaid pre/code block with our flowchart HTML table
    mermaid_pattern = r'<pre><code class="language-mermaid">.*?</code></pre>'
    html_body = re.sub(
        mermaid_pattern,
        FLOWCHART_HTML,
        html_body,
        flags=re.DOTALL
    )
    
    # 7. Render full HTML string
    full_html = HTML_TEMPLATE.format(content=html_body)
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Generated HTML: {html_path}")
    
    # 8. Convert HTML to PDF using xhtml2pdf
    with open(html_path, "r", encoding="utf-8") as html_file:
        html_content = html_file.read()
        
    with open(pdf_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(
            html_content,
            dest=result_file
        )
        
    if not pisa_status.err:
        print(f"PDF generated successfully at: {pdf_path}")
    else:
        print(f"Error generating PDF: {pisa_status.err}")

if __name__ == "__main__":
    main()
