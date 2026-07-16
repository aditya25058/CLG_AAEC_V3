#!/usr/bin/env python3
"""Convert the real hardware MoE activation study markdown file into a beautifully styled PDF."""
import os
import re
import markdown
import shutil
from xhtml2pdf import pisa

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Real-World MoE Activation Study: Qwen3-30B-A3B on NVIDIA H100 GPUs</title>
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
    
    h4 {
      font-size: 10pt;
      color: #475569;
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
      overflow: hidden;
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
  </style>
</head>
<body>
  {content}
</body>
</html>
"""

def add_page_breaks(text):
    # Add page break before major section headers (## 1. , ## 2. , etc.)
    pattern = r'(?=\n##\s+[1234567]\s+)'
    return re.sub(pattern, '\n<div class="page-break"></div>\n', text)

def replace_math_formulas(text):
    # Display Equations
    text = text.replace(r'$$\text{SwiGLU}(x) = \text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})$$',
                        '<pre>SwiGLU(x) = SiLU(x * W_gate) * (x * W_up)</pre>')
    text = text.replace(r'$$f^{(e)}_i = \frac{\text{Number of tokens activating neuron } i \text{ in expert } e}{\text{Total number of tokens routed to expert } e}$$',
                        '<pre>f^(e)_i = (Number of tokens activating neuron i in expert e) / (Total tokens routed to expert e)</pre>')
    
    # Inline math symbols
    # Section Method & Intro
    text = text.replace(r'$D_{FFN} = 768$', '<b>D<sub>FFN</sub> = 768</b>')
    text = text.replace(r'$W(n) = \left|\bigcup_{i=1}^{n} \mathcal{A}^{(e)}_i\right|$', '<b>W(n) = | &cup;<sub>i=1..n</sub> A<sup>(e)</sup><sub>i</sub> |</b>')
    text = text.replace(r'$W(n)$', '<b>W(n)</b>')
    text = text.replace(r'$n$', '<b>n</b>')
    text = text.replace(r'$n = 1$', '<b>n = 1</b>')
    text = text.replace(r'$n = 2$', '<b>n = 2</b>')
    text = text.replace(r'$n = 4$', '<b>n = 4</b>')
    text = text.replace(r'$n = 8$', '<b>n = 8</b>')
    text = text.replace(r'$n = 16$', '<b>n = 16</b>')
    text = text.replace(r'$n = 32$', '<b>n = 32</b>')
    
    # Section Jaccard & Lifetimes
    text = text.replace(r'$t_0$', '<b>t<sub>0</sub></b>')
    text = text.replace(r'$t_0 + d$', '<b>t<sub>0</sub> + d</b>')
    text = text.replace(r'$d$', '<b>d</b>')
    text = text.replace(r'$f^{(e)}_i$', '<b>f<sup>(e)</sup><sub>i</sub></b>')
    text = text.replace(r'$f^{(e)}_i > 0.80$', '<b>f<sup>(e)</sup><sub>i</sub> &gt; 0.80</b>')
    text = text.replace(r'$0.10 < f^{(e)}_i \le 0.80$', '<b>0.10 &lt; f<sup>(e)</sup><sub>i</sub> &le; 0.80</b>')
    text = text.replace(r'$f^{(e)}_i \le 0.10$', '<b>f<sup>(e)</sup><sub>i</sub> &le; 0.10</b>')
    
    # Section Entropy & Zipf
    text = text.replace(r'$H_l = -\sum_e p_e \log p_e$', '<b>H<sub>l</sub> = - &Sigma;<sub>e</sub> p<sub>e</sub> log(p<sub>e</sub>)</b>')
    text = text.replace(r'$\alpha$', '<b>&alpha;</b>')
    text = text.replace(r'Expert $A$ to Expert $B$', '<b>Expert A to Expert B</b>')
    text = text.replace(r'Expert $B$', '<b>Expert B</b>')
    text = text.replace(r'Expert $A$', '<b>Expert A</b>')
    
    # Section Model Quality
    text = text.replace(r'$\eta \in \{30\%, 40\%, 50\%, 60\%, 70\%, 80\%, 90\%\}$', '<b>&eta; &isin; {30%, 40%, 50%, 60%, 70%, 80%, 90%}</b>')
    text = text.replace(r'$\eta$', '<b>&eta;</b>')
    text = text.replace(r'$\eta = 90\%$', '<b>&eta; = 90%</b>')
    text = text.replace(r'$\eta = 80\%$', '<b>&eta; = 80%</b>')
    text = text.replace(r'$\eta = 70\%$', '<b>&eta; = 70%</b>')
    text = text.replace(r'$\eta = 60\%$', '<b>&eta; = 60%</b>')
    text = text.replace(r'$\eta = 50\%$', '<b>&eta; = 50%</b>')
    text = text.replace(r'$\eta = 40\%$', '<b>&eta; = 40%</b>')
    text = text.replace(r'$\eta = 30\%$', '<b>&eta; = 30%</b>')
    text = text.replace(r'$|a| > 0.05$', '<b>|a| &gt; 0.05</b>')
    
    # Section Cache sweep
    text = text.replace(r'$C \in \{32, 64, 96, 128, 160\}$', '<b>C &isin; {32, 64, 96, 128, 160}</b>')
    text = text.replace(r'$1 - (\text{Bytes Transferred} / \text{Baseline full expert bytes})$', '<b>1 - (Bytes Transferred / Baseline full expert bytes)</b>')
    text = text.replace(r'$47.45\text{ us}$', '<b>47.45 us</b>')
    text = text.replace(r'$300.71\text{ us}$', '<b>300.71 us</b>')
    
    # Section Packing
    text = text.replace(r'$W_{gate}[:, i]$', '<b>W_gate[:, i]</b>')
    text = text.replace(r'$W_{up}[:, i]$', '<b>W_up[:, i]</b>')
    text = text.replace(r'$W_{down}[i, :]$', '<b>W_down[i, :]</b>')
    text = text.replace(r'$24.576\text{ KB}$', '<b>24.576 KB</b>')
    text = text.replace(r'$18.874\text{ MB}$', '<b>18.874 MB</b>')
    
    # Discussion
    text = text.replace(r'$\tau \approx 10.3$', '<b>&tau; &asymp; 10.3</b>')
    text = text.replace(r'$9.44\text{ MB}$', '<b>9.44 MB</b>')
    text = text.replace(r'$24.58\text{ KB}$', '<b>24.58 KB</b>')
    text = text.replace(r'$944\text{ KB}$', '<b>944 KB</b>')
    text = text.replace(r'$W^{(i)}_t = (1 - \lambda) \cdot W^{(i)}_{t-1} + \lambda \cdot \mathbb{I}(a^{(i)}_t > 0)$', '<b>W^(i)_t = (1 - &lambda;) * W^(i)_{t-1} + &lambda; * I(a^(i)_t &gt; 0)</b>')
    text = text.replace(r'$\lambda$', '<b>&lambda;</b>')
    text = text.replace(r'$\theta_{\text{evict}}$', '<b>&theta;_evict</b>')
    text = text.replace(r'$12.288\text{ KB}$', '<b>12.288 KB</b>')
    text = text.replace(r'$18.874\text{ MB}$', '<b>18.874 MB</b>')
    
    # Additional cleanups
    text = text.replace(r'\mu', '&mu;')
    text = text.replace(r'$350.8\text{ us}$', '<b>350.8 us</b>')
    text = text.replace(r'$32.8\text{ us}$', '<b>32.8 us</b>')
    text = text.replace(r'$16,490.3\text{ us}$', '<b>16,490.3 us</b>')
    text = text.replace(r'$14,579.4\text{ us}$', '<b>14,579.4 us</b>')
    
    return text

def main():
    md_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/real_hardware_moe_activation_study.md"
    html_path = "/home/palakm/MoEServingSim/scratch/real_hardware_moe_activation_study.html"
    
    print(f"Reading study markdown: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # ==========================================
    # PDF 1: Full Activation Study Report
    # ==========================================
    pdf1_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/real_hardware_moe_activation_study.pdf"
    workspace_pdf1_path = "/home/palakm/MoEServingSim/real_hardware_moe_activation_study.pdf"
    
    print("Preprocessing PDF 1 (Full Report)...")
    md_content_p1 = replace_math_formulas(md_content)
    md_content_p1 = add_page_breaks(md_content_p1)
    md_content_p1 = re.sub(
        r'```mermaid[\s\S]*?```',
        '<div style="text-align: center; border: 1px solid #cbd5e1; padding: 10px; margin: 15px 0; background-color: #f8fafc; font-family: sans-serif; font-size: 9pt;">[ MoE serve architecture flow diagram: H100 GPU Trace Characterization -> Quality Validation & Systems Cache Design ]</div>',
        md_content_p1
    )
    
    html_body_p1 = markdown.markdown(md_content_p1, extensions=['tables', 'fenced_code'])
    full_html_p1 = HTML_TEMPLATE.replace("{content}", html_body_p1)
    
    print("Creating Full PDF...")
    with open(pdf1_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(full_html_p1, dest=result_file)
        
    if not pisa_status.err:
        print(f"Full PDF generated successfully: {pdf1_path}")
        shutil.copy(pdf1_path, workspace_pdf1_path)
        print(f"Copied to workspace root: {workspace_pdf1_path}")
    else:
        print(f"Error compiling Full PDF: {pisa_status.err}")
        
    # ==========================================
    # PDF 2: Systems implications & Telemetry Report (Sec 6 till End)
    # ==========================================
    pdf2_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/real_hardware_systems_telemetry_report.pdf"
    workspace_pdf2_path = "/home/palakm/MoEServingSim/real_hardware_systems_telemetry_report.pdf"
    md2_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/real_hardware_systems_telemetry_report.md"
    
    print(f"\nReading Systems Telemetry study markdown: {md2_path}")
    with open(md2_path, "r", encoding="utf-8") as f:
        md_content_p2 = f.read()
        
    md_content_p2 = replace_math_formulas(md_content_p2)
    # Add page breaks before major subsections in Systems PDF
    md_content_p2 = re.sub(r'(?=\n###\s+6\.[23456]\s+)', '\n<div class="page-break"></div>\n', md_content_p2)
    
    html_body_p2 = markdown.markdown(md_content_p2, extensions=['tables', 'fenced_code'])
    full_html_p2 = HTML_TEMPLATE.replace("{content}", html_body_p2)
    
    print("Creating Systems PDF...")
    with open(pdf2_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(full_html_p2, dest=result_file)
        
    if not pisa_status.err:
        print(f"Systems PDF generated successfully: {pdf2_path}")
        shutil.copy(pdf2_path, workspace_pdf2_path)
        print(f"Copied to workspace root: {workspace_pdf2_path}")
    else:
        print(f"Error compiling Systems PDF: {pisa_status.err}")

if __name__ == "__main__":
    main()
