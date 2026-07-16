#!/usr/bin/env python3
"""Convert the INet4AI submission draft markdown file into a PDF."""
import os
import re
import markdown
from xhtml2pdf import pisa

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Networking-Aware Communication Co-Design for Distributed MoE Serving</title>
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
      font-size: 20pt;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 12px;
      margin-top: 0;
      color: #1e293b;
      text-align: center;
    }}
    
    h2 {{
      font-size: 14pt;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 6px;
      color: #0f172a;
      margin-top: 2em;
    }}
    
    h3 {{
      font-size: 11pt;
      color: #334155;
    }}
    
    h4 {{
      font-size: 10pt;
      color: #475569;
    }}
    
    p {{
      margin-bottom: 1em;
      text-align: justify;
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
      font-style: normal;
    }}
    .alert-note {{
      background-color: #f0f9ff;
      border-left-color: #0ea5e9;
      color: #0369a1;
    }}
    .alert-tip {{
      background-color: #f0fdf4;
      border-left-color: #22c55e;
      color: #15803d;
    }}
    .alert-important {{
      background-color: #fef2f2;
      border-left-color: #ef4444;
      color: #b91c1c;
    }}
    .alert-warning {{
      background-color: #fffbeb;
      border-left-color: #f59e0b;
      color: #b45309;
    }}
    .alert-caution {{
      background-color: #faf5ff;
      border-left-color: #a855f7;
      color: #7e22ce;
    }}
    
    /* Lists */
    ul, ol {{
      margin-bottom: 1em;
      padding-left: 20px;
    }}
    
    li {{
      margin-bottom: 0.5em;
    }}
    
    /* Code block styling */
    pre {{
      background-color: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 10px;
      margin: 15px 0;
      font-family: 'Courier New', Courier, monospace;
      font-size: 9pt;
      overflow: hidden;
    }}
    
    code {{
      font-family: 'Courier New', Courier, monospace;
      font-size: 9pt;
      background-color: #f1f5f9;
      padding: 2px 4px;
      border-radius: 3px;
    }}
    
    /* Table styling */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 15px 0;
      font-size: 7.5pt;
      page-break-inside: avoid;
    }}
    
    th, td {{
      border: 1px solid #e2e8f0;
      padding: 4px 6px;
      text-align: left;
    }}
    
    th {{
      background-color: #f1f5f9;
      font-weight: bold;
      color: #0f172a;
    }}
    
    tr {{
      page-break-inside: avoid;
    }}
    
    tr:nth-child(even) {{
      background-color: #f8fafc;
    }}
    
    img {{
      width: 4.5in;
      height: 3.2in;
      display: block;
      margin: 15px auto;
      border: 1px solid #e2e8f0;
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
    # Add page break before major section headers (e.g. ## 2. , ## 3. )
    pattern = r'(?=\n##\s+[2345]\.\s+)'
    return re.sub(pattern, '\n<div class="page-break"></div>\n', text)

def replace_math_and_chars(text):
    # Replace LaTeX math blocks with cleaner text approximations for xhtml2pdf
    text = text.replace(r'$$\min_{A} \sum_{i} \sum_{j} a_{i,j} \cdot \Big[ \underbrace{Q_j(L_j)}_{\text{Compute Queue Delay}} + \underbrace{d(i, j) \cdot \frac{\text{Bytes}(x_i)}{\text{BW}(i, j)}}_{\text{Network Transport Delay}} \Big]$$', 
                        '<pre>Routing Objective: min Sum_i Sum_j a_{i,j} * [ Compute_Queue_Delay(j) + Network_Transport_Delay(i, j) ]</pre>')
    text = text.replace(r'$$Q(t+1) = \max\left(0, Q(t) + A(t) - S(t)\right)$$',
                        '<pre>Queue Equation: Q(t+1) = max(0, Q(t) + A(t) - S(t))</pre>')
    text = text.replace(r'$$D(t) = \max\left(0, Q(t) + A(t) - S(t) - \theta_{\text{buffer}}\right)$$',
                        '<pre>Packet Drops: D(t) = max(0, Q(t) + A(t) - S(t) - theta_buffer)</pre>')
    text = text.replace(r'$$T_{\text{stall}} = N_{\text{drops}} \times \text{RTO}$$',
                        '<pre>Timeout Stall Time: T_stall = N_drops * RTO</pre>')
    text = text.replace(r'$$I = \frac{\text{Concurrent Senders Targeting a Single Port}}{\text{Available Switch Receiver Ports}}$$',
                        '<pre>Incast Factor (I) = Concurrent Senders Targeting a Single Port / Available Switch Receiver Ports</pre>')
    text = text.replace(r'$$\eta = \frac{T_{\text{serialize}} + T_{\text{prop}}}{T_{\text{total}}} \times 100\%$$',
                        '<pre>Sustained BW Efficiency (eta) = (T_serialize + T_prop) / T_total * 100%</pre>')
    text = text.replace(r'$$T_{\text{congestion}} = \sum \Delta t \quad \text{s.t.} \quad Q_t \ge \theta_{\text{buffer}}$$',
                        '<pre>Congestion Time (T_congestion) = Sum Delta_t s.t. Q_t >= theta_buffer</pre>')
    text = text.replace(r'$$\omega = \frac{T_{\text{comm}} - T_{\text{exposed}}}{T_{\text{comm}}} \times 100\%$$',
                        '<pre>Overlap Ratio (omega) = (T_comm - T_exposed) / T_comm * 100%</pre>')
    text = text.replace(r'$x_t$', 'x_t')
    text = text.replace(r'$E_j$', 'E_j')
    text = text.replace(r'$a_{i,j} \in \{0, 1\}$', 'a_{i,j} in {0, 1}')
    text = text.replace(r'$Q_j(L_j)$', 'Q_j(L_j)')
    text = text.replace(r'$d(i, j)$', 'd(i, j)')
    text = text.replace(r'$\text{Bytes}(x_i)$', 'Bytes(x_i)')
    text = text.replace(r'$\text{BW}(i, j)$', 'BW(i, j)')
    text = text.replace(r'$k$', 'k')
    text = text.replace(r'$P$', 'P')
    text = text.replace(r'$T / P$', 'T / P')
    text = text.replace(r'$10\%$', '10%')
    text = text.replace(r'$15\%$', '15%')
    text = text.replace(r'$30\%$', '30%')
    text = text.replace(r'$14\%$', '14%')
    text = text.replace(r'$53\%$', '53%')
    text = text.replace(r'$1.87\times$', '1.87x')
    text = text.replace(r'$1.64\times$', '1.64x')
    text = text.replace(r'$18.01\times$', '18.01x')
    text = text.replace(r'$9.8\times$', '9.8x')
    text = text.replace(r'$32 \times 31 = 992$', '32 * 31 = 992')
    text = text.replace(r'$31 \times 20\text{ \mu s} = 620\text{ \mu s}$', '31 * 20 us = 620 us')
    text = text.replace(r'$O(1/P)$', 'O(1/P)')
    
    # Section 3.4 & 3.5 symbols
    text = text.replace(r'$Q(t)$', 'Q(t)')
    text = text.replace(r'$Q(t+1)$', 'Q(t+1)')
    text = text.replace(r'$A(t)$', 'A(t)')
    text = text.replace(r'$S(t)$', 'S(t)')
    text = text.replace(r'$C$', 'C')
    text = text.replace(r'$D(t)$', 'D(t)')
    text = text.replace(r'$N_{\text{drops}}$', 'N_drops')
    text = text.replace(r'$T_{\text{stall}}$', 'T_stall')
    text = text.replace(r'$I$', 'I')
    text = text.replace(r'$I = 31$', 'I = 31')
    text = text.replace(r'$I = 1$', 'I = 1')
    text = text.replace(r'$\eta$', 'eta')
    text = text.replace(r'$T_{\text{serialize}}$', 'T_serialize')
    text = text.replace(r'$T_{\text{prop}}$', 'T_prop')
    text = text.replace(r'$T_{\text{total}}$', 'T_total')
    text = text.replace(r'$T_{\text{congestion}}$', 'T_congestion')
    text = text.replace(r'$Q_t$', 'Q_t')
    text = text.replace(r'$\theta_{\text{buffer}}$', 'theta_buffer')
    text = text.replace(r'$\omega$', 'omega')
    text = text.replace(r'$T_{\text{comm}}$', 'T_comm')
    text = text.replace(r'$T_{\text{exposed}}$', 'T_exposed')
    
    # Section 3.8 local upgrades & crossover symbols
    text = text.replace(r'$$\text{Skew} = 0.70$$', 'Skew = 0.70')
    text = text.replace(r'$$\text{CoV} = 0.0000$$', 'CoV = 0.0000')
    text = text.replace(r'$$\text{Max/Mean} = 1.0000$$', 'Max/Mean = 1.0000')
    text = text.replace(r'$$T_{\text{activation}} = N_{\text{redirect}} \times 28.0 \ \mu\text{s}$$',
                        '<pre>T_activation = N_redirect * 28.0 us</pre>')
    text = text.replace(r'$$T_{\text{weight}} = 222.22 \ \mu\text{s}$$',
                        '<pre>T_weight = 222.22 us</pre>')
    text = text.replace(r'$$N_{\text{redirect}} \times 28.0 \ \mu\text{s} = 222.22 \ \mu\text{s} \implies N_{\text{redirect}} = 7.93 \text{ tokens}$$',
                        '<pre>N_redirect * 28.0 us = 222.22 us => N_redirect = 7.93 tokens</pre>')
    text = text.replace(r'$$N_{\text{redirect}} < 8$$', 'N_redirect < 8')
    text = text.replace(r'$$N_{\text{redirect}} \ge 8$$', 'N_redirect >= 8')
    text = text.replace(r'$$99.2\%$$', '99.2%')
    text = text.replace(r'$$152$$', '152')
    text = text.replace(r'$$4$$', '4')
    text = text.replace(r'$$112 \ \mu\text{s}$$', '112 us')
    text = text.replace(r'$$222.22 \ \mu\text{s}$$', '222.22 us')
    text = text.replace(r'$$134.02 \text{ ms}$$', '134.02 ms')
    text = text.replace(r'$$27.79 \text{ ms}$$', '27.79 ms')
    text = text.replace(r'$$70$$', '70')
    text = text.replace(r'$$0.8\%$$', '0.8%')
    text = text.replace(r'$$1.96 \text{ ms}$$', '1.96 ms')
    text = text.replace(r'$$1.737 \text{ ms}$$', '1.737 ms')
    text = text.replace(r'$$106.23 \text{ ms}$$', '106.23 ms')
    text = text.replace(r'$$0.62\text{ s}$$', '0.62 s')
    text = text.replace(r'$\text{Skew} = 0.70$', 'Skew = 0.70')
    text = text.replace(r'$\text{CoV} = 0.0000$', 'CoV = 0.0000')
    text = text.replace(r'$\text{Max/Mean} = 1.0000$', 'Max/Mean = 1.0000')
    text = text.replace(r'$N_{\text{redirect}} < 8$', 'N_redirect < 8')
    text = text.replace(r'$N_{\text{redirect}} \ge 8$', 'N_redirect >= 8')
    text = text.replace(r'$99.2\%$', '99.2%')
    text = text.replace(r'$152$', '152')
    text = text.replace(r'$4$', '4')
    text = text.replace(r'$112 \ \mu\text{s}$', '112 us')
    text = text.replace(r'$222.22 \ \mu\text{s}$', '222.22 us')
    text = text.replace(r'$134.02 \text{ ms}$', '134.02 ms')
    text = text.replace(r'$27.79 \text{ ms}$', '27.79 ms')
    text = text.replace(r'$70$', '70')
    text = text.replace(r'$0.8\%$', '0.8%')
    text = text.replace(r'$1.96 \text{ ms}$', '1.96 ms')
    text = text.replace(r'$1.737 \text{ ms}$', '1.737 ms')
    text = text.replace(r'$106.23 \text{ ms}$', '106.23 ms')
    text = text.replace(r'$0.62\text{ s}$', '0.62 s')
    text = text.replace(r'$50\text{ ns}$', '50 ns')
    text = text.replace(r'$200\text{ MB}$', '200 MB')
    text = text.replace(r'$900\text{ GB/s}$', '900 GB/s')
    text = text.replace(r'$0.22\text{ ms}$', '0.22 ms')
    text = text.replace(r'$28\text{ \mu s}$', '28 us')
    text = text.replace(r'$222\text{ \mu s}$', '222 us')
    
    return text

def main():
    md_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/inet4ai_submission_draft.md"
    html_path = "/home/palakm/MoEServingSim/scratch/inet4ai_submission_draft.html"
    pdf_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/inet4ai_submission_draft.pdf"
    workspace_pdf_path = "/home/palakm/MoEServingSim/inet4ai_submission_draft.pdf"
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    md_content = replace_math_and_chars(md_content)
    md_content = add_page_breaks(md_content)
    
    # Convert markdown to HTML
    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    # Process blockquotes into styled alerts
    alert_pattern = r'<blockquote>\s*<p>\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(?:\n|<br\s*/?>)?'
    html_body = re.sub(
        alert_pattern,
        lambda m: f'<blockquote class="alert alert-{m.group(1).lower()}"><p>',
        html_body,
        flags=re.IGNORECASE
    )
    
    full_html = HTML_TEMPLATE.format(content=html_body)
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"HTML draft written: {html_path}")
    
    # Generate PDF
    with open(pdf_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(full_html, dest=result_file)
        
    if not pisa_status.err:
        print(f"PDF generated successfully at: {pdf_path}")
        # Copy to workspace root
        import shutil
        shutil.copy(pdf_path, workspace_pdf_path)
        print(f"Copied to workspace root: {workspace_pdf_path}")
    else:
        print(f"Error generating PDF: {pisa_status.err}")

if __name__ == "__main__":
    main()
