"""
eval_report.py

Generates a PDF report from evaluation_results.json
showing accuracy statistics, category breakdown, and per-question results.
"""

import json
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

OUTPUT = "src/eval/evaluation_report.pdf"
RESULTS = "src/eval/evaluation_results.json"

GREEN  = HexColor("#10b981")
RED    = HexColor("#ef4444")
BLUE   = HexColor("#3b82f6")
PURPLE = HexColor("#8b5cf6")
ORANGE = HexColor("#f59e0b")
DARK   = HexColor("#0f172a")
LIGHT  = HexColor("#f8f9fc")
GRAY   = HexColor("#e2e6ef")
TEXT   = HexColor("#1e293b")
MUTED  = HexColor("#64748b")
WHITE  = white

def p(text, size=9, color=TEXT, bold=False, align="LEFT"):
    return Paragraph(text, ParagraphStyle("p", fontSize=size,
        fontName="Helvetica-Bold" if bold else "Helvetica",
        textColor=color, leading=14, spaceAfter=3,
        alignment={"LEFT":0,"CENTER":1,"RIGHT":2}.get(align,0)))

def make_bar(pct, color, width=3.5*inch):
    filled = max(0.01, width * (pct/100))
    empty = width - filled
    bar_f = Table([[""]], colWidths=[filled], rowHeights=[12])
    bar_f.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),color)]))
    bar_e = Table([[""]], colWidths=[empty], rowHeights=[12])
    bar_e.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),GRAY)]))
    outer = Table([[[bar_f, bar_e]]], colWidths=[width])
    return outer

with open(RESULTS) as f:
    data = json.load(f)

summary = data["summary"]
by_cat  = data["by_category"]
results = data["results"]

doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    rightMargin=0.8*inch, leftMargin=0.8*inch,
    topMargin=0.75*inch, bottomMargin=0.75*inch)

story = []

# ── HEADER ──
hdr = Table([[
    Paragraph("<b>INVESTIGATIVE RAG</b>", ParagraphStyle("h", fontSize=18,
        fontName="Helvetica-Bold", textColor=WHITE)),
    Paragraph("Evaluation Report", ParagraphStyle("hr", fontSize=10,
        fontName="Helvetica", textColor=HexColor("#94a3b8"), alignment=2)),
]], colWidths=[4*inch, 3*inch])
hdr.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),DARK),
    ("TOPPADDING",(0,0),(-1,-1),16),("BOTTOMPADDING",(0,0),(-1,-1),10),
    ("LEFTPADDING",(0,0),(0,0),16),("RIGHTPADDING",(-1,0),(-1,0),16),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(hdr)

sub = Table([[
    Paragraph(f"RAG Accuracy & Consistency Evaluation  |  {data['timestamp'][:10]}",
        ParagraphStyle("sb", fontSize=8, fontName="Helvetica", textColor=HexColor("#94a3b8"))),
    Paragraph("Capstone Group 2",
        ParagraphStyle("sb2", fontSize=8, fontName="Helvetica",
            textColor=HexColor("#64748b"), alignment=2)),
]], colWidths=[4.5*inch, 2.5*inch])
sub.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),HexColor("#1e293b")),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),7),
    ("LEFTPADDING",(0,0),(-1,-1),16),("RIGHTPADDING",(-1,0),(-1,0),16),
]))
story.append(sub)
story.append(Spacer(1, 14))

# ── SUMMARY CARDS ──
acc = summary["accuracy_pct"]
acc_color = GREEN if acc >= 70 else ORANGE if acc >= 50 else RED

cards = [
    (f"{summary['accuracy_pct']}%", "Overall Accuracy", acc_color),
    (f"{summary['passed']}/{summary['total']}", "Questions Passed", GREEN),
    (f"{summary['avg_keyword_score']*100:.0f}%", "Avg Keyword Match", BLUE),
    (f"{summary['avg_response_time_sec']}s", "Avg Response Time", PURPLE),
]
card_cells = []
for val, label, color in cards:
    cell = Table([[
        Paragraph(val, ParagraphStyle("cv", fontSize=20, fontName="Helvetica-Bold",
            textColor=color, alignment=1)),
        Paragraph(label, ParagraphStyle("cl", fontSize=8, fontName="Helvetica",
            textColor=MUTED, alignment=1)),
    ]], colWidths=[1.6*inch])
    cell.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LIGHT),
        ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("LINEABOVE",(0,0),(-1,0),3,color),
    ]))
    card_cells.append(cell)

cards_row = Table([card_cells], colWidths=[1.7*inch]*4)
cards_row.setStyle(TableStyle([
    ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),
]))
story.append(cards_row)
story.append(Spacer(1, 16))

# ── ACCURACY BAR ──
story.append(p("Overall System Accuracy", size=10, bold=True))
story.append(Spacer(1, 4))
acc_bar = Table([[
    Table([[""]], colWidths=[6.9*inch*(acc/100)], rowHeights=[20]),
]], colWidths=[6.9*inch])
acc_bar.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1), GRAY),
]))
filled_bar = Table([[""]], colWidths=[6.9*inch*(acc/100)], rowHeights=[20])
filled_bar.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1), acc_color)]))
story.append(filled_bar)
story.append(Spacer(1, 4))
story.append(p(f"{acc}% of questions passed ({summary['passed']} out of {summary['total']})",
    color=MUTED, size=8))
story.append(Spacer(1, 14))

# ── BY CATEGORY ──
story.append(p("Results by Category", size=10, bold=True))
story.append(Spacer(1, 6))

cat_rows = [[
    p("Category", bold=True, color=MUTED, size=8),
    p("Pass/Total", bold=True, color=MUTED, size=8),
    p("Accuracy", bold=True, color=MUTED, size=8),
    p("Progress", bold=True, color=MUTED, size=8),
]]
for cat, stats in by_cat.items():
    cat_acc = stats["passed"] / stats["total"] * 100
    cat_color = GREEN if cat_acc >= 70 else ORANGE if cat_acc >= 50 else RED
    cat_rows.append([
        p(cat, size=8.5),
        p(f"{stats['passed']}/{stats['total']}", size=8.5, bold=True),
        p(f"{cat_acc:.0f}%", size=8.5, bold=True, color=cat_color),
        make_bar(cat_acc, cat_color, width=2.5*inch),
    ])

cat_table = Table(cat_rows, colWidths=[2.2*inch, 0.9*inch, 0.7*inch, 2.6*inch])
cat_table.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),HexColor("#f0f2f7")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LIGHT]),
    ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
    ("LEFTPADDING",(0,0),(-1,-1),8),
    ("GRID",(0,0),(-1,-1),0.3,GRAY),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(cat_table)
story.append(Spacer(1, 14))

# ── PER QUESTION RESULTS ──
story.append(p("Per Question Results", size=10, bold=True))
story.append(Spacer(1, 6))

q_rows = [[
    p("ID", bold=True, color=MUTED, size=8),
    p("Question", bold=True, color=MUTED, size=8),
    p("Dataset", bold=True, color=MUTED, size=8),
    p("KW Score", bold=True, color=MUTED, size=8),
    p("Time", bold=True, color=MUTED, size=8),
    p("Status", bold=True, color=MUTED, size=8),
]]
for r in results:
    status_color = GREEN if r["passed"] else RED
    kw_color = GREEN if r["keyword_score"] >= 0.7 else ORANGE if r["keyword_score"] >= 0.5 else RED
    q_rows.append([
        p(r["id"], size=7.5, color=MUTED),
        p(r["question"][:55] + ("..." if len(r["question"]) > 55 else ""), size=7.5),
        p(r["dataset"].upper(), size=7.5),
        p(f"{r['keyword_score']*100:.0f}%", size=8, bold=True, color=kw_color),
        p(f"{r['response_time_sec']}s", size=7.5, color=MUTED),
        p(r["status"], size=8, bold=True, color=status_color),
    ])

q_table = Table(q_rows, colWidths=[0.7*inch, 2.9*inch, 0.6*inch, 0.7*inch, 0.5*inch, 0.5*inch])
q_table.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),HexColor("#f0f2f7")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LIGHT]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),6),
    ("GRID",(0,0),(-1,-1),0.3,GRAY),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(q_table)
story.append(Spacer(1, 14))

# ── FOOTER ──
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1, 5))
story.append(p("Capstone Group 2 — Multi-Agent RAG / Investigative Intelligence  |  Evaluation Framework: DeepEval",
    size=7.5, color=MUTED, align="CENTER"))

doc.build(story)
print(f"Report saved to: {OUTPUT}")
