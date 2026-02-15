"""
PDF Report Generator for Data Analysis Reports
Generates professional PDF reports from analysis results with charts
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from datetime import datetime
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def generate_pdf_report(analysis_data: dict, include_charts: bool = True) -> bytes:
    """
    Generate a professional PDF report from analysis data with optional charts
    
    Args:
        analysis_data: Dictionary containing analysis results
        include_charts: Whether to include visualization charts
        
    Returns:
        PDF file content as bytes
    """
    
    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                           rightMargin=0.5*inch, leftMargin=0.5*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Container for PDF elements
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Title
    story.append(Paragraph("Professional Data Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Dataset Info
    story.append(Paragraph(f"Dataset: <b>{analysis_data.get('table_name', 'Unknown')}</b>", heading_style))
    story.append(Paragraph(f"Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b>", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Overview
    story.append(Paragraph("Executive Overview", heading_style))
    story.append(Paragraph(analysis_data.get('overview', 'N/A'), body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Dataset Statistics
    story.append(Paragraph("Dataset Statistics", heading_style))
    stats = analysis_data.get('statistical_summary', {})
    
    stats_data = [
        ['Metric', 'Value'],
        ['Total Rows', f"{stats.get('row_count', 0):,}"],
        ['Total Columns', str(stats.get('column_count', 0))],
        ['Numeric Columns', str(stats.get('numeric_columns', 0))],
        ['Categorical Columns', str(stats.get('categorical_columns', 0))],
        ['Memory Usage', str(stats.get('memory_usage', 'N/A'))]
    ]
    
    stats_table = Table(stats_data, colWidths=[2.5*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
    ]))
    
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Data Quality Report
    story.append(Paragraph("Data Quality Assessment", heading_style))
    quality = analysis_data.get('data_quality', {})
    
    quality_data = [
        ['Quality Metric', 'Value'],
        ['Completeness', f"{quality.get('completeness', 0):.1f}%"],
        ['Duplicate Rows', str(quality.get('duplicate_rows', 0))],
        ['Columns with Missing Data', str(quality.get('columns_with_missing', 0))]
    ]
    
    quality_table = Table(quality_data, colWidths=[2.5*inch, 2*inch])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f0f8')])
    ]))
    
    story.append(quality_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Numeric Statistics
    numeric_stats = analysis_data.get('numeric_stats', {})
    if numeric_stats:
        story.append(Paragraph("Numeric Column Statistics", heading_style))
        
        numeric_data = [['Column', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']]
        for col, stats in list(numeric_stats.items())[:8]:  # Limit to 8 columns
            numeric_data.append([
                col[:15],  # Truncate long names
                f"{stats.get('mean', 0):.2f}",
                f"{stats.get('median', 0):.2f}",
                f"{stats.get('std', 0):.2f}",
                f"{stats.get('min', 0):.2f}",
                f"{stats.get('max', 0):.2f}"
            ])
        
        numeric_table = Table(numeric_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        numeric_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fffacd')])
        ]))
        
        story.append(numeric_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Anomalies
    anomalies = analysis_data.get('anomalies', [])
    if anomalies:
        story.append(Paragraph("Detected Anomalies", heading_style))
        for i, anomaly in enumerate(anomalies[:5], 1):
            story.append(Paragraph(f"• {anomaly}", body_style))
        story.append(Spacer(1, 0.2*inch))
    else:
        story.append(Paragraph("No Significant Anomalies Detected", heading_style))
        story.append(Spacer(1, 0.2*inch))
    
    # Add Charts Section if data available
    if include_charts:
        try:
            raw_data = analysis_data.get('raw_data')
            numeric_cols = analysis_data.get('numeric_cols', [])
            category_analysis = analysis_data.get('category_analysis', {})
            
            story.append(PageBreak())
            story.append(Paragraph("Data Visualizations", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Numeric Distributions
            if numeric_cols and raw_data is not None:
                story.append(Paragraph("Numeric Column Distributions", ParagraphStyle(
                    'SubHeading', parent=styles['Heading3'], fontSize=12, 
                    textColor=colors.HexColor('#667eea'), spaceAfter=8
                )))
                
                # Create histogram charts
                for col in numeric_cols[:2]:  # First 2 numeric columns
                    try:
                        fig = px.histogram(raw_data, x=col, nbins=30, title=f"Distribution of {col}",
                                         color_discrete_sequence=['#667eea'])
                        fig.update_layout(height=350, showlegend=False)
                        
                        # Convert plotly to image
                        img_bytes = fig.to_image(format="png", width=800, height=300)
                        img_buffer = io.BytesIO(img_bytes)
                        img = Image(img_buffer, width=6*inch, height=2.2*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.15*inch))
                    except:
                        pass
            
            # Category Analysis
            if category_analysis:
                story.append(PageBreak())
                story.append(Paragraph("Category Analysis", ParagraphStyle(
                    'SubHeading', parent=styles['Heading3'], fontSize=12,
                    textColor=colors.HexColor('#764ba2'), spaceAfter=8
                )))
                
                for col_name, data_info in list(category_analysis.items())[:2]:  # First 2 categories
                    try:
                        top_values = data_info.get('top_values', {})
                        if top_values:
                            fig = px.bar(x=list(top_values.values()), y=list(top_values.keys()),
                                       title=f"Top Values in {col_name}", orientation='h',
                                       color_discrete_sequence=['#764ba2'])
                            fig.update_layout(height=350, showlegend=False)
                            
                            img_bytes = fig.to_image(format="png", width=800, height=300)
                            img_buffer = io.BytesIO(img_bytes)
                            img = Image(img_buffer, width=6*inch, height=2.2*inch)
                            story.append(img)
                            story.append(Spacer(1, 0.15*inch))
                    except:
                        pass
            
            # Correlation Heatmap
            if numeric_cols and len(numeric_cols) >= 2 and raw_data is not None:
                story.append(PageBreak())
                story.append(Paragraph("Correlation Analysis", ParagraphStyle(
                    'SubHeading', parent=styles['Heading3'], fontSize=12,
                    textColor=colors.HexColor('#667eea'), spaceAfter=8
                )))
                
                try:
                    corr_data = raw_data[numeric_cols[:5]].corr()
                    fig = px.imshow(corr_data, text_auto=True, aspect="auto",
                                   color_continuous_scale='RdBu',
                                   title="Correlation Between Numeric Columns")
                    fig.update_layout(height=400, width=600)
                    
                    img_bytes = fig.to_image(format="png", width=700, height=500)
                    img_buffer = io.BytesIO(img_bytes)
                    img = Image(img_buffer, width=6*inch, height=4.3*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
                except:
                    pass
        except Exception as e:
            pass  # Skip charts if any error occurs
    
    # Page break before findings
    story.append(PageBreak())
    
    # Key Findings
    story.append(Paragraph("Key Findings", heading_style))
    findings = analysis_data.get('key_findings', [])
    for i, finding in enumerate(findings, 1):
        story.append(Paragraph(f"<b>{i}.</b> {finding}", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    recommendations = analysis_data.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"<b>{i}.</b> {rec}", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # AI-Generated Analysis
    story.append(Paragraph("Detailed Analysis Report", heading_style))
    full_report = analysis_data.get('full_report', 'N/A')
    # Limit report length for PDF
    truncated_report = full_report[:1500] if len(full_report) > 1500 else full_report
    story.append(Paragraph(truncated_report, body_style))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Generated by Retail Insights Assistant | Professional Data Analytics", footer_style))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF bytes
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


if __name__ == "__main__":
    # Example usage
    sample_analysis = {
        "table_name": "Sample_Data",
        "overview": "Sample dataset for testing",
        "statistical_summary": {
            "row_count": 1000,
            "column_count": 10,
            "numeric_columns": 5,
            "categorical_columns": 5,
            "memory_usage": "1.5 MB"
        },
        "data_quality": {
            "completeness": 95.5,
            "duplicate_rows": 10,
            "columns_with_missing": 2
        },
        "numeric_stats": {
            "sales": {"mean": 500, "median": 450, "std": 150, "min": 100, "max": 1000}
        },
        "key_findings": ["Finding 1", "Finding 2"],
        "recommendations": ["Recommendation 1"],
        "full_report": "This is a sample report"
    }
    
    pdf = generate_pdf_report(sample_analysis)
    with open("sample_report.pdf", "wb") as f:
        f.write(pdf)
    print("PDF generated successfully")
