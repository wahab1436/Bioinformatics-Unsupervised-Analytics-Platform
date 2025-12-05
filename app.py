import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enterprise Bioinformatics Analytics Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'pipeline_status' not in st.session_state:
    st.session_state.pipeline_status = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Sidebar
with st.sidebar:
    st.markdown("## Bioinformatics Analytics Platform")
    st.markdown("---")
    
    # Navigation
    st.markdown("### Navigation")
    page = st.selectbox(
        "Select Module",
        [
            "Data Upload & Validation",
            "Preprocessing Overview",
            "Dimensionality Reduction",
            "Clustering Hub",
            "Marker Genes & Heatmaps",
            "Pathway Enrichment",
            "Clinical Correlation",
            "Anomaly Detection",
            "AI-Driven Insights",
            "Live Monitoring & Logs",
            "Export & Data Deletion"
        ]
    )
    
    st.markdown("---")
    
    # Platform Status
    st.markdown("### Platform Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        data_status = "Available" if st.session_state.data is not None else "No Data"
        status_color = "status-success" if data_status == "Available" else "status-warning"
        st.markdown(f'<span class="status-badge {status_color}">{data_status}</span>', unsafe_allow_html=True)
    
    with status_col2:
        if st.session_state.processed_data is not None:
            st.markdown('<span class="status-badge status-success">Processed</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-warning">Raw</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### Quick Actions")
    
    if st.button("Clear Session", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    if st.button("Generate Sample Data", type="primary"):
        # Generate sample data for demonstration
        np.random.seed(42)
        n_samples = 200
        n_genes = 1000
        
        # Generate expression data
        expression_data = np.random.randn(n_samples, n_genes)
        
        # Add some structure
        cluster_centers = np.random.randn(4, 100) * 3
        cluster_sizes = [50, 50, 50, 50]
        
        start_idx = 0
        for i in range(4):
            end_idx = start_idx + cluster_sizes[i]
            expression_data[start_idx:end_idx, i*100:(i+1)*100] += cluster_centers[i]
            start_idx = end_idx
        
        # Create sample IDs
        sample_ids = [f"Sample_{i:03d}" for i in range(n_samples)]
        gene_ids = [f"Gene_{i:05d}" for i in range(n_genes)]
        
        # Create expression dataframe
        expression_df = pd.DataFrame(
            expression_data,
            index=sample_ids,
            columns=gene_ids
        )
        
        # Add some missing values
        mask = np.random.choice([True, False], size=expression_data.shape, p=[0.05, 0.95])
        expression_df[mask] = np.nan
        
        # Create metadata
        metadata = pd.DataFrame({
            'sample_id': sample_ids,
            'ER_status': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.7, 0.3]),
            'stage': np.random.choice(['I', 'II', 'III'], n_samples, p=[0.3, 0.5, 0.2]),
            'grade': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.5, 0.3]),
            'age': np.random.normal(58, 10, n_samples).astype(int),
            'relapse': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'survival_time': np.random.exponential(60, n_samples) + 12
        })
        
        st.session_state.data = expression_df
        st.session_state.metadata = metadata
        st.success("Sample data generated successfully")
        st.rerun()
    
    st.markdown("---")
    
    # System Info
    st.markdown("### System Information")
    st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("Version: 2.0.0")
    st.caption("Status: Operational")

# Main content based on selected page
if page == "Data Upload & Validation":
    exec(open("pages/01_upload.py").read())
elif page == "Preprocessing Overview":
    exec(open("pages/02_preprocessing.py").read())
elif page == "Dimensionality Reduction":
    exec(open("pages/03_dimred.py").read())
elif page == "Clustering Hub":
    exec(open("pages/04_clustering.py").read())
elif page == "Marker Genes & Heatmaps":
    exec(open("pages/05_marker_genes.py").read())
elif page == "Pathway Enrichment":
    exec(open("pages/06_pathway.py").read())
elif page == "Clinical Correlation":
    exec(open("pages/07_clinical.py").read())
elif page == "Anomaly Detection":
    exec(open("pages/08_anomaly.py").read())
elif page == "AI-Driven Insights":
    exec(open("pages/09_ai_insights.py").read())
elif page == "Live Monitoring & Logs":
    exec(open("pages/10_monitoring.py").read())
elif page == "Export & Data Deletion":
    exec(open("pages/11_export_delete.py").read())

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Enterprise Bioinformatics Analytics Platform | Version 2.0.0 | Secure Processing Environment</p>
    <p>For research use only. All data is processed in-memory and cleared upon session termination.</p>
</div>
""", unsafe_allow_html=True)
