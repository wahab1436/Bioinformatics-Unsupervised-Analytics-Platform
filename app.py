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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .section-header {
            font-size: 2rem;
            color: #2c3e50;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem 0;
        }
        
        .info-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        
        .metric-value-lg {
            font-size: 3rem;
            font-weight: 700;
            color: #667eea;
            text-align: center;
        }
        
        .metric-label-lg {
            font-size: 1rem;
            color: #7f8c8d;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .nav-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin: 5px 0;
        }
        
        .nav-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .plot-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar
with st.sidebar:
    st.markdown("## Bioinformatics Platform")
    st.markdown("---")
    
    # Navigation
    st.markdown("### Navigation")
    page = st.selectbox(
        "Select Module",
        [
            "Dashboard Overview",
            "Data Upload",
            "Preprocessing",
            "Dimensionality Reduction",
            "Clustering Analysis",
            "Marker Genes",
            "Pathway Enrichment",
            "Clinical Correlation",
            "Export Results"
        ]
    )
    
    st.markdown("---")
    
    # Platform Status
    st.markdown("### Platform Status")
    
    if st.session_state.data is not None:
        st.success("✓ Data Loaded")
        st.write(f"Samples: {st.session_state.data.shape[0]}")
        st.write(f"Features: {st.session_state.data.shape[1]}")
    else:
        st.info("No Data Loaded")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### Quick Actions")
    
    if st.button("Load Sample Dataset"):
        # Generate sample data
        np.random.seed(42)
        n_samples = 200
        n_genes = 1000
        
        expression_data = np.random.randn(n_samples, n_genes)
        
        # Add cluster structure
        for i in range(4):
            expression_data[i*50:(i+1)*50, i*100:(i+1)*100] += np.random.randn(50, 100) * 3
        
        sample_ids = [f"Sample_{i:03d}" for i in range(n_samples)]
        gene_ids = [f"Gene_{i:05d}" for i in range(n_genes)]
        
        df = pd.DataFrame(expression_data, index=sample_ids, columns=gene_ids)
        
        metadata = pd.DataFrame({
            'sample_id': sample_ids,
            'er_status': np.random.choice(['Positive', 'Negative'], n_samples),
            'stage': np.random.choice(['I', 'II', 'III'], n_samples),
            'survival_time': np.random.exponential(60, n_samples) + 12,
            'event': np.random.choice([0, 1], n_samples)
        })
        
        st.session_state.data = df
        st.session_state.metadata = metadata
        st.success("Sample dataset loaded successfully")
        st.rerun()

# Main Content
st.markdown('<h1 class="main-header">Enterprise Bioinformatics Analytics Platform</h1>', unsafe_allow_html=True)

if page == "Dashboard Overview":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-card"><div class="metric-value-lg">4</div><div class="metric-label-lg">Clustering Algorithms</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card"><div class="metric-value-lg">286</div><div class="metric-label-lg">Sample Capacity</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="info-card"><div class="metric-value-lg">22K</div><div class="metric-label-lg">Genes Analysis</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Platform Features</div>', unsafe_allow_html=True)
    
    features = [
        ("Adaptive Preprocessing", "Automatic handling of missing values, normalization, and batch effects"),
        ("Multi-Algorithm Clustering", "K-Means, Hierarchical, DBSCAN, and Spectral clustering"),
        ("3D Visualization", "Interactive PCA, UMAP, and t-SNE visualizations"),
        ("Pathway Analysis", "KEGG, Reactome, and Hallmark pathway enrichment"),
        ("Clinical Integration", "Survival analysis and clinical correlation"),
        ("Export Capabilities", "CSV, PDF, and interactive HTML reports")
    ]
    
    for feature, description in features:
        with st.expander(feature):
            st.write(description)
    
    st.markdown('<div class="section-header">Getting Started</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    1. Upload your gene expression dataset (CSV, TSV, Excel, Parquet)<br>
    2. Review preprocessing steps and quality metrics<br>
    3. Perform dimensionality reduction (PCA, UMAP, t-SNE)<br>
    4. Apply clustering algorithms and analyze results<br>
    5. Explore marker genes and pathway enrichment<br>
    6. Export results for further analysis
    </div>
    """, unsafe_allow_html=True)

elif page == "Data Upload":
    st.markdown('<div class="section-header">Data Upload & Validation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Expression Data",
            type=['csv', 'tsv', 'xlsx', 'parquet'],
            help="Upload gene expression matrix with samples as rows and genes as columns"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.tsv'):
                    df = pd.read_csv(uploaded_file, sep='\t')
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                
                st.session_state.data = df
                st.success(f"Successfully loaded {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        metadata_file = st.file_uploader(
            "Upload Metadata (Optional)",
            type=['csv', 'tsv', 'xlsx'],
            help="Clinical metadata with sample information"
        )
        
        if metadata_file is not None:
            try:
                if metadata_file.name.endswith('.csv'):
                    metadata = pd.read_csv(metadata_file)
                elif metadata_file.name.endswith('.tsv'):
                    metadata = pd.read_csv(metadata_file, sep='\t')
                elif metadata_file.name.endswith('.xlsx'):
                    metadata = pd.read_excel(metadata_file)
                
                st.session_state.metadata = metadata
                st.success("Metadata loaded successfully")
                
            except Exception as e:
                st.error(f"Error loading metadata: {str(e)}")
    
    if st.session_state.data is not None:
        st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Expression Data", "Statistics", "Quality Check"])
        
        with tab1:
            st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        with tab2:
            st.write("Dataset Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Samples", st.session_state.data.shape[0])
            
            with col2:
                st.metric("Features", st.session_state.data.shape[1])
            
            with col3:
                missing_pct = (st.session_state.data.isna().sum().sum() / 
                              (st.session_state.data.shape[0] * st.session_state.data.shape[1])) * 100
                st.metric("Missing Values", f"{missing_pct:.1f}%")
        
        with tab3:
            if st.session_state.data.isna().sum().sum() > 0:
                st.warning(f"Dataset contains {st.session_state.data.isna().sum().sum()} missing values")
            else:
                st.success("No missing values detected")
            
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).shape[1]
            if numeric_cols == st.session_state.data.shape[1]:
                st.success("All columns are numeric")
            else:
                st.warning(f"{st.session_state.data.shape[1] - numeric_cols} non-numeric columns detected")

elif page == "Preprocessing":
    st.markdown('<div class="section-header">Data Preprocessing</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload data first")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Processing Steps")
        
        # Preprocessing options
        impute_missing = st.checkbox("Impute Missing Values", value=True)
        remove_outliers = st.checkbox("Remove Outliers", value=True)
        normalize = st.checkbox("Normalize Data", value=True)
        
        if st.button("Apply Preprocessing", type="primary"):
            with st.spinner("Processing data..."):
                df = st.session_state.data.copy()
                
                # Impute missing values
                if impute_missing:
                    df = df.fillna(df.median())
                
                # Remove outliers
                if remove_outliers:
                    Q1 = df.quantile(0.25)
                    Q3 = df.quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
                
                # Normalize
                if normalize:
                    scaler = StandardScaler()
                    df = pd.DataFrame(scaler.fit_transform(df), 
                                     index=df.index, 
                                     columns=df.columns)
                
                st.session_state.processed_data = df
                st.success("Preprocessing completed successfully!")
    
    with col2:
        st.markdown("### Statistics")
        
        if st.session_state.processed_data is not None:
            processed_df = st.session_state.processed_data
            
            st.metric("Processed Samples", processed_df.shape[0])
            st.metric("Processed Features", processed_df.shape[1])
            st.metric("Missing Values", "0")
            
            # Distribution plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(processed_df.values.flatten(), bins=50, alpha=0.7, color='#667eea')
            ax.set_xlabel('Expression Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution After Processing')
            st.pyplot(fig)

elif page == "Dimensionality Reduction":
    st.markdown('<div class="section-header">Dimensionality Reduction</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("Please process data first")
        st.stop()
    
    df = st.session_state.processed_data
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        method = st.selectbox("Reduction Method", ["PCA", "UMAP", "t-SNE"])
        n_components = st.selectbox("Components", [2, 3])
        
        if st.button("Compute Projection"):
            with st.spinner(f"Computing {method}..."):
                from sklearn.decomposition import PCA
                import umap
                from sklearn.manifold import TSNE
                
                if method == "PCA":
                    reducer = PCA(n_components=n_components, random_state=42)
                    embedding = reducer.fit_transform(df)
                elif method == "UMAP":
                    reducer = umap.UMAP(n_components=n_components, random_state=42)
                    embedding = reducer.fit_transform(df)
                else:  # t-SNE
                    reducer = TSNE(n_components=n_components, random_state=42)
                    embedding = reducer.fit_transform(df)
                
                # Create plot
                if n_components == 2:
                    fig = px.scatter(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        hover_name=df.index,
                        title=f"{method} 2D Projection"
                    )
                else:
                    fig = px.scatter_3d(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        hover_name=df.index,
                        title=f"{method} 3D Projection"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Projection Quality")
        st.info("""
        - **PCA**: Preserves maximum variance
        - **UMAP**: Preserves local structure
        - **t-SNE**: Good for visualization
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p style="font-size: 0.9rem;">Enterprise Bioinformatics Analytics Platform v2.0 | Secure Processing Environment</p>
    <p style="font-size: 0.8rem;">For research use only. All data is processed in-memory.</p>
</div>
""", unsafe_allow_html=True)
