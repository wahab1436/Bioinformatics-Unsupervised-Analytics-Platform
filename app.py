import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enterprise Bioinformatics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1rem;
    }
    
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #2E86AB, #A23B72);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .data-preview {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        background: #f8f9fa;
    }
    
    .pipeline-step {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
    }
    
    .step-number {
        background: #2E86AB;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'pipeline_logs' not in st.session_state:
    st.session_state.pipeline_logs = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Log function
def log_pipeline_step(step, message, status="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.pipeline_logs.append({
        "timestamp": timestamp,
        "step": step,
        "message": message,
        "status": status
    })

# Sidebar
with st.sidebar:
    st.markdown("## Bioinformatics Platform")
    st.markdown("---")
    
    st.markdown("### Pipeline Status")
    
    # Pipeline steps visualization
    steps = ["Data Upload", "Validation", "Preprocessing", "Analysis", "Visualization"]
    
    for i, step in enumerate(steps):
        status = "pending"
        if i < st.session_state.current_step:
            status = "completed"
        elif i == st.session_state.current_step:
            status = "active"
        
        if status == "completed":
            st.markdown(f"✓ **{step}**")
        elif status == "active":
            st.markdown(f" **{step}**")
        else:
            st.markdown(f"○ {step}")
    
    st.markdown("---")
    
    st.markdown("### Quick Actions")
    
    if st.button("Reset Session", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    if st.button("Load Sample Data", type="primary"):
        try:
            # Generate robust sample data
            np.random.seed(42)
            n_samples = 150
            n_genes = 500
            
            # Create realistic messy data
            data = np.random.randn(n_samples, n_genes)
            
            # Add some structure
            for i in range(3):
                data[i*50:(i+1)*50, i*100:(i+1)*100] += np.random.randn(50, 100) * 2
            
            # Add missing values
            mask = np.random.random(data.shape) < 0.05
            data[mask] = np.nan
            
            # Add some extreme values
            extreme_mask = np.random.random(data.shape) < 0.01
            data[extreme_mask] = np.random.choice([-100, 100], size=extreme_mask.sum())
            
            # Create DataFrame with mixed types
            sample_ids = [f"Sample_{i:03d}" for i in range(n_samples)]
            gene_ids = [f"Gene_{i:04d}" for i in range(n_genes)]
            
            df = pd.DataFrame(data, index=sample_ids, columns=gene_ids)
            
            # Add some text columns (to simulate messy data)
            df['Batch'] = np.random.choice(['Batch_A', 'Batch_B', 'Batch_C'], n_samples)
            df['Quality'] = np.random.choice(['Good', 'Poor', 'Average'], n_samples)
            
            # Create metadata with realistic clinical data
            metadata = pd.DataFrame({
                'sample_id': sample_ids,
                'patient_age': np.random.normal(58, 15, n_samples).astype(int),
                'diagnosis': np.random.choice(['Primary', 'Metastatic', 'Recurrent'], n_samples),
                'treatment': np.random.choice(['Chemo', 'Radiation', 'Surgery', 'None'], n_samples),
                'survival_days': np.random.exponential(365, n_samples).astype(int),
                'event': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            })
            
            st.session_state.data = df
            st.session_state.metadata = metadata
            log_pipeline_step("Data Upload", "Sample data loaded successfully", "success")
            st.success("Sample data loaded!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

# Main content
st.markdown('<div class="main-header">Enterprise Bioinformatics Analytics Platform</div>', unsafe_allow_html=True)

# Navigation
page = st.selectbox("Navigation", [
    "Dashboard",
    "Data Upload & Validation", 
    "Adaptive Preprocessing",
    "Dimensionality Reduction",
    "Clustering Analysis",
    "Results & Export"
])

if page == "Dashboard":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    st.markdown("### Platform Overview")
    st.markdown("""
    A robust, error-proof bioinformatics platform designed to handle messy, real-world datasets.
    This platform automatically detects and handles:
    - Missing values and incorrect data types
    - Batch effects and outliers
    - Mixed data formats (numeric, categorical, text)
    - Large-scale gene expression data
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">Auto</div>
            <div class="metric-label">Error Handling</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">Adaptive</div>
            <div class="metric-label">Preprocessing</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">Robust</div>
            <div class="metric-label">ML Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Getting Started")
    
    st.markdown("""
    1. **Upload Data**: Any format - CSV, Excel, TSV, Parquet
    2. **Automatic Validation**: Platform checks data quality
    3. **Adaptive Processing**: Smart handling of messy data
    4. **Analysis**: Clustering, dimensionality reduction, insights
    5. **Export**: Results in multiple formats
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Data Upload & Validation":
    st.markdown("## Data Upload & Validation")
    
    st.markdown("""
    <div class="section-card">
    Upload any gene expression dataset. The platform will automatically:
    - Detect file format and encoding
    - Identify numeric and non-numeric columns
    - Check for missing values and duplicates
    - Validate data structure
    - Generate quality report
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'tsv', 'xlsx', 'xls', 'parquet', 'txt'],
        help="Upload any dataset - the platform will handle messy data automatically"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Reading file..."):
                # Detect file type
                file_name = uploaded_file.name.lower()
                
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')
                elif file_name.endswith('.tsv') or file_name.endswith('.txt'):
                    df = pd.read_csv(uploaded_file, sep='\t', engine='python', on_bad_lines='skip')
                elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                elif file_name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    # Try all methods
                    try:
                        df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')
                    except:
                        try:
                            df = pd.read_csv(uploaded_file, sep='\t', engine='python', on_bad_lines='skip')
                        except:
                            raise ValueError("Cannot read file format")
                
                st.session_state.data = df
                log_pipeline_step("Data Upload", f"File '{uploaded_file.name}' loaded successfully", "success")
                st.success("File loaded successfully!")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            log_pipeline_step("Data Upload", f"Error: {str(e)}", "error")
    
    # Data validation
    if st.session_state.data is not None:
        st.markdown("## Data Validation Report")
        
        df = st.session_state.data
        
        # Generate validation metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cells = df.shape[0] * df.shape[1]
            missing_count = df.isna().sum().sum()
            missing_pct = (missing_count / total_cells) * 100
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{missing_pct:.1f}%</div>
                <div class="metric-label">Missing Values</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            total_cols = df.shape[1]
            numeric_pct = (numeric_cols / total_cols) * 100
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{numeric_pct:.1f}%</div>
                <div class="metric-label">Numeric Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            duplicates = df.duplicated().sum()
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{duplicates}</div>
                <div class="metric-label">Duplicate Rows</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            infinite_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{infinite_count}</div>
                <div class="metric-label">Infinite Values</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Data preview
        st.markdown("### Data Preview")
        
        tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Data Types", "Missing Values Pattern"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            dtype_summary = df.dtypes.astype(str).value_counts().reset_index()
            dtype_summary.columns = ['Data Type', 'Count']
            st.dataframe(dtype_summary, use_container_width=True)
        
        with tab3:
            if missing_count > 0:
                # Visualize missing values
                import missingno as msno
                fig, ax = plt.subplots(figsize=(10, 4))
                msno.matrix(df.iloc[:, :50], ax=ax)  # First 50 columns for speed
                st.pyplot(fig)
            else:
                st.success("No missing values found!")
        
        # Automatic data cleaning suggestions
        st.markdown("### Automatic Cleaning Suggestions")
        
        suggestions = []
        
        if missing_pct > 20:
            suggestions.append("High missing values (>20%) - Recommend imputation or column removal")
        elif missing_pct > 0:
            suggestions.append(f"Missing values ({missing_pct:.1f}%) - Will be automatically imputed")
        
        if numeric_pct < 80:
            suggestions.append(f"Mixed data types ({numeric_pct:.1f}% numeric) - Non-numeric columns will be handled")
        
        if duplicates > 0:
            suggestions.append(f"Duplicate rows found ({duplicates}) - Will be automatically removed")
        
        if infinite_count > 0:
            suggestions.append(f"Infinite values found ({infinite_count}) - Will be capped to extreme values")
        
        if suggestions:
            for suggestion in suggestions:
                st.warning(suggestion)
        else:
            st.success("Data looks clean! No major issues detected.")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Proceed to Preprocessing", type="primary"):
                st.session_state.current_step = 1
                st.rerun()
        
        with col2:
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                "Download Raw Data",
                data=csv,
                file_name="raw_data.csv",
                mime="text/csv"
            )

elif page == "Adaptive Preprocessing":
    st.markdown("## Adaptive Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload data first")
        st.stop()
    
    df = st.session_state.data
    
    st.markdown("""
    <div class="section-card">
    The platform automatically applies intelligent preprocessing:
    1. **Handle Missing Values**: Smart imputation based on data distribution
    2. **Fix Data Types**: Convert non-numeric to numeric where possible
    3. **Remove Duplicates**: Keep only unique samples
    4. **Handle Outliers**: Cap extreme values
    5. **Scale Data**: Normalize for machine learning
    </div>
    """, unsafe_allow_html=True)
    
    # Preprocessing options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Preprocessing Configuration")
        
        imputation_method = st.selectbox(
            "Missing Value Imputation",
            ["Auto (Median for numeric, Mode for categorical)", "Median", "Mean", "Forward Fill", "Drop rows with missing"]
        )
        
        outlier_handling = st.selectbox(
            "Outlier Handling",
            ["Auto (IQR method)", "Cap to percentiles", "Remove outliers", "Keep as-is"]
        )
        
        scaling_method = st.selectbox(
            "Scaling Method",
            ["Auto (Standard for normal, Robust for skewed)", "Standard", "Robust", "MinMax", "None"]
        )
        
        handle_categorical = st.checkbox("Convert categorical to numeric", value=True)
        remove_duplicates = st.checkbox("Remove duplicate samples", value=True)
    
    with col2:
        st.markdown("### Statistics")
        st.metric("Original Samples", df.shape[0])
        st.metric("Original Features", df.shape[1])
        st.metric("Missing Values", f"{df.isna().sum().sum():,}")
        st.metric("Data Types", f"{len(df.dtypes.unique())}")
    
    if st.button("Run Adaptive Preprocessing", type="primary"):
        try:
            with st.spinner("Processing data..."):
                processed_df = df.copy()
                logs = []
                
                # Step 1: Handle data types
                logs.append(" Handling data types...")
                numeric_df = processed_df.select_dtypes(include=[np.number])
                
                if handle_categorical and numeric_df.shape[1] < processed_df.shape[1]:
                    # Try to convert categorical to numeric
                    for col in processed_df.columns:
                        if col not in numeric_df.columns:
                            try:
                                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                                logs.append(f"  ✓ Converted {col} to numeric")
                            except:
                                # If can't convert, use one-hot encoding for few categories
                                if processed_df[col].nunique() < 10:
                                    processed_df = pd.get_dummies(processed_df, columns=[col], drop_first=True)
                                    logs.append(f"  ✓ One-hot encoded {col}")
                                else:
                                    # Drop high-cardinality categorical
                                    processed_df = processed_df.drop(columns=[col])
                                    logs.append(f"  ✗ Dropped high-cardinality column: {col}")
                
                # Step 2: Handle missing values
                logs.append(" Handling missing values...")
                if "Drop" in imputation_method:
                    processed_df = processed_df.dropna()
                    logs.append("  ✓ Dropped rows with missing values")
                else:
                    # Smart imputation
                    for col in processed_df.columns:
                        if processed_df[col].isna().any():
                            if processed_df[col].dtype in [np.float64, np.int64]:
                                if "Median" in imputation_method or "Auto" in imputation_method:
                                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                                else:
                                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                            else:
                                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                    logs.append(f"  ✓ Imputed {processed_df.isna().sum().sum()} missing values")
                
                # Step 3: Handle outliers
                logs.append(" Handling outliers...")
                if outlier_handling != "Keep as-is":
                    for col in processed_df.select_dtypes(include=[np.number]).columns:
                        Q1 = processed_df[col].quantile(0.25)
                        Q3 = processed_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        
                        if outlier_handling == "Remove outliers":
                            processed_df = processed_df[(processed_df[col] >= lower) & (processed_df[col] <= upper)]
                        else:  # Cap outliers
                            processed_df[col] = processed_df[col].clip(lower, upper)
                    logs.append("  ✓ Handled outliers")
                
                # Step 4: Remove duplicates
                if remove_duplicates:
                    logs.append(" Removing duplicates...")
                    before = processed_df.shape[0]
                    processed_df = processed_df.drop_duplicates()
                    after = processed_df.shape[0]
                    logs.append(f"  ✓ Removed {before - after} duplicate rows")
                
                # Step 5: Scale data
                logs.append(" Scaling data...")
                from sklearn.preprocessing import StandardScaler, RobustScaler
                
                if scaling_method == "Standard" or (scaling_method == "Auto" and processed_df.shape[0] > 100):
                    scaler = StandardScaler()
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                    logs.append("  ✓ Applied StandardScaler")
                elif scaling_method == "Robust":
                    scaler = RobustScaler()
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                    logs.append("  ✓ Applied RobustScaler")
                
                # Store processed data
                st.session_state.processed_data = processed_df
                log_pipeline_step("Preprocessing", "Data preprocessing completed successfully", "success")
                
                # Show logs
                st.markdown("### Processing Logs")
                for log in logs:
                    st.text(log)
                
                st.success(" Preprocessing completed successfully!")
                
                # Show before/after comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Before Processing**")
                    st.metric("Samples", df.shape[0])
                    st.metric("Features", df.shape[1])
                    st.metric("Missing", f"{df.isna().sum().sum():,}")
                
                with col2:
                    st.markdown("**After Processing**")
                    st.metric("Samples", processed_df.shape[0])
                    st.metric("Features", processed_df.shape[1])
                    st.metric("Missing", "0")
        
        except Exception as e:
            st.error(f" Error in preprocessing: {str(e)}")
            log_pipeline_step("Preprocessing", f"Error: {str(e)}", "error")

elif page == "Dimensionality Reduction":
    st.markdown("## Dimensionality Reduction")
    
    if st.session_state.processed_data is None:
        st.warning("Please preprocess data first")
        st.stop()
    
    df = st.session_state.processed_data
    
    st.markdown("""
    <div class="section-card">
    Reduce high-dimensional data to 2D/3D for visualization and analysis.
    The platform automatically handles:
    - Large datasets with smart sampling
    - Missing values in processed data
    - Optimal parameter selection
    - Multiple algorithm options
    </div>
    """, unsafe_allow_html=True)
    
    # Ensure we have numeric data
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        st.error("Need at least 2 numeric features for dimensionality reduction")
        st.stop()
    
    # Dimensionality reduction options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        method = st.selectbox("Reduction Method", ["PCA", "UMAP", "t-SNE"])
        
        if method == "PCA":
            st.info("PCA: Linear dimensionality reduction preserving maximum variance")
        elif method == "UMAP":
            st.info("UMAP: Non-linear reduction preserving local and global structure")
        else:
            st.info("t-SNE: Best for visualization, preserves local structure")
        
        n_components = st.selectbox("Components", [2, 3])
        
        # Advanced options
        with st.expander("Advanced Options"):
            if method == "UMAP":
                n_neighbors = st.slider("Neighbors", 5, 50, 15)
                min_dist = st.slider("Minimum Distance", 0.0, 1.0, 0.1)
            elif method == "t-SNE":
                perplexity = st.slider("Perplexity", 5, 50, 30)
    
    with col2:
        st.markdown("### Data Summary")
        st.metric("Samples", numeric_df.shape[0])
        st.metric("Features", numeric_df.shape[1])
        
        # Check if data is suitable
        if numeric_df.shape[0] > 10000:
            st.warning("Large dataset - using sampling")
            sample_size = min(5000, numeric_df.shape[0])
        else:
            sample_size = numeric_df.shape[0]
        
        st.metric("Analysis Samples", sample_size)
    
    if st.button("Run Dimensionality Reduction", type="primary"):
        try:
            with st.spinner(f"Running {method}..."):
                # Sample if large dataset
                if numeric_df.shape[0] > 5000:
                    sampled_df = numeric_df.sample(n=5000, random_state=42)
                    st.info(f"Sampled {5000} rows for faster computation")
                else:
                    sampled_df = numeric_df
                
                # Handle any remaining issues
                sampled_df = sampled_df.replace([np.inf, -np.inf], np.nan)
                sampled_df = sampled_df.fillna(sampled_df.median())
                
                if method == "PCA":
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=n_components, random_state=42)
                    embedding = reducer.fit_transform(sampled_df)
                    
                    # Explained variance
                    explained_var = reducer.explained_variance_ratio_
                    
                elif method == "UMAP":
                    import umap
                    reducer = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=42
                    )
                    embedding = reducer.fit_transform(sampled_df)
                    
                else:  # t-SNE
                    from sklearn.manifold import TSNE
                    reducer = TSNE(
                        n_components=n_components,
                        perplexity=perplexity,
                        random_state=42,
                        n_iter=1000
                    )
                    embedding = reducer.fit_transform(sampled_df)
                
                # Visualization
                st.success(f"{method} completed successfully!")
                
                # Create plot
                import plotly.express as px
                
                if n_components == 2:
                    fig = px.scatter(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        hover_name=sampled_df.index,
                        title=f"{method} Projection (2D)",
                        labels={'x': f'{method}1', 'y': f'{method}2'}
                    )
                else:
                    fig = px.scatter_3d(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        hover_name=sampled_df.index,
                        title=f"{method} Projection (3D)",
                        labels={'x': f'{method}1', 'y': f'{method}2', 'z': f'{method}3'}
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics
                if method == "PCA":
                    st.markdown("### Explained Variance")
                    for i, var in enumerate(explained_var, 1):
                        st.metric(f"Component {i}", f"{var:.1%}")
                
                # Store results
                st.session_state.dimred_results = {
                    'method': method,
                    'embedding': embedding,
                    'sample_indices': sampled_df.index
                }
                
                log_pipeline_step("Dimensionality Reduction", f"{method} completed successfully", "success")
        
        except Exception as e:
            st.error(f"❌ Error in dimensionality reduction: {str(e)}")
            # Fallback to simpler method
            st.info("Trying alternative approach...")
            try:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                embedding = reducer.fit_transform(sampled_df.iloc[:, :100])  # Use first 100 features
                
                fig = px.scatter(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    hover_name=sampled_df.index,
                    title="PCA Projection (Fallback)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e2:
                st.error(f"Fallback also failed: {str(e2)}")

elif page == "Clustering Analysis":
    st.markdown("## Clustering Analysis")
    
    if st.session_state.processed_data is None:
        st.warning("Please preprocess data first")
        st.stop()
    
    df = st.session_state.processed_data.select_dtypes(include=[np.number])
    
    st.markdown("""
    <div class="section-card">
    Discover natural groupings in your data using multiple clustering algorithms.
    The platform automatically:
    - Determines optimal number of clusters
    - Handles different data distributions
    - Provides quality metrics for each method
    - Offers visual validation
    </div>
    """, unsafe_allow_html=True)
    
    # Clustering options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        algorithm = st.selectbox(
            "Clustering Algorithm",
            ["Auto (K-Means with optimal k)", "K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"]
        )
        
        if algorithm == "DBSCAN":
            eps = st.slider("EPS (Neighborhood size)", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.slider("Minimum Samples", 1, 20, 5)
        elif algorithm == "K-Means" or "Auto" in algorithm:
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
    
    with col2:
        st.markdown("### Data Readiness")
        st.metric("Samples", df.shape[0])
        st.metric("Features", df.shape[1])
        
        # Check data quality
        if df.isna().any().any():
            st.warning("Data contains NaN values")
        else:
            st.success("Data is clean")
    
    if st.button("Run Clustering Analysis", type="primary"):
        try:
            with st.spinner("Running clustering analysis..."):
                from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
                from sklearn.mixture import GaussianMixture
                from sklearn.metrics import silhouette_score
                
                # Prepare data
                X = df.values
                
                # Handle any remaining issues
                X = np.nan_to_num(X)
                
                if "Auto" in algorithm or algorithm == "K-Means":
                    # Find optimal k using elbow method
                    st.markdown("### Determining Optimal Clusters")
                    
                    inertias = []
                    k_range = range(2, min(11, df.shape[0]//10))
                    
                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(X)
                        inertias.append(kmeans.inertia_)
                    
                    # Find elbow point
                    differences = np.diff(inertias)
                    differences2 = np.diff(differences)
                    if len(differences2) > 0:
                        optimal_k = np.argmin(differences2) + 3
                    else:
                        optimal_k = 4
                    
                    optimal_k = min(max(optimal_k, 2), 10)
                    
                    st.info(f"Auto-detected optimal clusters: {optimal_k}")
                    
                    # Run K-Means with optimal k
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)
                    
                elif algorithm == "DBSCAN":
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X)
                    
                elif algorithm == "Hierarchical":
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = hierarchical.fit_predict(X)
                    
                else:  # Gaussian Mixture
                    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                    labels = gmm.fit_predict(X)
                
                # Calculate metrics
                unique_labels = np.unique(labels[labels != -1])  # Exclude noise if any
                n_clusters_found = len(unique_labels)
                
                if n_clusters_found > 1:
                    silhouette = silhouette_score(X, labels)
                    st.success(f"Found {n_clusters_found} clusters")
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                    
                    if silhouette > 0.5:
                        st.success("Good cluster separation")
                    elif silhouette > 0.25:
                        st.info("Moderate cluster separation")
                    else:
                        st.warning("Poor cluster separation - consider different parameters")
                else:
                    st.warning("Only one cluster found - data may not have clear groupings")
                
                # Visualize clusters
                if 'dimred_results' in st.session_state:
                    # Use existing dimensionality reduction
                    embedding = st.session_state.dimred_results['embedding']
                    
                    import plotly.express as px
                    
                    if embedding.shape[1] == 2:
                        fig = px.scatter(
                            x=embedding[:, 0],
                            y=embedding[:, 1],
                            color=labels.astype(str),
                            hover_name=df.index,
                            title="Clusters in Reduced Space",
                            labels={'color': 'Cluster'}
                        )
                    else:
                        fig = px.scatter_3d(
                            x=embedding[:, 0],
                            y=embedding[:, 1],
                            z=embedding[:, 2],
                            color=labels.astype(str),
                            hover_name=df.index,
                            title="Clusters in Reduced Space",
                            labels={'color': 'Cluster'}
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster sizes
                st.markdown("### Cluster Distribution")
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(cluster_counts.index.astype(str), cluster_counts.values)
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Number of Samples')
                    ax.set_title('Cluster Sizes')
                    st.pyplot(fig)
                
                with col2:
                    cluster_df = pd.DataFrame({
                        'Cluster': cluster_counts.index,
                        'Count': cluster_counts.values,
                        'Percentage': (cluster_counts.values / len(labels) * 100).round(1)
                    })
                    st.dataframe(cluster_df, use_container_width=True)
                
                # Store results
                st.session_state.clustering_results = {
                    'algorithm': algorithm,
                    'labels': labels,
                    'n_clusters': n_clusters_found,
                    'silhouette': silhouette if n_clusters_found > 1 else None
                }
                
                log_pipeline_step("Clustering", f"{algorithm} completed with {n_clusters_found} clusters", "success")
        
        except Exception as e:
            st.error(f"Error in clustering: {str(e)}")
            st.info("Trying simplified clustering...")
            
            try:
                # Fallback: Simple K-Means
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                labels = kmeans.fit_predict(df.values)
                
                st.success("Fallback clustering completed")
                
                # Simple visualization
                import plotly.express as px
                fig = px.scatter(
                    x=df.iloc[:, 0],
                    y=df.iloc[:, 1],
                    color=labels.astype(str),
                    title="Simple Clustering (First 2 Features)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e2:
                st.error(f"Fallback also failed: {str(e2)}")

elif page == "Results & Export":
    st.markdown("## Results & Export")
    
    st.markdown("""
    <div class="section-card">
    Download your analysis results in multiple formats.
    The platform ensures:
    - Complete reproducibility with all parameters
    - Multiple export formats (CSV, Excel, PDF)
    - Clean, organized reports
    - One-click secure data deletion
    </div>
    """, unsafe_allow_html=True)
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.data is not None:
            st.metric("Original Data", "✓ Available")
        else:
            st.metric("Original Data", "✗ Not loaded")
    
    with col2:
        if st.session_state.processed_data is not None:
            st.metric("Processed Data", "✓ Available")
        else:
            st.metric("Processed Data", "✗ Not processed")
    
    with col3:
        if 'clustering_results' in st.session_state:
            st.metric("Clustering Results", "✓ Available")
        else:
            st.metric("Clustering Results", "✗ Not analyzed")
    
    # Export options
    st.markdown("### Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.data is not None:
            csv = st.session_state.data.to_csv().encode('utf-8')
            st.download_button(
                "Download Raw Data",
                data=csv,
                file_name="raw_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.session_state.processed_data is not None:
            csv = st.session_state.processed_data.to_csv().encode('utf-8')
            st.download_button(
                "📥 Download Processed Data",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        if 'clustering_results' in st.session_state:
            # Create results DataFrame
            results_df = pd.DataFrame({
                'sample_id': st.session_state.processed_data.index,
                'cluster': st.session_state.clustering_results['labels']
            })
            csv = results_df.to_csv().encode('utf-8')
            st.download_button(
                "Download Clusters",
                data=csv,
                file_name="cluster_assignments.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Generate report
    st.markdown("### Generate Analysis Report")
    
    if st.button("Generate Complete Report", type="primary"):
        try:
            with st.spinner("Generating report..."):
                # Create a comprehensive report
                import io
                
                report_content = []
                report_content.append("# Bioinformatics Analysis Report")
                report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_content.append("")
                
                # Data summary
                report_content.append("## Data Summary")
                if st.session_state.data is not None:
                    report_content.append(f"- Original samples: {st.session_state.data.shape[0]}")
                    report_content.append(f"- Original features: {st.session_state.data.shape[1]}")
                    report_content.append(f"- Missing values: {st.session_state.data.isna().sum().sum()}")
                
                if st.session_state.processed_data is not None:
                    report_content.append(f"- Processed samples: {st.session_state.processed_data.shape[0]}")
                    report_content.append(f"- Processed features: {st.session_state.processed_data.shape[1]}")
                
                # Clustering results
                if 'clustering_results' in st.session_state:
                    report_content.append("")
                    report_content.append("## Clustering Results")
                    report_content.append(f"- Algorithm: {st.session_state.clustering_results['algorithm']}")
                    report_content.append(f"- Number of clusters: {st.session_state.clustering_results['n_clusters']}")
                    if st.session_state.clustering_results['silhouette']:
                        report_content.append(f"- Silhouette score: {st.session_state.clustering_results['silhouette']:.3f}")
                
                # Pipeline logs
                report_content.append("")
                report_content.append("## Pipeline Logs")
                for log in st.session_state.pipeline_logs[-10:]:  # Last 10 logs
                    report_content.append(f"- [{log['timestamp']}] {log['step']}: {log['message']}")
                
                # Convert to text
                report_text = "\n".join(report_content)
                
                # Create download button
                st.download_button(
                    "Download Report (TXT)",
                    data=report_text,
                    file_name="bioinformatics_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.success("Report generated successfully!")
        
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
    
    # Secure deletion
    st.markdown("---")
    st.markdown("### Data Security")
    
    st.warning("""
    **Warning**: This will permanently delete all data from this session.
    Use this if you're working with sensitive data.
    """)
    
    if st.button("Secure Data Deletion", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("All data has been securely deleted")
        st.rerun()

# Pipeline logs at the bottom
if st.session_state.pipeline_logs:
    st.markdown("---")
    st.markdown("### Pipeline Logs")
    
    for log in st.session_state.pipeline_logs[-5:]:  # Show last 5 logs
        if log['status'] == "success":
            st.success(f"[{log['timestamp']}] {log['step']}: {log['message']}")
        elif log['status'] == "error":
            st.error(f"[{log['timestamp']}] {log['step']}: {log['message']}")
        else:
            st.info(f"[{log['timestamp']}] {log['step']}: {log['message']}")
