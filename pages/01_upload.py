import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

st.title("Data Upload & Validation")

st.markdown("""
<div class="info-box">
    Upload gene expression datasets in CSV, TSV, XLSX, or Parquet format.
    The platform will automatically validate and detect the expression matrix structure.
</div>
""", unsafe_allow_html=True)

# File upload section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'tsv', 'xlsx', 'parquet', 'feather'],
        help="Upload gene expression data with samples as rows and genes as columns"
    )
    
    if uploaded_file is not None:
        # Determine file type and read
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            with st.spinner("Reading file..."):
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension == 'tsv':
                    df = pd.read_csv(uploaded_file, sep='\t')
                elif file_extension == 'xlsx':
                    df = pd.read_excel(uploaded_file)
                elif file_extension in ['parquet', 'feather']:
                    df = pd.read_parquet(uploaded_file) if file_extension == 'parquet' else pd.read_feather(uploaded_file)
                
                st.session_state.data = df
                st.success(f"Successfully loaded {uploaded_file.name}")
                
                # Log upload
                st.session_state.pipeline_status['upload'] = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': uploaded_file.name,
                    'status': 'success',
                    'shape': df.shape
                }
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.session_state.pipeline_status['upload'] = {
                'timestamp': datetime.now().isoformat(),
                'filename': uploaded_file.name,
                'status': 'error',
                'error': str(e)
            }

with col2:
    st.markdown("### Upload Metadata (Optional)")
    
    metadata_file = st.file_uploader(
        "Choose metadata file",
        type=['csv', 'tsv', 'xlsx'],
        help="Optional: Upload clinical metadata with sample IDs"
    )
    
    if metadata_file is not None:
        try:
            file_extension = metadata_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                metadata_df = pd.read_csv(metadata_file)
            elif file_extension == 'tsv':
                metadata_df = pd.read_csv(metadata_file, sep='\t')
            elif file_extension == 'xlsx':
                metadata_df = pd.read_excel(metadata_file)
            
            st.session_state.metadata = metadata_df
            st.success(f"Successfully loaded metadata from {metadata_file.name}")
            
        except Exception as e:
            st.error(f"Error reading metadata: {str(e)}")

# Data validation section
if st.session_state.data is not None:
    st.markdown("---")
    st.markdown("### Data Validation Report")
    
    df = st.session_state.data
    
    # Create validation metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-label">Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]:,}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_count = df.isna().sum().sum()
        missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{missing_pct:.1f}%</div>
            <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{numeric_cols}</div>
            <div class="metric-label">Numeric Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview tabs
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Data Types", "Missing Values"])
    
    with tab1:
        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=300
        )
    
    with tab2:
        dtype_summary = df.dtypes.value_counts().reset_index()
        dtype_summary.columns = ['Data Type', 'Count']
        st.dataframe(dtype_summary, use_container_width=True)
        
        # Show non-numeric columns
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            st.warning(f"Non-numeric columns detected: {', '.join(non_numeric[:5])}")
    
    with tab3:
        # Missing values heatmap
        missing_matrix = df.isna().astype(int)
        
        if missing_count > 0:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(missing_matrix.T, cmap='Reds', cbar_kws={'label': 'Missing (1=Yes)'}, ax=ax)
            ax.set_xlabel('Samples')
            ax.set_ylabel('Features')
            ax.set_title('Missing Values Pattern')
            st.pyplot(fig)
        else:
            st.success("No missing values detected in the dataset")
    
    # Schema validation
    st.markdown("### Schema Validation")
    
    validation_passed = True
    validation_results = []
    
    # Check 1: Minimum sample count
    if df.shape[0] < 10:
        validation_results.append(("Sample Count", "FAIL", "Less than 10 samples"))
        validation_passed = False
    else:
        validation_results.append(("Sample Count", "PASS", f"{df.shape[0]} samples"))
    
    # Check 2: Minimum feature count
    if df.shape[1] < 100:
        validation_results.append(("Feature Count", "WARNING", "Less than 100 features"))
    else:
        validation_results.append(("Feature Count", "PASS", f"{df.shape[1]} features"))
    
    # Check 3: Numeric data percentage
    numeric_pct = (numeric_cols / df.shape[1]) * 100
    if numeric_pct < 90:
        validation_results.append(("Numeric Data", "WARNING", f"{numeric_pct:.1f}% numeric columns"))
    else:
        validation_results.append(("Numeric Data", "PASS", f"{numeric_pct:.1f}% numeric columns"))
    
    # Check 4: Missing value threshold
    if missing_pct > 50:
        validation_results.append(("Missing Values", "FAIL", f"{missing_pct:.1f}% missing values"))
        validation_passed = False
    elif missing_pct > 20:
        validation_results.append(("Missing Values", "WARNING", f"{missing_pct:.1f}% missing values"))
    else:
        validation_results.append(("Missing Values", "PASS", f"{missing_pct:.1f}% missing values"))
    
    # Display validation results
    validation_df = pd.DataFrame(validation_results, columns=['Check', 'Status', 'Details'])
    
    # Apply styling
    def color_status(val):
        if val == 'PASS':
            return 'background-color: #d4edda; color: #155724;'
        elif val == 'WARNING':
            return 'background-color: #fff3cd; color: #856404;'
        else:
            return 'background-color: #f8d7da; color: #721c24;'
    
    st.dataframe(
        validation_df.style.applymap(color_status, subset=['Status']),
        use_container_width=True
    )
    
    if validation_passed:
        st.success("Data validation passed. Proceed to preprocessing.")
    else:
        st.error("Data validation failed. Please check your dataset.")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Proceed to Preprocessing", type="primary"):
            st.switch_page("pages/02_preprocessing.py")
    
    with col2:
        if st.button("Reset Data", type="secondary"):
            st.session_state.data = None
            st.session_state.metadata = None
            st.rerun()
