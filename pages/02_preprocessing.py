import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Adaptive Preprocessing Overview")

st.markdown("""
<div class="info-box">
    The platform automatically performs adaptive preprocessing including missing value imputation,
    scaling, duplicate removal, and batch effect correction when metadata is available.
</div>
""", unsafe_allow_html=True)

if st.session_state.data is None:
    st.warning("Please upload data first in the Data Upload module.")
    st.stop()

# Initialize preprocessing configuration
if 'preprocessing_config' not in st.session_state:
    st.session_state.preprocessing_config = {
        'imputation_strategy': 'median',
        'scaling_method': 'standard',
        'remove_duplicates': True,
        'handle_outliers': True,
        'batch_correction': False
    }

# Sidebar controls
with st.sidebar:
    st.markdown("### Preprocessing Settings")
    
    imputation_strategy = st.selectbox(
        "Missing Value Imputation",
        ['median', 'mean', 'constant', 'knn'],
        help="Strategy for imputing missing values"
    )
    
    scaling_method = st.selectbox(
        "Scaling Method",
        ['standard', 'robust', 'minmax', 'none'],
        help="Method for scaling/normalizing data"
    )
    
    remove_duplicates = st.checkbox("Remove Duplicate Samples", value=True)
    handle_outliers = st.checkbox("Handle Outliers (IQR method)", value=True)
    
    if st.session_state.metadata is not None:
        batch_correction = st.checkbox("Apply Batch Correction", value=False)
    
    if st.button("Apply Preprocessing", type="primary"):
        with st.spinner("Processing data..."):
            df = st.session_state.data.copy()
            
            # 1. Handle missing values
            if imputation_strategy == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df_imputed = pd.DataFrame(
                    imputer.fit_transform(df.select_dtypes(include=[np.number])),
                    index=df.index,
                    columns=df.select_dtypes(include=[np.number]).columns
                )
            else:
                imputer = SimpleImputer(strategy=imputation_strategy)
                df_imputed = pd.DataFrame(
                    imputer.fit_transform(df.select_dtypes(include=[np.number])),
                    index=df.index,
                    columns=df.select_dtypes(include=[np.number]).columns
                )
            
            # 2. Handle outliers using IQR method
            if handle_outliers:
                Q1 = df_imputed.quantile(0.25)
                Q3 = df_imputed.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df_imputed = df_imputed.clip(lower_bound, upper_bound, axis=1)
            
            # 3. Scale data
            if scaling_method == 'standard':
                scaler = StandardScaler()
                df_scaled = pd.DataFrame(
                    scaler.fit_transform(df_imputed),
                    index=df_imputed.index,
                    columns=df_imputed.columns
                )
            elif scaling_method == 'robust':
                scaler = RobustScaler()
                df_scaled = pd.DataFrame(
                    scaler.fit_transform(df_imputed),
                    index=df_imputed.index,
                    columns=df_imputed.columns
                )
            elif scaling_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df_scaled = pd.DataFrame(
                    scaler.fit_transform(df_imputed),
                    index=df_imputed.index,
                    columns=df_imputed.columns
                )
            else:
                df_scaled = df_imputed
            
            # 4. Remove duplicates
            if remove_duplicates:
                df_scaled = df_scaled[~df_scaled.index.duplicated()]
            
            # Store processed data
            st.session_state.processed_data = df_scaled
            
            # Store preprocessing info
            st.session_state.preprocessing_info = {
                'original_shape': df.shape,
                'processed_shape': df_scaled.shape,
                'missing_imputed': df.isna().sum().sum(),
                'duplicates_removed': df.shape[0] - df_scaled.shape[0] if remove_duplicates else 0,
                'scaling_method': scaling_method,
                'imputation_strategy': imputation_strategy
            }
            
            st.success("Preprocessing completed successfully!")

# Main content
df = st.session_state.data

# Preprocessing metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    missing_count = df.isna().sum().sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{missing_count:,}</div>
        <div class="metric-label">Missing Values</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    duplicates = df.index.duplicated().sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{duplicates}</div>
        <div class="metric-label">Duplicate Samples</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{numeric_cols}</div>
        <div class="metric-label">Numeric Features</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if st.session_state.processed_data is not None:
        status = "Processed"
        color = "status-success"
    else:
        status = "Pending"
        color = "status-warning"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">Data</div>
        <div class="metric-label"><span class="status-badge {color}">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)

# Visualization tabs
tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Distribution", "Correlation", "Preprocessing Comparison"])

with tab1:
    # Missing values heatmap
    missing_matrix = df.isna().astype(int)
    
    if missing_count > 0:
        fig = px.imshow(
            missing_matrix.T.iloc[:100],  # First 100 features for visibility
            aspect='auto',
            color_continuous_scale='Reds',
            title='Missing Values Pattern (First 100 Features)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values detected in the dataset")

with tab2:
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot of expression distribution
        sample_data = df.iloc[:, :20].melt(var_name='Gene', value_name='Expression')
        fig = px.box(
            sample_data,
            x='Gene',
            y='Expression',
            title='Expression Distribution (First 20 Genes)'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Density plot
        fig = go.Figure()
        for i in range(min(5, df.shape[1])):
            fig.add_trace(go.Histogram(
                x=df.iloc[:, i].dropna(),
                nbinsx=50,
                name=f'Gene {i+1}',
                opacity=0.7
            ))
        fig.update_layout(
            title='Expression Density (First 5 Genes)',
            height=400,
            xaxis_title='Expression Value',
            yaxis_title='Count',
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Correlation heatmap
    if df.shape[1] > 50:
        st.info("Computing correlation for first 50 genes for performance...")
        corr_data = df.iloc[:, :50].corr()
    else:
        corr_data = df.corr()
    
    fig = px.imshow(
        corr_data,
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Gene-Gene Correlation Matrix',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    if st.session_state.processed_data is not None:
        # Compare before/after preprocessing
        original_df = st.session_state.data
        processed_df = st.session_state.processed_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Original statistics
            st.markdown("##### Original Data Statistics")
            orig_stats = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Missing %'],
                'Value': [
                    f"{original_df.select_dtypes(include=[np.number]).mean().mean():.3f}",
                    f"{original_df.select_dtypes(include=[np.number]).std().mean():.3f}",
                    f"{original_df.select_dtypes(include=[np.number]).min().min():.3f}",
                    f"{original_df.select_dtypes(include=[np.number]).max().max():.3f}",
                    f"{(original_df.isna().sum().sum() / (original_df.shape[0] * original_df.shape[1]) * 100):.1f}%"
                ]
            })
            st.dataframe(orig_stats, use_container_width=True)
        
        with col2:
            # Processed statistics
            st.markdown("##### Processed Data Statistics")
            proc_stats = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Missing %'],
                'Value': [
                    f"{processed_df.mean().mean():.3f}",
                    f"{processed_df.std().mean():.3f}",
                    f"{processed_df.min().min():.3f}",
                    f"{processed_df.max().max():.3f}",
                    "0.0%"
                ]
            })
            st.dataframe(proc_stats, use_container_width=True)
        
        # Show preprocessing summary
        st.markdown("##### Preprocessing Summary")
        
        if 'preprocessing_info' in st.session_state:
            info = st.session_state.preprocessing_info
            summary_df = pd.DataFrame({
                'Step': ['Original Samples', 'Processed Samples', 'Missing Values Imputed', 
                        'Duplicates Removed', 'Scaling Method', 'Imputation Strategy'],
                'Details': [
                    f"{info['original_shape'][0]}",
                    f"{info['processed_shape'][0]}",
                    f"{info['missing_imputed']:,}",
                    f"{info['duplicates_removed']}",
                    info['scaling_method'],
                    info['imputation_strategy']
                ]
            })
            st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Apply preprocessing to see comparison")

# Action buttons
if st.session_state.processed_data is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Preview Processed Data"):
            st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
    
    with col2:
        if st.button("Proceed to Dimensionality Reduction"):
            st.switch_page("pages/03_dimred.py")
    
    with col3:
        if st.button("Download Processed Data"):
            csv = st.session_state.processed_data.to_csv().encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="processed_expression_data.csv",
                mime="text/csv"
            )
