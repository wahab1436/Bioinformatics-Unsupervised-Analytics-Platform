import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Dimensionality Reduction")

st.markdown("""
<div class="info-box">
    Visualize high-dimensional gene expression data in 2D/3D space using PCA, UMAP, or t-SNE.
    Color points by clusters or clinical variables for pattern discovery.
</div>
""", unsafe_allow_html=True)

if st.session_state.processed_data is None:
    st.warning("Please preprocess data first in the Preprocessing module.")
    st.stop()

df = st.session_state.processed_data

# Sidebar controls
with st.sidebar:
    st.markdown("### Reduction Settings")
    
    method = st.selectbox(
        "Reduction Method",
        ["PCA", "UMAP", "t-SNE"],
        help="Choose dimensionality reduction algorithm"
    )
    
    n_components = st.selectbox(
        "Number of Components",
        [2, 3],
        help="Number of dimensions for visualization"
    )
    
    if method == "UMAP":
        n_neighbors = st.slider("UMAP Neighbors", 5, 50, 15)
        min_dist = st.slider("UMAP Min Distance", 0.0, 1.0, 0.1, 0.05)
    elif method == "t-SNE":
        perplexity = st.slider("t-SNE Perplexity", 5, 50, 30)
    
    coloring = st.selectbox(
        "Color By",
        ["None", "Cluster"] + (["ER Status", "Stage", "Grade"] 
                              if st.session_state.metadata is not None else [])
    )
    
    point_size = st.slider("Point Size", 1, 10, 5)
    
    if st.button("Compute Embedding", type="primary"):
        with st.spinner(f"Computing {method} embedding..."):
            # Compute embedding
            if method == "PCA":
                reducer = PCA(n_components=n_components, random_state=42)
                embedding = reducer.fit_transform(df)
                explained_var = reducer.explained_variance_ratio_
                
                st.session_state.dimred_results = {
                    'method': method,
                    'embedding': embedding,
                    'explained_variance': explained_var,
                    'reducer': reducer
                }
                
            elif method == "UMAP":
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=42
                )
                embedding = reducer.fit_transform(df)
                
                st.session_state.dimred_results = {
                    'method': method,
                    'embedding': embedding,
                    'reducer': reducer
                }
                
            else:  # t-SNE
                reducer = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=42,
                    n_iter=1000
                )
                embedding = reducer.fit_transform(df)
                
                st.session_state.dimred_results = {
                    'method': method,
                    'embedding': embedding,
                    'reducer': reducer
                }
            
            st.success(f"{method} embedding computed successfully!")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Visualization")
    
    if 'dimred_results' in st.session_state:
        results = st.session_state.dimred_results
        embedding = results['embedding']
        
        # Prepare color data
        if coloring != "None":
            if coloring == "Cluster" and 'clusters' in st.session_state.analysis_results:
                color_data = st.session_state.analysis_results['clusters']
                color_label = "Cluster"
            elif coloring in ["ER Status", "Stage", "Grade"] and st.session_state.metadata is not None:
                # Map sample IDs to metadata
                metadata_df = st.session_state.metadata.set_index('sample_id')
                common_samples = df.index.intersection(metadata_df.index)
                
                if len(common_samples) > 0:
                    color_map = {}
                    for sample in df.index:
                        if sample in metadata_df.index:
                            color_map[sample] = metadata_df.loc[sample, coloring.lower().replace(' ', '_')]
                        else:
                            color_map[sample] = 'Unknown'
                    
                    color_data = [color_map[sample] for sample in df.index]
                    color_label = coloring
                else:
                    color_data = None
                    color_label = None
            else:
                color_data = None
                color_label = None
        else:
            color_data = None
            color_label = None
        
        # Create plot
        if n_components == 2:
            plot_df = pd.DataFrame({
                'Component 1': embedding[:, 0],
                'Component 2': embedding[:, 1],
                'Sample': df.index,
                color_label: color_data if color_data else None
            })
            
            if color_data:
                fig = px.scatter(
                    plot_df,
                    x='Component 1',
                    y='Component 2',
                    color=color_label,
                    hover_name='Sample',
                    title=f"{results['method']} 2D Projection",
                    opacity=0.8
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x='Component 1',
                    y='Component 2',
                    hover_name='Sample',
                    title=f"{results['method']} 2D Projection",
                    opacity=0.8
                )
            
            fig.update_traces(marker=dict(size=point_size))
            
        else:  # 3D
            plot_df = pd.DataFrame({
                'Component 1': embedding[:, 0],
                'Component 2': embedding[:, 1],
                'Component 3': embedding[:, 2],
                'Sample': df.index,
                color_label: color_data if color_data else None
            })
            
            if color_data:
                fig = px.scatter_3d(
                    plot_df,
                    x='Component 1',
                    y='Component 2',
                    z='Component 3',
                    color=color_label,
                    hover_name='Sample',
                    title=f"{results['method']} 3D Projection",
                    opacity=0.8
                )
            else:
                fig = px.scatter_3d(
                    plot_df,
                    x='Component 1',
                    y='Component 2',
                    z='Component 3',
                    hover_name='Sample',
                    title=f"{results['method']} 3D Projection",
                    opacity=0.8
                )
            
            fig.update_traces(marker=dict(size=point_size))
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interactive controls
        st.markdown("#### Interactive Controls")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            show_labels = st.checkbox("Show Sample Labels", value=False)
        
        with col_b:
            if color_data:
                show_legend = st.checkbox("Show Legend", value=True)
        
        with col_c:
            export_plot = st.button("Export Plot as PNG")
    
    else:
        st.info("Compute an embedding to visualize the data")

with col2:
    st.markdown("### Statistics")
    
    if 'dimred_results' in st.session_state:
        results = st.session_state.dimred_results
        
        st.markdown(f"**Method:** {results['method']}")
        st.markdown(f"**Components:** {n_components}")
        st.markdown(f"**Samples:** {df.shape[0]}")
        st.markdown(f"**Features:** {df.shape[1]}")
        
        if 'explained_variance' in results:
            st.markdown("#### Variance Explained")
            for i, var in enumerate(results['explained_variance'], 1):
                st.metric(f"Component {i}", f"{var:.1%}")
        
        # Quality metrics
        st.markdown("#### Quality Metrics")
        
        # Calculate reconstruction error for PCA
        if results['method'] == 'PCA':
            from sklearn.metrics import mean_squared_error
            reconstructed = results['reducer'].inverse_transform(results['embedding'])
            mse = mean_squared_error(df.values, reconstructed)
            st.metric("Reconstruction MSE", f"{mse:.4f}")
        
        # Calculate nearest neighbor preservation for UMAP/t-SNE
        if results['method'] in ['UMAP', 't-SNE']:
            from sklearn.neighbors import NearestNeighbors
            
            # Sample for performance
            if df.shape[0] > 1000:
                sample_idx = np.random.choice(df.shape[0], 1000, replace=False)
                high_dim = df.iloc[sample_idx].values
                low_dim = results['embedding'][sample_idx]
            else:
                high_dim = df.values
                low_dim = results['embedding']
            
            # Find nearest neighbors in high and low dimensions
            nbrs_high = NearestNeighbors(n_neighbors=10).fit(high_dim)
            nbrs_low = NearestNeighbors(n_neighbors=10).fit(low_dim)
            
            distances_high, indices_high = nbrs_high.kneighbors(high_dim)
            distances_low, indices_low = nbrs_low.kneighbors(low_dim)
            
            # Calculate neighbor preservation
            preserved = 0
            total = 0
            
            for i in range(len(indices_high)):
                preserved += len(set(indices_high[i][1:]) & set(indices_low[i][1:]))
                total += len(indices_high[i][1:])
            
            preservation_score = preserved / total
            st.metric("Neighbor Preservation", f"{preservation_score:.1%}")

# Additional visualizations
if 'dimred_results' in st.session_state:
    st.markdown("---")
    st.markdown("### Additional Analysis")
    
    tab1, tab2 = st.tabs(["Component Analysis", "Cluster Separation"])
    
    with tab1:
        if results['method'] == 'PCA' and 'explained_variance' in results:
            # Scree plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, len(results['explained_variance']) + 1)),
                y=results['explained_variance'],
                name='Individual'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(results['explained_variance']) + 1)),
                y=np.cumsum(results['explained_variance']),
                name='Cumulative',
                mode='lines+markers'
            ))
            fig.update_layout(
                title='Scree Plot - Variance Explained by Principal Components',
                xaxis_title='Principal Component',
                yaxis_title='Variance Explained',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'clusters' in st.session_state.analysis_results and n_components == 2:
            from sklearn.metrics import silhouette_score
            
            clusters = st.session_state.analysis_results['clusters']
            silhouette = silhouette_score(results['embedding'], clusters)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Silhouette Score", f"{silhouette:.3f}")
                
                if silhouette > 0.5:
                    st.success("Good cluster separation")
                elif silhouette > 0.25:
                    st.info("Moderate cluster separation")
                else:
                    st.warning("Poor cluster separation")
            
            with col2:
                # Cluster distribution in embedding space
                cluster_df = pd.DataFrame({
                    'Component 1': results['embedding'][:, 0],
                    'Component 2': results['embedding'][:, 1],
                    'Cluster': clusters
                })
                
                fig = px.scatter(
                    cluster_df,
                    x='Component 1',
                    y='Component 2',
                    color='Cluster',
                    title='Cluster Separation in Reduced Space',
                    opacity=0.7
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# Export options
if 'dimred_results' in st.session_state:
    st.markdown("---")
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export embedding data
        embedding_df = pd.DataFrame(
            st.session_state.dimred_results['embedding'],
            columns=[f"{results['method']}{i+1}" for i in range(n_components)],
            index=df.index
        )
        
        csv_data = embedding_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Embedding Data",
            data=csv_data,
            file_name=f"{results['method'].lower()}_embedding.csv",
            mime="text/csv",
            type="primary"
        )
    
    with col2:
        if st.button("Proceed to Clustering"):
            st.switch_page("pages/04_clustering.py")
