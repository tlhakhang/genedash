import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scanpy as sc
from collections import OrderedDict
import scipy
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

def create_qc_metrics_figure(adata, title='Quality Control Metrics Distribution'):
    """
    Create a professional figure showing distribution plots of QC metrics
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with QC metrics in observation annotations
    title : str, optional
        Custom title for the figure
    
    Returns:
    --------
    plotly.graph_objs._figure.Figure
        Interactive distribution plot of QC metrics
    """
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import numpy as np
    import scipy.stats as stats

    # Professional color palette with RGBA for transparency
    colors = {
        'total_counts': {
            'line': 'rgb(44, 82, 130)',      # Deep blue
            'fill': 'rgba(44, 82, 130, 0.2)'  # Deep blue with transparency
        },
        'mitochondrial': {
            'line': 'rgb(39, 103, 73)',      # Dark green
            'fill': 'rgba(39, 103, 73, 0.2)'  # Dark green with transparency
        },
        'ribosomal': {
            'line': 'rgb(192, 86, 33)',      # Burnt orange
            'fill': 'rgba(192, 86, 33, 0.2)'  # Burnt orange with transparency
        }
    }

    # Create subplot figure
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f'<b>Total Counts</b><br><sup>Mean: {adata.obs["total_counts"].mean():.2f}</sup>', 
            f'<b>Mitochondrial %</b><br><sup>Mean: {adata.obs["pct_counts_mt"].mean():.2f}%</sup>', 
            f'<b>Ribosomal %</b><br><sup>Mean: {adata.obs["pct_counts_ribo"].mean():.2f}%</sup>'
        ],
        horizontal_spacing=0.05
    )

    # Metrics to plot
    metrics = [
        ('total_counts', colors['total_counts'], 'Total Counts', 'Counts'),
        ('pct_counts_mt', colors['mitochondrial'], 'Mitochondrial %', 'Percentage'),
        ('pct_counts_ribo', colors['ribosomal'], 'Ribosomal %', 'Percentage')
    ]

    # Create distribution plots for each metric
    for i, (metric, color_scheme, label, y_title) in enumerate(metrics, 1):
        data = adata.obs[metric]
        
        # Kernel Density Estimation
        kernel = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        density = kernel(x_range)

        # Calculate statistics
        mean = np.mean(data)
        median = np.median(data)

        # Distribution trace
        fig.add_trace(
            go.Scatter(
                x=x_range, 
                y=density, 
                mode='lines', 
                name=label,
                line=dict(color=color_scheme['line'], width=3),
                fill='tozeroy',
                fillcolor=color_scheme['fill']
            ),
            row=1, col=i
        )

        # Add mean line
        fig.add_shape(
            type='line',
            x0=mean,
            x1=mean,
            y0=0,
            y1=density.max(),
            line=dict(color=color_scheme['line'], width=2, dash='dot'),
            row=1, col=i
        )

        # Add median line
        fig.add_shape(
            type='line',
            x0=median,
            x1=median,
            y0=0,
            y1=density.max(),
            line=dict(color=color_scheme['line'], width=2, dash='dash'),
            row=1, col=i
        )

    # Update layout with professional styling
    fig.update_layout(
        title={
            'text': f'<b>{title}</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=500,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, Arial, sans-serif'),
        showlegend=False,
        margin=dict(t=80, l=50, r=50, b=50)
    )

    # Update axes styling
    for i in range(1, 4):
        fig.update_xaxes(
            title_text=['Counts', 'Percentage', 'Percentage'][i-1],
            gridcolor='rgba(0,0,0,0.05)',
            showline=True,
            linecolor='rgba(0,0,0,0.2)',
            tickfont=dict(size=10),
            row=1, 
            col=i
        )
        
        fig.update_yaxes(
            title_text='Density',
            gridcolor='rgba(0,0,0,0.05)',
            showline=True,
            linecolor='rgba(0,0,0,0.2)',
            tickfont=dict(size=10),
            row=1, 
            col=i
        )

    return fig

def handle_qc_metrics_plot_error(e):
    """Create an error visualization if plot generation fails"""
    import plotly.graph_objs as go
    
    fig = go.Figure()
    fig.add_annotation(
        text=f"Error generating plot: {str(e)}",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color='red', size=16)
    )

    return fig


class DataManager:
    
        
    def __init__(self, h5ad_path, cache_size=1000):
        self.h5ad_path = h5ad_path
        self.gene_cache = OrderedDict()
        self.cache_size = cache_size
        self.temp_files = set()
        self.adata = None  # Initialize to None
        self.load_data()


        
    def cleanup_temp_files(self):  # Add this method
        import os
        for file in self.temp_files:
            try:
                os.remove(file)
            except:
                pass
        self.temp_files.clear()
        
    def __del__(self):
        try:
            if hasattr(self, 'adata') and self.adata is not None:
                try:
                    self.adata.file.close()
                except:
                    pass
            self.cleanup_temp_files()
        except:
            pass



    def load_data(self):
        # Load in backed mode
        self.adata = sc.read_h5ad(self.h5ad_path, backed='r')
        
        # Store UMAP as numpy array instead of DataFrame
        self.umap = np.array(self.adata.obsm['X_umap'])
        
        # Store necessary data as numpy arrays
        self.genes = self.adata.var_names.tolist()
        self.obs_names = np.array(self.adata.obs_names)
        
        # Get categorical columns
        self.cat_columns = self.adata.obs.select_dtypes(
            include=['object', 'category']
        ).columns
        
        # Store categorical data as dictionary of arrays
        self.obs = {
            col: np.array(self.adata.obs[col]) 
            for col in self.cat_columns
        }
        
        # Store X matrix format
        self.is_sparse = scipy.sparse.issparse(self.adata.X)
    
    def get_gene_expression(self, gene_name):
        if gene_name in self.gene_cache:
            return self.gene_cache[gene_name]
            
        gene_idx = self.adata.var_names.get_loc(gene_name)
        
        if self.is_sparse:
            # Get the specific column efficiently from sparse matrix
            expr = self.adata.X[:, gene_idx].toarray().flatten()
        else:
            expr = self.adata.X[:, gene_idx]
            
        if len(self.gene_cache) >= self.cache_size:
            self.gene_cache.popitem(last=False)
        self.gene_cache[gene_name] = expr
        return expr

    def get_umap_df(self):
        # Create DataFrame only when needed
        return pd.DataFrame(
            self.umap,
            columns=['UMAP1', 'UMAP2'],
            index=self.obs_names
        )




def create_app(h5ad_path):
    manager = DataManager(h5ad_path)
    app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.YETI,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True
    )

    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Single-cell RNA-seq Explorer</title>
            {%favicon%}
            {%css%}
            <style>
                :root {
                    --primary-color: #4299E1;
                    --primary-dark: #2B6CB0;
                    --bg-color: #F7FAFC;
                    --text-color: #1A202C;
                    --border-color: #E2E8F0;
                    --hover-bg: #EBF8FF;
                }
                
                * { 
                    box-sizing: border-box; 
                    margin: 0;
                    padding: 0;
                }
                
                body { 
                    font-family: 'Inter', system-ui, -apple-system, sans-serif;
                    background-color: var(--bg-color);
                    color: var(--text-color);
                    line-height: 1.5;
                }
                
                .app-container {
                    max-width: 1440px;
                    margin: 0 auto;
                    padding: 2rem;
                }
                
                .app-header {
                    background: white;
                    padding: 1.5rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    border-radius: 8px;
                    margin-bottom: 2rem;
                }
                
                .app-title {
                    font-size: 2rem;
                    font-weight: 700;
                    color: var(--primary-dark);
                    letter-spacing: -0.025em;
                }
                
                .control-card {
                    background: white;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    transition: all 0.2s ease;
                }
                
                .control-card:hover {
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                }
                
                .control-label {
                    font-size: 0.875rem;
                    font-weight: 500;
                    color: #4A5568;
                    margin-bottom: 0.5rem;
                }
                
                .dash-dropdown {
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    transition: all 0.2s ease;
                }
                
                .dash-dropdown:hover {
                    border-color: var(--primary-color);
                }
                
                .dash-dropdown .Select-control {
                    border-radius: 6px;
                    border: none;
                    box-shadow: none;
                }
                
                .custom-button {
                    background: white;
                    border: 1px solid var(--border-color);
                    color: var(--text-color);
                    padding: 0.75rem 1rem;
                    border-radius: 6px;
                    font-weight: 500;
                    transition: all 0.2s ease;
                }
                
                .custom-button:hover:not(:disabled) {
                    background: var(--hover-bg);
                    border-color: var(--primary-color);
                    color: var(--primary-dark);
                }
                
                .custom-button:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }
                
                .plot-container {
                    background: white;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    overflow: hidden;
                }
                
                .plot-header {
                    background: var(--primary-color);
                    color: white;
                    padding: 1rem;
                    font-weight: 600;
                }
                
                .tab-container .nav-link {
                    color: var(--text-color);
                    font-weight: 500;
                    padding: 1rem;
                    border: none;
                    border-bottom: 2px solid transparent;
                    transition: all 0.2s ease;
                }
                
                .tab-container .nav-link.active {
                    color: var(--primary-color);
                    border-bottom: 2px solid var(--primary-color);
                }
                
                .loading-spinner {
                    border-color: var(--primary-color);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Single-cell RNA-seq Explorer", className="app-title text-center mb-0")
                ], className="app-header")
            ])
        ]),

        # Controls Row
        dbc.Row([
            # Color Selection
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Label("Color by", className="control-label"),
                        dcc.Dropdown(
                            id='color-select',
                            options=[
                                {'label': 'Gene Expression', 'value': 'gene'},
                                *[{'label': col, 'value': col} for col in manager.cat_columns]
                            ],
                            value='gene',
                            className="w-100",
                            clearable=False
                        )
                    ], className="p-3")
                ], className="control-card h-100")
            ], width=4),
            
            # Gene Selection
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id='gene-select-container', children=[
                            html.Label("Select Gene", className="control-label"),
                            dcc.Dropdown(
                                id='gene-select',
                                options=[{'label': g, 'value': g} for g in manager.genes],
                                value=manager.genes[0] if manager.genes else None,
                                placeholder="Select a gene",
                                className="w-100"
                            )
                        ])
                    ], className="p-3")
                ], className="control-card h-100")
            ], width=4),
            
            # Export Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Label("Export Options", className="control-label"),
                        dbc.ButtonGroup([
                            dbc.Button(
                                [html.I(className="bi bi-file-earmark-text me-2"), "Selected Cells"],
                                id='export-button',
                                className="custom-button mb-2 w-100",
                                disabled=True
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-diagram-3 me-2"), "Clicked Cluster"],
                                id='export-cluster-button',
                                className="custom-button mb-2 w-100",
                                disabled=True
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-file-earmark-binary me-2"), "H5AD"],
                                id='export-h5ad-button',
                                className="custom-button w-100",
                                disabled=True
                            )
                        ], vertical=True, className="w-100")
                    ], className="p-3")
                ], className="control-card h-100")
            ], width=4)
        ], className="mb-4 g-3"),

        # UMAP Plot
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("UMAP Visualization", className="plot-header"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-umap",
                            children=dcc.Graph(
                                id='umap-plot',
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToAdd': ['lasso2d'],
                                    'toImageButtonOptions': {'height': None, 'width': None}
                                },
                                style={'height': '70vh'}
                            ),
                            type="default",
                            className="loading-spinner"
                        )
                    ], className="p-0")
                ], className="plot-container")
            ], width=12)
        ], className="mb-4"),

        # QC Metrics and Cluster Info Tabs
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Tabs([
                        dbc.Tab(
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='qc-metrics',
                                        figure=create_qc_metrics_figure(manager.adata),
                                        config={
                                            'displayModeBar': True,
                                            'displaylogo': False,
                                            'toImageButtonOptions': {'height': None, 'width': None}
                                        }
                                    )
                                ], className="p-3")
                            ),
                            label="Quality Control Metrics",
                            tab_id="qc-tab",
                            className="custom-tab"
                        ),
                        dbc.Tab(
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div(id='click-data')
                                ], className="p-3")
                            ),
                            label="Cluster Information",
                            tab_id="cluster-tab",
                            className="custom-tab"
                        )
                    ], className="nav-fill tab-container")
                ], className="plot-container")
            ], width=12)
        ]),

        dcc.Download(id="download-selection"),
        dcc.Download(id="download-h5ad")
    ], fluid=True, className="app-container")

    @callback(
    Output('export-h5ad-button', 'disabled'),
    [Input('umap-plot', 'clickData'),
     Input('color-select', 'value')]
)
    def toggle_h5ad_export_button(clickData, color_by):
        return clickData is None or color_by == 'gene'

    @callback(
    Output("download-h5ad", "data"),
    Input("export-h5ad-button", "n_clicks"),
    [State('umap-plot', 'clickData'),
     State('color-select', 'value')],
    prevent_initial_call=True
    )
    def export_h5ad(n_clicks, click_data, color_by):
        if not click_data or color_by == 'gene':
            return None
            
        # Get clicked category
        point_idx = click_data['points'][0]['pointIndex']
        clicked_category = str(manager.obs[color_by][point_idx])
        
        # Create boolean mask for selected cells
        mask = np.array([str(x) == clicked_category for x in manager.obs[color_by]])
        
        # Get the subset of data
        subset_adata = manager.adata[mask]
        
        # Convert to memory mode
        subset_adata = subset_adata.to_memory()
        
        # Clean up the obs (metadata) to ensure compatibility
        for col in subset_adata.obs.columns:
            if subset_adata.obs[col].dtype.name == 'category':
                subset_adata.obs[col] = subset_adata.obs[col].astype(str)
            # Remove any complex dtypes
            if subset_adata.obs[col].dtype == 'object':
                subset_adata.obs[col] = subset_adata.obs[col].astype(str)
        
        # Create temporary file path with timestamp
        import time
        temp_path = f"selected_cluster_{clicked_category}_{int(time.time())}.h5ad"
        
        # Write with minimal compression
        subset_adata.write_h5ad(
            temp_path,
            compression='gzip',
            compression_opts=1  # Use minimal compression
        )
        
        # Add to temp files for cleanup
        manager.temp_files.add(temp_path)
        
        # Return the file for download
        return dcc.send_file(
            temp_path,
            filename=f"selected_cluster_{clicked_category}.h5ad"
        )




    
    @callback(
        Output('gene-select-container', 'style'),
        Input('color-select', 'value')
    )
    def toggle_gene_select(color_by):
        base_style = {'display': 'block'}
        if color_by == 'gene':
            return base_style
        return {**base_style, 'display': 'none'}


    @callback(
        Output('export-button', 'disabled'),
        Input('umap-plot', 'selectedData')
    )
    def toggle_export_button(selected_data):
        return selected_data is None

    @callback(
        Output('export-cluster-button', 'disabled'),
        [Input('umap-plot', 'clickData'),
         Input('color-select', 'value')]
    )
    def toggle_cluster_export_button(clickData, color_by):
        return clickData is None or color_by == 'gene'

    @callback(
    Output("download-selection", "data"),
    [Input("export-button", "n_clicks"),
     Input("export-cluster-button", "n_clicks")],
    [State('umap-plot', 'selectedData'),
     State('umap-plot', 'clickData'),
     State('color-select', 'value')],
    prevent_initial_call=True
        )
    def export_selection(n_clicks_select, n_clicks_cluster, selected_data, click_data, color_by):
        ctx = dash.callback_context
        if not ctx.triggered:
            return None
            
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'export-button' and selected_data:
            point_indices = [p['pointIndex'] for p in selected_data['points']]
            selected_cells = manager.obs_names[point_indices]
        elif button_id == 'export-cluster-button' and click_data and color_by != 'gene':
            point_idx = click_data['points'][0]['pointIndex']
            clicked_category = str(manager.obs[color_by][point_idx])
            mask = manager.obs[color_by].astype(str) == clicked_category
            selected_cells = manager.obs_names[mask]
        else:
            return None
            
        df_selected = pd.DataFrame({'cell_barcode': selected_cells})
        return dcc.send_data_frame(df_selected.to_csv, "selected_cells.csv", index=False)
    
    @callback(
    Output('click-data', 'children'),
    [Input('umap-plot', 'clickData'),
    Input('color-select', 'value')]
    )
    def display_click_data(clickData, color_by):
        if not clickData or color_by == 'gene':
            return ''
        
        # Get the clicked point index from customdata
        point_idx = clickData['points'][0]['customdata']
        clicked_category = str(manager.obs[color_by][point_idx])
        
        # Calculate overall cluster frequencies
        categories, counts = np.unique(manager.obs[color_by], return_counts=True)
        categories = categories.astype(str)
        
        # Create mask for clicked cluster
        cluster_mask = manager.obs[color_by].astype(str) == clicked_category
        
        # Calculate Sample_Type frequencies within the clicked cluster
        sample_types = manager.obs['Sample_Type'][cluster_mask]
        type_categories, type_counts = np.unique(sample_types, return_counts=True)
        
        # Calculate percentages
        total_in_cluster = type_counts.sum()
        type_percentages = (type_counts / total_in_cluster) * 100
        
        # Create subplot with two graphs
        fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=[f'Cell Counts by {color_by}', 
                                        f'Sample Type Distribution in Cluster {clicked_category}'])
        
        # First subplot - cluster distribution
        fig.add_trace(
            go.Bar(
                x=categories,
                y=counts,
                text=counts,
                textposition='auto',
                marker_color=['lightgrey' if str(cat) != clicked_category else '#4299E1' 
                            for cat in categories]
            ),
            row=1, col=1
        )
        
        # Second subplot - Sample Type distribution
        fig.add_trace(
            go.Bar(
                x=type_categories,
                y=type_percentages,
                text=[f'{p:.1f}%' for p in type_percentages],
                textposition='auto',
                marker_color='#4299E1'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Selected Cluster: {clicked_category}',
            showlegend=False,
            width=1200,
            height=400,
            margin=dict(t=50, l=50, r=50, b=100),
            plot_bgcolor='white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text=color_by, row=1, col=1)
        fig.update_xaxes(title_text='Sample Type', row=1, col=2)
        fig.update_yaxes(title_text='Number of Cells', row=1, col=1)
        fig.update_yaxes(title_text='Percentage in Cluster (%)', row=1, col=2)
        
        return dcc.Graph(figure=fig)







    @callback(
    Output('umap-plot', 'figure'),
    [Input('color-select', 'value'),
    Input('gene-select', 'value')],
    [State('umap-plot', 'figure')]
    )   
    def update_plot(color_by, gene, current_fig):
        umap_coords = manager.umap
        
        if color_by == 'gene' and gene is not None:
            color_values = manager.get_gene_expression(gene)
            colorscale = 'viridis'
            showscale = True
            
            fig = go.Figure(data=[go.Scattergl(
                x=umap_coords[:,0],
                y=umap_coords[:,1],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_values,
                    colorscale=colorscale,
                    showscale=showscale,
                    colorbar=dict(
                        title=dict(text=gene, side="right"),
                        thickness=15,
                        len=0.75
                    ),
                    opacity=0.7
                ),
                customdata=np.arange(len(umap_coords)),
                hovertemplate="<br>".join([
                    "Index: %{customdata}",
                    f"{gene}: %{{text}}<extra></extra>"
                ]),
                text=[f"{val:.2f}" for val in color_values]
            )])
            
        else:
            categories = np.unique(manager.obs[color_by])
            n_colors = len(categories)
            
            category_to_int = {cat: i for i, cat in enumerate(categories)}
            color_values = np.array([category_to_int[cat] for cat in manager.obs[color_by]])
            
            colors = (px.colors.qualitative.Bold + 
                    px.colors.qualitative.Dark24 + 
                    px.colors.qualitative.Alphabet)
            colors = colors[:n_colors]
            
            # Create separate traces for each category for better legend
            traces = []
            for i, cat in enumerate(categories):
                mask = manager.obs[color_by] == cat
                traces.append(
                    go.Scattergl(
                        x=umap_coords[mask,0],
                        y=umap_coords[mask,1],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=colors[i],
                            opacity=0.7
                        ),
                        name=str(cat),
                        customdata=np.arange(len(umap_coords))[mask],
                        hovertemplate="<br>".join([
                            "Index: %{customdata}",
                            f"{color_by}: {str(cat)}<extra></extra>"
                        ]),
                        showlegend=True,
                        legendgroup=str(cat)
                    )
                )
            
            fig = go.Figure(data=traces)

        if current_fig:
            fig.update_layout(
                xaxis=dict(range=current_fig['layout']['xaxis']['range']),
                yaxis=dict(range=current_fig['layout']['yaxis']['range'])
            )

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            width=None,
            height=800,
            margin=dict(
                t=50, 
                l=50, 
                r=100,  # Increased right margin for legend
                b=50
            ),
            xaxis=dict(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#E2E8F0',
                scaleanchor='y',
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#E2E8F0'
            ),
            legend=dict(
                title=dict(
                    text=color_by,
                    font=dict(
                        size=14,
                        family="Arial",
                        color="black"
                    )
                ),
                font=dict(
                    size=12,
                    family="Arial"
                ),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                itemsizing='constant',
                itemwidth=30,
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=1.15,  # Moved legend further right
                tracegroupgap=5  # Add space between legend items
            )
        )

        return fig





    return app

if __name__ == '__main__':

    app = create_app("please.final.h5ad")
    
    app.server.config['SERVER_NAME'] = None
    
    app.server.run(
        host='localhost',
        port=8050,
        debug=True
    )

