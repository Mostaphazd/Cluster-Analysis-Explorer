import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
import math
# Load environment variables from .env file
load_dotenv()
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    Birch,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
# Optional: UMAP for better 2D projections
import umap  # from umap-learn
import mplcursors
# Optional: PDF export for reports
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
# For sample datasets
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_blobs


class ClusterExplorerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Cluster Analysis Explorer")
        self.geometry("1400x800")

        # Apply modern theme
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Modern look

        # Configure colors
        self.style.configure('TButton', padding=6)
        self.style.configure('TLabelframe', padding=5)
        self.style.configure('TNotebook.Tab', padding=[10, 5])

        # Data-related attributes
        self.df = None              # original dataframe
        self.df_original = None     # backup before cleaning
        self.df_processed = None    # after dropping NA in selected columns
        self.selected_columns = []  # user-selected columns
        self.X = None               # preprocessed matrix used for clustering
        self.labels = None          # cluster labels
        self.numeric_cols = []      # selected numeric cols
        self.categorical_cols = []  # selected categorical cols
        self.scaler = None
        self.cleaning_log = []      # track cleaning operations

        # UI-related
        self.data_tree = None       # Treeview for clustered data table

        # Projection choice: PCA or UMAP
        self.projection_var = tk.StringVar(value="PCA")

        # Hover cursor for tooltips
        self.hover_cursor = None

        # Data view mode: show only clustering columns OR all CSV columns
        self.view_columns_var = tk.StringVar(value="cluster")

        # Whether to sort displayed rows by cluster label
        self.sort_by_cluster_var = tk.BooleanVar(value=True)

        # Show centroids on plot
        self.show_centroids_var = tk.BooleanVar(value=True)

        # Store model for centroid access
        self.current_model = None

        self._build_ui()

    # =========================== UI SETUP ===========================

    
    
    def show_about(self):
            win = tk.Toplevel(self)
            win.title("About Cluster Analysis Explorer")
            win.transient(self)
            win.grab_set()
            win.geometry("700x520")

            # ---- Outer frame with canvas + scrollbar ----
            outer = tk.Frame(win)
            outer.pack(fill=tk.BOTH, expand=True)

            canvas = tk.Canvas(outer, highlightthickness=0)
            vscroll = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
            canvas.configure(yscrollcommand=vscroll.set)

            vscroll.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Inner frame that will actually hold the content
            container = tk.Frame(canvas, padx=15, pady=15)
            canvas.create_window((0, 0), window=container, anchor="nw")

            def _on_configure(event):
                # Update scrollregion when the size of the inner frame changes
                canvas.configure(scrollregion=canvas.bbox("all"))

            container.bind("<Configure>", _on_configure)

            # Optional: enable mouse wheel scrolling inside the About window
            def _on_mousewheel(event):
                # On Windows / Mac, event.delta is used
                canvas.yview_scroll(-int(event.delta / 120), "units")

            canvas.bind_all("<MouseWheel>", _on_mousewheel)

            # ---- Content in the scrollable container ----

            title = tk.Label(
                container,
                text="Cluster Analysis Explorer",
                font=("Helvetica", 14, "bold")
            )
            title.pack(anchor="w")

            subtitle = tk.Label(
                container,
                text=(
                    "An interactive tool for exploring clustering algorithms, data analysis, validation metrics,\n"
                    "and the structure of your datasets using PCA/UMAP visualizations and rich reports."
                ),
                wraplength=640,
                justify="left"
            )
            subtitle.pack(anchor="w", pady=(4, 10))

            def section(header, text):
                hdr = tk.Label(container, text=header, font=("Helvetica", 11, "bold"))
                hdr.pack(anchor="w", pady=(8, 0))
                body = tk.Label(container, text=text, wraplength=640, justify="left")
                body.pack(anchor="w")

            section(
                "Top Bar",
                "• Load CSV: import your own dataset for clustering.\n"
                "• Sample datasets (when available): quickly load well-known datasets like Iris, Wine, "
                "Breast Cancer, or synthetic blobs.\n"
                "• The status label shows which file or sample is currently loaded."
            )

            section(
                "Left Panel – Main Workflow",
                "• 1. Select Columns: choose which columns to include. Numeric features are scaled; "
                "categorical features are one-hot encoded.\n"
                "• 1b. Data Cleaning: inspect missing values, remove duplicates, detect basic outliers, "
                "and reset to the original data if needed.\n"
                "• 2. Algorithm & Parameters: pick a clustering algorithm (KMeans, Hierarchical, "
                "DBSCAN, GMM, BIRCH, or custom code) and tune its parameters.\n"
                "• 2b. Projection & Visualization: choose PCA or UMAP for the 2D Cluster Map, and "
                "adjust projection settings.\n"
                "• 3a. Run Clustering: execute the chosen algorithm.\n"
                "• 3b. Elbow Method: analyze inertia and silhouette curves and suggest k using the "
                "Elbow on inertia.\n"
                "• 3c. Auto-Suggest: automatically test multiple algorithms and settings using the "
                "Silhouette score to recommend a configuration.\n"
                "• 4. Export & New Record: export the clustered data to CSV, or add a new record and "
                "see which cluster it joins with the current model.\n"
                "• 5. Reporting: save a PDF summary or generate an AI-enhanced narrative report."
            )

            section(
                "Right Panel – Tabs",
                "• EDA & Visualization: exploratory plots (histograms, box plots, scatter plots, "
                "correlation heatmaps) for understanding feature distributions.\n"
                "• Clustered Data: a table view of the processed data, with each row color-coded by "
                "cluster label and an explicit cluster column.\n"
                "• Cluster Map (PCA / UMAP): a 2D projection of the clustered data. Points are colored "
                "by cluster; hovering reveals original values, and optional labels can show key attributes.\n"
                "• Validation Metrics: internal validation scores (Silhouette, Davies–Bouldin, "
                "Calinski–Harabasz) and per-cluster counts.\n"
                "• Cluster Report: human-readable summaries of each cluster, comparing averages and "
                "dominant categories against the overall dataset."
            )

            section(
                "Developers",
                "This project was developed by:\n"
                "• Hasan Eid – hasan.eid@lau.edu\n"
                "• Christ Trad – christ.trad@lau.edu\n"
                "• Mustafa Zeinedeen – mostapha.zeineddin@lau.edu"
            )

            close_btn = ttk.Button(container, text="Close", command=win.destroy)
            close_btn.pack(anchor="e", pady=(12, 0))



    def _build_ui(self):
        # ---- Top frame: load CSV and sample datasets ----
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        load_btn = ttk.Button(top_frame, text="Load CSV", command=self.load_csv)
        load_btn.pack(side=tk.LEFT, padx=2)

        # Sample datasets dropdown
       

        self.file_label = tk.Label(top_frame, text="No file loaded", anchor="w", fg="gray")
        self.file_label.pack(side=tk.LEFT, padx=10)


  # Small (i) button to open the About window
        info_btn = ttk.Button(top_frame, text="ⓘ", width=3, command=self.show_about)
        info_btn.pack(side=tk.RIGHT, padx=5)

        # ---- Main frame (left controls + right notebook) ----
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)






        # Left panel for controls WITH vertical scrolling
        left_container = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=7)

        # Canvas + scrollbar + inner frame
        # Canvas + scrollbar + inner frame
        left_canvas = tk.Canvas(left_container, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_frame = tk.Frame(left_canvas)

        # Put the frame inside the canvas with a little margin
        left_canvas.create_window((10, 10), window=left_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        # Pack scrollbar and canvas
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        # Pack scrollbar and canvas
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Update scroll region whenever the inner frame resizes
        def _on_frame_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        left_frame.bind("<Configure>", _on_frame_configure)

        # ------------ Mouse wheel / trackpad scrolling ------------

        def _on_mousewheel(event):
            # Windows / macOS: <MouseWheel> with event.delta
            if event.delta:
                left_canvas.yview_scroll(int(-event.delta / 120), "units")

        def _on_linux_scroll(event):
            # Linux/X11: Button-4 = up, Button-5 = down
            if event.num == 4:
                left_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                left_canvas.yview_scroll(1, "units")

        def _bind_mousewheel(_event):
            left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            left_canvas.bind_all("<Button-4>", _on_linux_scroll)
            left_canvas.bind_all("<Button-5>", _on_linux_scroll)

        def _unbind_mousewheel(_event):
            left_canvas.unbind_all("<MouseWheel>")
            left_canvas.unbind_all("<Button-4>")
            left_canvas.unbind_all("<Button-5>")

        # Enable scrolling when mouse is over the left panel
        left_canvas.bind("<Enter>", _bind_mousewheel)
        left_canvas.bind("<Leave>", _unbind_mousewheel)



        # For some Linux / X11 systems (Button-4/5 events)
        left_canvas.bind_all("<Button-4>", lambda e: left_canvas.yview_scroll(-1, "units"))
        left_canvas.bind_all("<Button-5>", lambda e: left_canvas.yview_scroll(1, "units"))

        # Right panel for notebook (data, plot, metrics, report)
        right_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---- Column selection ----
        col_frame = tk.LabelFrame(left_frame, text="1. Select Columns")
        col_frame.pack(fill=tk.X, padx=5, pady=8)

        self.col_listbox = tk.Listbox(
            col_frame,
            selectmode=tk.MULTIPLE,
            height=10,
            exportselection=False
        )
        self.col_listbox.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        col_scroll = tk.Scrollbar(col_frame, orient=tk.VERTICAL, command=self.col_listbox.yview)
        col_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.col_listbox.config(yscrollcommand=col_scroll.set)

        apply_cols_btn = ttk.Button(left_frame, text="Use Selected Columns", command=self.set_selected_columns)
        apply_cols_btn.pack(fill=tk.X, padx=5, pady=8)

        # ---- Data Cleaning Section ----
        clean_frame = tk.LabelFrame(left_frame, text="1b. Data Cleaning")
        clean_frame.pack(fill=tk.X, padx=5, pady=8)

        ttk.Button(clean_frame, text="Handle Missing Values", command=self.handle_missing_values).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(clean_frame, text="Remove Duplicates", command=self.remove_duplicates).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(clean_frame, text="Detect Outliers", command=self.detect_outliers).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(clean_frame, text="View Data Info", command=self.show_data_info).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(clean_frame, text="Reset to Original", command=self.reset_to_original).pack(fill=tk.X, padx=5, pady=2)

        # ---- Algorithm settings ----
        algo_frame = tk.LabelFrame(left_frame, text="2. Algorithm & Parameters")
        algo_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(algo_frame, text="Algorithm:").grid(row=0, column=0, sticky="w")

        self.algo_var = tk.StringVar(value="kmeans")
        algo_options = [
            "kmeans",
            "minibatch_kmeans",
            "hierarchical",
            "gmm",
            "spectral",
            "birch",
            "dbscan",
            "custom",
        ]
        self.algo_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.algo_var,
            values=algo_options,
            state="readonly"
        )
        self.algo_combo.grid(row=0, column=1, sticky="we", padx=5, pady=2)
        self.algo_combo.bind("<<ComboboxSelected>>", self.on_algo_change)

        # n_clusters for KMeans / Hierarchical / GMM / Spectral / Birch
        tk.Label(algo_frame, text="n_clusters / k:").grid(row=1, column=0, sticky="w")
        self.n_clusters_var = tk.IntVar(value=3)
        self.n_clusters_spin = tk.Spinbox(
            algo_frame,
            from_=2,
            to=20,
            textvariable=self.n_clusters_var,
            width=5
        )
        self.n_clusters_spin.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # Hierarchical: linkage
        tk.Label(algo_frame, text="Linkage:").grid(row=2, column=0, sticky="w")
        

        # Hierarchical: linkage
        self.linkage_label = tk.Label(algo_frame, text="Linkage:")
        self.linkage_label.grid(row=2, column=0, sticky="w")
        self.linkage_var = tk.StringVar(value="ward")
        self.linkage_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.linkage_var,
            values=["ward", "complete", "average", "single"],
            state="readonly"
        )
        self.linkage_combo.grid(row=2, column=1, sticky="we", padx=5, pady=2)



        # DBSCAN: eps
        tk.Label(algo_frame, text="eps (DBSCAN):").grid(row=3, column=0, sticky="w")
        self.eps_var = tk.DoubleVar(value=0.5)
        self.eps_entry = tk.Entry(algo_frame, textvariable=self.eps_var, width=7)
        self.eps_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        # DBSCAN: min_samples
        tk.Label(algo_frame, text="min_samples:").grid(row=4, column=0, sticky="w")
        self.min_samples_var = tk.IntVar(value=5)
        self.min_samples_entry = tk.Entry(algo_frame, textvariable=self.min_samples_var, width=7)
        self.min_samples_entry.grid(row=4, column=1, sticky="w", padx=5, pady=2)

        # Custom code button
        self.custom_code_btn = ttk.Button(algo_frame, text="Edit Custom Code", command=self.open_custom_code_editor)
        self.custom_code_btn.grid(row=5, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        # Initial visibility based on default algo
        self.update_algo_widgets_visibility()

        # Store custom code
        self.custom_code = '''# Custom clustering code
# Available variables:
#   X - preprocessed data matrix (numpy array)
#   n_samples - number of samples
#
# You must set:
#   labels - cluster labels (numpy array of integers)
#
# Example using sklearn's MeanShift:
from sklearn.cluster import MeanShift

model = MeanShift()
labels = model.fit_predict(X)
'''

        # ---- Run button ----
        run_btn = ttk.Button(left_frame, text="3. Run Clustering", command=self.run_clustering)
        run_btn.pack(fill=tk.X, padx=5, pady=10)

        # ---- Elbow method button ----
        elbow_btn = ttk.Button(left_frame, text="3b. Elbow Method (Find Optimal k)", command=self.show_elbow_method)
        elbow_btn.pack(fill=tk.X, padx=5, pady=5)

        # ---- Auto-suggest button ----
        suggest_btn = ttk.Button(
            left_frame,
            text="3c. Auto-Suggest Algorithm / Params",
            command=self.auto_suggest
        )
        suggest_btn.pack(fill=tk.X, padx=5, pady=5)

        # ---- Export CSV button ----
        self.export_btn = ttk.Button(left_frame, text="4. Export Clustered CSV", command=self.export_clustered_csv)
        self.export_btn.pack(fill=tk.X, padx=5, pady=5)

        # ---- Add New Record button ----
        self.add_record_btn = ttk.Button(left_frame, text="5. Add New Record", command=self.add_new_record)
        self.add_record_btn.pack(fill=tk.X, padx=5, pady=5)

        # ---- Save report as PDF ----
        self.export_pdf_btn = ttk.Button(left_frame, text="6. Save Report as PDF", command=self.save_report_as_pdf)
        self.export_pdf_btn.pack(fill=tk.X, padx=5, pady=5)

        # ---- Right side: Notebook ----
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # EDA tab (first tab - data exploration)
        self.eda_frame = tk.Frame(self.notebook)
        self.notebook.add(self.eda_frame, text="EDA & Visualization")

        # Clustered Data tab (table view)
        self.data_frame = tk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Clustered Data")

        # Cluster Map tab (PCA/UMAP)
        self.plot_frame = tk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Cluster Map (PCA / UMAP)")

        # Metrics tab
        self.metrics_frame = tk.Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Validation Metrics")

        # Report tab
        self.report_frame = tk.Frame(self.notebook)
        self.notebook.add(self.report_frame, text="Cluster Report")

        self._build_eda_tab()
        self._build_data_tab()
        self._build_plot_tab()
        self._build_metrics_tab()
        self._build_report_tab()

    def _build_eda_tab(self):
        # Controls at top
        controls_frame = tk.Frame(self.eda_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(controls_frame, text="Visualization Type:").pack(side=tk.LEFT, padx=5)
        self.eda_plot_var = tk.StringVar(value="Histograms")
        eda_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.eda_plot_var,
            values=["Histograms", "Box Plots", "Correlation Matrix", "Scatter Matrix", "Distribution Summary"],
            state="readonly",
            width=20
        )
        eda_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(controls_frame, text="Generate Plot", command=self.generate_eda_plot).pack(side=tk.LEFT, padx=10)

        # Figure for EDA plots
        self.eda_fig = Figure(figsize=(8, 6))
        self.eda_canvas = FigureCanvasTkAgg(self.eda_fig, master=self.eda_frame)
        self.eda_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar for EDA
        toolbar_frame = tk.Frame(self.eda_frame)
        toolbar_frame.pack(fill=tk.X)
        self.eda_toolbar = NavigationToolbar2Tk(self.eda_canvas, toolbar_frame)

    def _build_data_tab(self):
        # Top controls (view mode + sort)
        controls_frame = tk.Frame(self.data_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(controls_frame, text="Data view:").pack(side=tk.LEFT)

        tk.Radiobutton(
            controls_frame,
            text="Clustering columns only",
            variable=self.view_columns_var,
            value="cluster",
            command=self.update_data_view,
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            controls_frame,
            text="All CSV columns",
            variable=self.view_columns_var,
            value="all",
            command=self.update_data_view,
        ).pack(side=tk.LEFT, padx=5)

        tk.Checkbutton(
            controls_frame,
            text="Sort by cluster label",
            variable=self.sort_by_cluster_var,
            command=self.update_data_view,
        ).pack(side=tk.LEFT, padx=5)

        # Treeview for showing clustered data
        self.data_tree = ttk.Treeview(self.data_frame, show="headings")
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbars
        y_scroll = tk.Scrollbar(self.data_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=y_scroll.set)

        x_scroll = tk.Scrollbar(self.data_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_tree.configure(xscrollcommand=x_scroll.set)

    def _build_plot_tab(self):
        # Controls at the top of the Cluster Map tab
        controls_frame = tk.Frame(self.plot_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(controls_frame, text="Projection:").pack(side=tk.LEFT, padx=5, pady=5)
        self.projection_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.projection_var,
            values=["PCA", "UMAP"],
            state="readonly",
            width=8
        )
        self.projection_combo.pack(side=tk.LEFT, padx=5, pady=5)
        self.projection_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        ttk.Checkbutton(
            controls_frame,
            text="Show Centroids",
            variable=self.show_centroids_var,
            command=self.update_plot
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(controls_frame, text="(Hover over points to see row details)").pack(
            side=tk.LEFT, padx=10, pady=5
        )

        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Clusters (2D projection)")
        self.ax.set_xlabel("Dim 1")
        self.ax.set_ylabel("Dim 2")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_metrics_tab(self):
        # Create a paned window for text and plots
        metrics_pane = ttk.PanedWindow(self.metrics_frame, orient=tk.HORIZONTAL)
        metrics_pane.pack(fill=tk.BOTH, expand=True)

        # Left side: text metrics
        text_frame = tk.Frame(metrics_pane)
        metrics_pane.add(text_frame, weight=1)

        self.metrics_text = tk.Text(text_frame, wrap="word", width=50)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        metrics_scroll = ttk.Scrollbar(text_frame, command=self.metrics_text.yview)
        metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.metrics_text.config(yscrollcommand=metrics_scroll.set)

        # Right side: metric visualizations
        plot_frame = tk.Frame(metrics_pane)
        metrics_pane.add(plot_frame, weight=2)

        # Controls for metric plots
        plot_controls = tk.Frame(plot_frame)
        plot_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(plot_controls, text="Visualization:").pack(side=tk.LEFT, padx=5)
        self.metrics_plot_var = tk.StringVar(value="Silhouette Plot")
        metrics_combo = ttk.Combobox(
            plot_controls,
            textvariable=self.metrics_plot_var,
            values=["Silhouette Plot", "Cluster Sizes", "Metrics Summary", "Silhouette by Cluster"],
            state="readonly",
            width=18
        )
        metrics_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_controls, text="Generate", command=self.generate_metrics_plot).pack(side=tk.LEFT, padx=5)

        # Figure for metrics plots
        self.metrics_fig = Figure(figsize=(6, 4))
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=plot_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = tk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.metrics_toolbar = NavigationToolbar2Tk(self.metrics_canvas, toolbar_frame)

    def _build_report_tab(self):
        # Controls at top
        controls_frame = tk.Frame(self.report_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Generate AI-Enhanced Report", command=self.generate_ai_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export Professional PDF", command=self.export_professional_pdf).pack(side=tk.LEFT, padx=5)

        # Status label for API
        self.api_status_label = tk.Label(controls_frame, text="", fg="gray")
        self.api_status_label.pack(side=tk.RIGHT, padx=10)
        self.check_api_status()

        # Report text area
        self.report_text = tk.Text(self.report_frame, wrap="word", font=("Consolas", 10))
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar for report
        report_scroll = ttk.Scrollbar(self.report_frame, command=self.report_text.yview)
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_text.config(yscrollcommand=report_scroll.set)

    def check_api_status(self):
        """Check if Gemini API key is configured."""
        api_key = 'AIzaSyBH71LLI0IYxMckKedDWJOnr8fzZGONTvc' #os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_api_key_here':
            self.api_status_label.config(text="✓ Gemini API configured", fg="green")
            return True
        else:
            self.api_status_label.config(text="⚠ No API key (using template report)", fg="orange")
            return False

    # ========================= DATA HANDLING =========================






    def load_csv(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            self.df = pd.read_csv(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
            return

        self.file_label.config(text=filepath.split("/")[-1], fg="black")
        self.populate_column_list()

        # Backup original and reset
        self.df_original = self.df.copy()
        self.cleaning_log = []
        self.selected_columns = []
        self.df_processed = None
        self.X = None
        self.labels = None
        self.clear_data_view()
        self.clear_plot()
        self.clear_metrics()
        self.clear_report()

  
    # ========================= DATA CLEANING =========================

    def handle_missing_values(self):
        """Open dialog to handle missing values."""
        if self.df is None:
            messagebox.showwarning("No data", "Load data first.")
            return

        missing = self.df.isnull().sum()
        total_missing = missing.sum()

        if total_missing == 0:
            messagebox.showinfo("No Missing Values", "The dataset has no missing values.")
            return

        # Create dialog
        win = tk.Toplevel(self)
        win.title("Handle Missing Values")
        win.geometry("500x400")
        win.grab_set()

        # Show missing values info
        info_frame = tk.LabelFrame(win, text="Missing Values Summary")
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        info_text = tk.Text(info_frame, height=8, width=50)
        info_text.pack(padx=5, pady=5)

        info_text.insert(tk.END, f"Total missing values: {total_missing}\n\n")
        for col in self.df.columns:
            if missing[col] > 0:
                pct = 100 * missing[col] / len(self.df)
                info_text.insert(tk.END, f"{col}: {missing[col]} ({pct:.1f}%)\n")
        info_text.config(state=tk.DISABLED)

        # Options
        options_frame = tk.LabelFrame(win, text="Select Action")
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        action_var = tk.StringVar(value="drop_rows")

        ttk.Radiobutton(options_frame, text="Drop rows with missing values", variable=action_var, value="drop_rows").pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(options_frame, text="Fill numeric with mean", variable=action_var, value="fill_mean").pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(options_frame, text="Fill numeric with median", variable=action_var, value="fill_median").pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(options_frame, text="Fill with 0", variable=action_var, value="fill_zero").pack(anchor="w", padx=10, pady=2)

        def apply_action():
            action = action_var.get()
            before_count = len(self.df)

            if action == "drop_rows":
                self.df = self.df.dropna()
                msg = f"Dropped {before_count - len(self.df)} rows with missing values."
            elif action == "fill_mean":
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
                msg = "Filled numeric missing values with column means."
            elif action == "fill_median":
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
                msg = "Filled numeric missing values with column medians."
            elif action == "fill_zero":
                self.df = self.df.fillna(0)
                msg = "Filled all missing values with 0."

            self.cleaning_log.append(msg)
            self.populate_column_list()
            win.destroy()
            messagebox.showinfo("Cleaning Applied", f"{msg}\n\nRows remaining: {len(self.df)}")

        ttk.Button(win, text="Apply", command=apply_action).pack(pady=10)

    def remove_duplicates(self):
        """Remove duplicate rows."""
        if self.df is None:
            messagebox.showwarning("No data", "Load data first.")
            return

        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)

        if removed > 0:
            msg = f"Removed {removed} duplicate rows."
            self.cleaning_log.append(msg)
            self.populate_column_list()
            messagebox.showinfo("Duplicates Removed", f"{msg}\n\nRows remaining: {len(self.df)}")
        else:
            messagebox.showinfo("No Duplicates", "No duplicate rows found.")

    def detect_outliers(self):
        """Detect and optionally remove outliers using IQR method."""
        if self.df is None:
            messagebox.showwarning("No data", "Load data first.")
            return

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            messagebox.showwarning("No Numeric Data", "No numeric columns found for outlier detection.")
            return

        # Calculate outliers using IQR
        outlier_info = {}
        total_outliers = 0

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if outliers > 0:
                outlier_info[col] = outliers
                total_outliers += outliers

        # Show dialog
        win = tk.Toplevel(self)
        win.title("Outlier Detection (IQR Method)")
        win.geometry("450x350")
        win.grab_set()

        info_text = tk.Text(win, height=12, width=50)
        info_text.pack(padx=10, pady=10)

        if total_outliers == 0:
            info_text.insert(tk.END, "No outliers detected using IQR method.")
        else:
            info_text.insert(tk.END, f"Outliers detected (IQR method):\n\n")
            for col, count in outlier_info.items():
                pct = 100 * count / len(self.df)
                info_text.insert(tk.END, f"{col}: {count} outliers ({pct:.1f}%)\n")

        info_text.config(state=tk.DISABLED)

        def remove_outliers():
            rows_before = len(self.df)
            mask = pd.Series([True] * len(self.df))
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = mask & (self.df[col] >= lower) & (self.df[col] <= upper)
            self.df = self.df[mask]
            removed = rows_before - len(self.df)
            msg = f"Removed {removed} rows containing outliers."
            self.cleaning_log.append(msg)
            self.populate_column_list()
            win.destroy()
            messagebox.showinfo("Outliers Removed", f"{msg}\n\nRows remaining: {len(self.df)}")

        if total_outliers > 0:
            ttk.Button(win, text="Remove Outlier Rows", command=remove_outliers).pack(pady=5)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=5)

    def show_data_info(self):
        """Show comprehensive data information."""
        if self.df is None:
            messagebox.showwarning("No data", "Load data first.")
            return

        win = tk.Toplevel(self)
        win.title("Data Information")
        win.geometry("600x500")

        text = tk.Text(win, wrap="word")
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(win, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

        # Basic info
        text.insert(tk.END, "=" * 50 + "\n")
        text.insert(tk.END, "DATASET OVERVIEW\n")
        text.insert(tk.END, "=" * 50 + "\n\n")
        text.insert(tk.END, f"Total Rows: {len(self.df)}\n")
        text.insert(tk.END, f"Total Columns: {len(self.df.columns)}\n")
        text.insert(tk.END, f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB\n\n")

        # Column types
        text.insert(tk.END, "COLUMN TYPES:\n")
        text.insert(tk.END, "-" * 30 + "\n")
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        text.insert(tk.END, f"Numeric ({len(numeric)}): {', '.join(numeric) if numeric else 'None'}\n")
        text.insert(tk.END, f"Categorical ({len(categorical)}): {', '.join(categorical) if categorical else 'None'}\n\n")

        # Missing values
        missing = self.df.isnull().sum().sum()
        text.insert(tk.END, f"Total Missing Values: {missing}\n\n")

        # Numeric summary
        if numeric:
            text.insert(tk.END, "NUMERIC STATISTICS:\n")
            text.insert(tk.END, "-" * 30 + "\n")
            desc = self.df[numeric].describe().T
            text.insert(tk.END, desc.to_string() + "\n\n")

        # Cleaning log
        if self.cleaning_log:
            text.insert(tk.END, "CLEANING OPERATIONS PERFORMED:\n")
            text.insert(tk.END, "-" * 30 + "\n")
            for log in self.cleaning_log:
                text.insert(tk.END, f"• {log}\n")

        text.config(state=tk.DISABLED)

    def reset_to_original(self):
        """Reset data to original state before cleaning."""
        if self.df_original is None:
            messagebox.showwarning("No Original Data", "No original data to reset to.")
            return

        if messagebox.askyesno("Confirm Reset", "Reset data to original state? All cleaning will be undone."):
            self.df = self.df_original.copy()
            self.cleaning_log = []
            self.populate_column_list()
            messagebox.showinfo("Reset Complete", f"Data reset to original.\n\nRows: {len(self.df)}")

    # ========================= EDA VISUALIZATIONS =========================

    def generate_eda_plot(self):
        """Generate EDA visualization based on selection."""
        if self.df is None:
            messagebox.showwarning("No data", "Load data first.")
            return

        plot_type = self.eda_plot_var.get()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            messagebox.showwarning("No Numeric Data", "No numeric columns for visualization.")
            return

        self.eda_fig.clear()

        try:
            if plot_type == "Histograms":
                n_cols = len(numeric_cols)
                n_rows = (n_cols + 2) // 3
                for i, col in enumerate(numeric_cols, 1):
                    ax = self.eda_fig.add_subplot(n_rows, 3, i)
                    ax.hist(self.df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
                    ax.set_title(col, fontsize=8)
                    ax.tick_params(labelsize=6)
                self.eda_fig.suptitle("Histograms of Numeric Features", fontsize=10)

            elif plot_type == "Box Plots":
                n_cols = len(numeric_cols)
                n_rows = (n_cols + 2) // 3
                for i, col in enumerate(numeric_cols, 1):
                    ax = self.eda_fig.add_subplot(n_rows, 3, i)
                    ax.boxplot(self.df[col].dropna())
                    ax.set_title(col, fontsize=8)
                    ax.tick_params(labelsize=6)
                self.eda_fig.suptitle("Box Plots of Numeric Features", fontsize=10)

            elif plot_type == "Correlation Matrix":
                ax = self.eda_fig.add_subplot(111)
                corr = self.df[numeric_cols].corr()
                im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(range(len(numeric_cols)))
                ax.set_yticks(range(len(numeric_cols)))
                ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=7)
                ax.set_yticklabels(numeric_cols, fontsize=7)
                self.eda_fig.colorbar(im, ax=ax)
                ax.set_title("Correlation Matrix", fontsize=10)

            elif plot_type == "Scatter Matrix":
                # Use max 5 columns for readability
                cols_to_use = numeric_cols[:5]
                n = len(cols_to_use)
                for i in range(n):
                    for j in range(n):
                        ax = self.eda_fig.add_subplot(n, n, i * n + j + 1)
                        if i == j:
                            ax.hist(self.df[cols_to_use[i]].dropna(), bins=15, alpha=0.7)
                        else:
                            ax.scatter(self.df[cols_to_use[j]], self.df[cols_to_use[i]], alpha=0.3, s=5)
                        if i == n - 1:
                            ax.set_xlabel(cols_to_use[j], fontsize=6)
                        if j == 0:
                            ax.set_ylabel(cols_to_use[i], fontsize=6)
                        ax.tick_params(labelsize=5)
                self.eda_fig.suptitle("Scatter Matrix (first 5 numeric features)", fontsize=10)

            elif plot_type == "Distribution Summary":
                ax = self.eda_fig.add_subplot(111)
                data_to_plot = [self.df[col].dropna().values for col in numeric_cols]
                ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
                ax.set_xticks(range(1, len(numeric_cols) + 1))
                ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=7)
                ax.set_title("Distribution Summary (Violin Plots)", fontsize=10)

            self.eda_fig.tight_layout()
            self.eda_canvas.draw()

        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate plot:\n{e}")

    def populate_column_list(self):
        self.col_listbox.delete(0, tk.END)
        if self.df is None:
            return
        for col in self.df.columns:
            self.col_listbox.insert(tk.END, col)

    def set_selected_columns(self):
        if self.df is None:
            messagebox.showwarning("No data", "Please load a CSV first.")
            return

        indices = self.col_listbox.curselection()
        if not indices:
            messagebox.showwarning("No columns", "Please select at least one column.")
            return

        self.selected_columns = [self.df.columns[i] for i in indices]

        # Classify numeric vs categorical within selection
        self.numeric_cols = []
        self.categorical_cols = []

        for col in self.selected_columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

        msg = (
            "Selected columns:\n"
            f"  • Numeric: {self.numeric_cols or 'None'}\n"
            f"  • Categorical: {self.categorical_cols or 'None'}"
        )
        messagebox.showinfo("Columns set", msg)

        # Reset downstream
        self.df_processed = None
        self.X = None
        self.labels = None
        self.clear_data_view()
        self.clear_plot()
        self.clear_metrics()
        self.clear_report()

    def preprocess_data(self):
        if self.df is None or not self.selected_columns:
            messagebox.showwarning("Missing info", "Load data and select columns first.")
            return False

        # Drop rows with NA in selected columns
        df_sel = self.df[self.selected_columns].dropna().copy()
        if df_sel.empty:
            messagebox.showwarning("No data", "All rows have missing values in selected columns.")
            return False

        # Normalize categorical columns: strip spaces, unify casing
        for col in self.categorical_cols:
            if col in df_sel.columns:
                df_sel[col] = (
                    df_sel[col]
                    .astype(str)
                    .str.strip()
                    .str.title()
                )

        # Keep processed df for reporting and data view
        self.df_processed = df_sel

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df_sel, columns=self.categorical_cols, drop_first=False)

        # Identify numeric columns in the encoded frame
        all_cols = df_encoded.columns.tolist()
        numeric_encoded_cols = []
        for col in self.numeric_cols:
            if col in all_cols:
                numeric_encoded_cols.append(col)

        # Scale numeric columns to [0, 1]
        self.scaler = MinMaxScaler()
        if numeric_encoded_cols:
            df_encoded[numeric_encoded_cols] = self.scaler.fit_transform(df_encoded[numeric_encoded_cols])

        self.X = df_encoded.values
        return True

    # ====================== ALGO / CLUSTERING =======================

    def on_algo_change(self, event=None):
        self.update_algo_widgets_visibility()

    
    def update_algo_widgets_visibility(self):
        algo = self.algo_var.get()

        algo_uses_k = ["kmeans", "minibatch_kmeans", "hierarchical", "gmm", "spectral", "birch"]

        # n_clusters spinbox
        if algo in algo_uses_k:
            self.n_clusters_spin.config(state="normal")
        else:
            self.n_clusters_spin.config(state="disabled")

        # Linkage controls: only visible for hierarchical
        if algo == "hierarchical":
            try:
                self.linkage_label.grid()
                self.linkage_combo.grid()
            except Exception:
                pass
            self.linkage_combo.config(state="readonly")
        else:
            # Hide linkage row completely for non-hierarchical algorithms
            try:
                self.linkage_label.grid_remove()
                self.linkage_combo.grid_remove()
            except Exception:
                pass

        # DBSCAN parameters
        if algo == "dbscan":
            self.eps_entry.config(state="normal")
            self.min_samples_entry.config(state="normal")
        else:
            self.eps_entry.config(state="disabled")
            self.min_samples_entry.config(state="disabled")

        # Show/hide custom code button
        if algo == "custom":
            self.custom_code_btn.grid()
        else:
            self.custom_code_btn.grid_remove()



    def open_custom_code_editor(self):
        """Open a window to edit custom clustering code."""
        win = tk.Toplevel(self)
        win.title("Custom Clustering Code Editor")
        win.geometry("700x500")
        win.grab_set()

        # Instructions
        instructions = tk.Label(
            win,
            text="Write your custom clustering code below. Use 'X' as input data and set 'labels' as output.",
            wraplength=680,
            justify="left"
        )
        instructions.pack(padx=10, pady=5, anchor="w")

        # Code editor
        editor_frame = tk.Frame(win)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.code_editor = tk.Text(editor_frame, wrap="none", font=("Consolas", 10))
        self.code_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbars
        y_scroll = ttk.Scrollbar(editor_frame, command=self.code_editor.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.code_editor.config(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=self.code_editor.xview)
        x_scroll.pack(fill=tk.X, padx=10)
        self.code_editor.config(xscrollcommand=x_scroll.set)

        # Insert current code
        self.code_editor.insert("1.0", self.custom_code)

        # Buttons
        btn_frame = tk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        def save_code():
            self.custom_code = self.code_editor.get("1.0", tk.END)
            win.destroy()
            messagebox.showinfo("Saved", "Custom code saved. Click 'Run Clustering' to execute.")

        def load_example(example_name):
            examples = {
                "MeanShift": '''from sklearn.cluster import MeanShift

model = MeanShift()
labels = model.fit_predict(X)''',
                "OPTICS": '''from sklearn.cluster import OPTICS

model = OPTICS(min_samples=5, xi=0.05)
labels = model.fit_predict(X)''',
                "Affinity Propagation": '''from sklearn.cluster import AffinityPropagation

model = AffinityPropagation(random_state=0)
labels = model.fit_predict(X)''',
                "Agglomerative (Custom)": '''from sklearn.cluster import AgglomerativeClustering

# You can customize linkage, distance threshold, etc.
model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=1.5,
    linkage='average'
)
labels = model.fit_predict(X)'''
            }
            if example_name in examples:
                self.code_editor.delete("1.0", tk.END)
                self.code_editor.insert("1.0", examples[example_name])

        ttk.Button(btn_frame, text="Save & Close", command=save_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=5)

        # Example dropdown
        ttk.Label(btn_frame, text="Load Example:").pack(side=tk.LEFT, padx=(20, 5))
        example_var = tk.StringVar(value="Select...")
        example_combo = ttk.Combobox(
            btn_frame,
            textvariable=example_var,
            values=["MeanShift", "OPTICS", "Affinity Propagation", "Agglomerative (Custom)"],
            state="readonly",
            width=20
        )
        example_combo.pack(side=tk.LEFT)
        example_combo.bind("<<ComboboxSelected>>", lambda e: load_example(example_var.get()))


    def run_clustering(self):
        if not self.preprocess_data():
            return

        algo = self.algo_var.get()

        try:
            if algo == "kmeans":
                n_clusters = int(self.n_clusters_var.get())
                model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
                self.labels = model.fit_predict(self.X)

            elif algo == "minibatch_kmeans":
                from sklearn.cluster import MiniBatchKMeans
                n_clusters = int(self.n_clusters_var.get())
                model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
                self.labels = model.fit_predict(self.X)

            elif algo == "hierarchical":
                n_clusters = int(self.n_clusters_var.get())
                linkage = self.linkage_var.get()
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                self.labels = model.fit_predict(self.X)

            elif algo == "gmm":
                n_clusters = int(self.n_clusters_var.get())
                model = GaussianMixture(n_components=n_clusters, random_state=0)
                model.fit(self.X)
                self.labels = model.predict(self.X)

            elif algo == "spectral":
                n_clusters = int(self.n_clusters_var.get())
                model = SpectralClustering(
                    n_clusters=n_clusters,
                    assign_labels="kmeans",
                    affinity="nearest_neighbors",
                    random_state=0,
                )
                self.labels = model.fit_predict(self.X)

            elif algo == "birch":
                n_clusters = int(self.n_clusters_var.get())
                model = Birch(n_clusters=n_clusters)
                self.labels = model.fit_predict(self.X)

            elif algo == "dbscan":
                eps = float(self.eps_var.get())
                min_samples = int(self.min_samples_var.get())
                model = DBSCAN(eps=eps, min_samples=min_samples)
                self.labels = model.fit_predict(self.X)

            elif algo == "custom":
                # Execute custom code
                X = self.X
                n_samples = len(X)

                # Create execution namespace
                exec_namespace = {
                    'X': X,
                    'n_samples': n_samples,
                    'np': np,
                    'pd': pd,
                }

                try:
                    exec(self.custom_code, exec_namespace)

                    if 'labels' not in exec_namespace:
                        messagebox.showerror("Error", "Custom code must set 'labels' variable.")
                        return

                    self.labels = np.array(exec_namespace['labels'])

                    # Try to get model if defined
                    if 'model' in exec_namespace:
                        model = exec_namespace['model']
                    else:
                        model = None

                except Exception as e:
                    messagebox.showerror("Custom Code Error", f"Error executing custom code:\n{e}")
                    return

            else:
                messagebox.showerror("Error", "Unknown algorithm selected.")
                return

            # Store model for centroid access





 # 🔹 Normalize labels so cluster IDs are clean integers (0,1,2,...,-1)
            try:
                self.labels = np.asarray(self.labels)

                # If labels are floats like 0.0, 1.0, 2.0 → convert to ints
                if np.issubdtype(self.labels.dtype, np.floating):
                    if np.all(np.isfinite(self.labels)):
                        self.labels = np.rint(self.labels).astype(int)

                # If labels are a weird shape, flatten them
                if self.labels.ndim > 1:
                    self.labels = self.labels.reshape(-1)
            except Exception:
                # If anything weird happens, leave labels as-is
                pass





            self.current_model = model

        except Exception as e:
            messagebox.showerror("Error", f"Clustering failed:\n{e}")
            return

        self.update_data_view()
        self.update_plot()
        self.update_metrics()
        self.update_report()

    def auto_suggest(self):
        """Try several algorithms/params and suggest the configuration with the best Silhouette score."""
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV file first.")
            return
        if not self.selected_columns:
            messagebox.showwarning("No columns", "Select columns for clustering first.")
            return

        if not self.preprocess_data():
            return

        X = self.X
        n_samples = len(X)
        if n_samples < 5:
            messagebox.showwarning(
                "Too few samples",
                "Need at least 5 samples for a meaningful suggestion."
            )
            return

        results = []

        def evaluate_config(algo_name, params, fit_predict_fn):
            try:
                labels = fit_predict_fn()
                labels = np.array(labels)
                unique = np.unique(labels)
                if len(unique) < 2:
                    return
                counts = [np.sum(labels == lbl) for lbl in unique if lbl != -1]
                if any(c < 2 for c in counts):
                    return
                score = silhouette_score(X, labels)
                results.append({
                    "algo": algo_name,
                    "params": params,
                    "score": score,
                    "labels": labels,
                })
            except Exception:
                # If any config fails, just skip it
                pass

        max_k = min(8, n_samples - 1)

        # Try KMeans, Hierarchical (ward), GMM for k=2..max_k
        for k in range(2, max_k + 1):
            # KMeans
            def fp_kmeans(k=k):
                model = KMeans(n_clusters=k, random_state=0, n_init="auto")
                return model.fit_predict(X)
            evaluate_config(
                "kmeans",
                {"n_clusters": k},
                fp_kmeans
            )

            # Hierarchical (ward)
            def fp_hier(k=k):
                model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                return model.fit_predict(X)
            evaluate_config(
                "hierarchical",
                {"n_clusters": k, "linkage": "ward"},
                fp_hier
            )

            # GMM
            def fp_gmm(k=k):
                model = GaussianMixture(n_components=k, random_state=0)
                return model.fit_predict(X)
            evaluate_config(
                "gmm",
                {"n_clusters": k},
                fp_gmm
            )

        # Try a few DBSCAN settings
        eps_values = [0.2, 0.4, 0.6, 0.8]
        min_samples_values = [3, 5]

        for eps in eps_values:
            for ms in min_samples_values:
                def fp_db(eps=eps, ms=ms):
                    model = DBSCAN(eps=eps, min_samples=ms)
                    return model.fit_predict(X)
                evaluate_config(
                    "dbscan",
                    {"eps": eps, "min_samples": ms},
                    fp_db
                )

        if not results:
            messagebox.showwarning(
                "No suggestion",
                "Unable to find a good clustering configuration. Try adjusting parameters manually."
            )
            return

        # Sort by Silhouette descending
        results.sort(key=lambda r: r["score"], reverse=True)
        best = results[0]

        # Update UI controls to suggested configuration
        self.algo_var.set(best["algo"])
        self.on_algo_change()

        if best["algo"] in ["kmeans", "hierarchical", "gmm"]:
            self.n_clusters_var.set(best["params"]["n_clusters"])
            if best["algo"] == "hierarchical":
                self.linkage_var.set(best["params"].get("linkage", "ward"))
        elif best["algo"] == "dbscan":
            self.eps_var.set(best["params"]["eps"])
            self.min_samples_var.set(best["params"]["min_samples"])

        # Show a small summary of top 3
        top = results[:3]
        lines = ["Auto-suggested configuration based on Silhouette score:\n"]
        lines.append(
            f"Best: {best['algo']}  |  params={best['params']}  |  Silhouette={best['score']:.3f}"
        )
        if len(top) > 1:
            lines.append("\nOther strong candidates:")
            for cand in top[1:]:
                lines.append(
                    f"- {cand['algo']}  |  params={cand['params']}  |  Silhouette={cand['score']:.3f}"
                )

        messagebox.showinfo("Suggestion", "\n".join(lines))

        # Finally, run clustering with the suggested settings
        self.run_clustering()

    def show_elbow_method(self):
        """Show elbow method plot to help find optimal k."""
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV file first.")
            return
        if not self.selected_columns:
            messagebox.showwarning("No columns", "Select columns for clustering first.")
            return

        if not self.preprocess_data():
            return

        X = self.X
        n_samples = len(X)

        if n_samples < 3:
            messagebox.showwarning("Too few samples", "Need at least 3 samples for elbow method.")
            return

        max_k = min(10, n_samples - 1)
        k_range = range(2, max_k + 1)

        inertias = []
        silhouettes = []

        # Calculate metrics for each k
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)

                # Silhouette score
                if len(np.unique(labels)) >= 2:
                    sil = silhouette_score(X, labels)
                    silhouettes.append(sil)
                else:
                    silhouettes.append(0)
            except Exception:
                inertias.append(None)
                silhouettes.append(None)

        # Create popup window with elbow plot
        win = tk.Toplevel(self)
        win.title("Elbow Method - Find Optimal k")
        win.geometry("800x600")

        fig = Figure(figsize=(8, 5))

    

 
        valid_inertias = [i for i in inertias if i is not None]
        # ----- Compute elbow k from inertia using relative improvement rule -----
        valid_pairs = [
            (k, i) for k, i in zip(k_range, inertias) if i is not None
        ]
        elbow_k_holder = {"k": None}

        if len(valid_pairs) >= 3:
            ks = [p[0] for p in valid_pairs]
            vals = [p[1] for p in valid_pairs]

            # improvement from k -> k_next
            rel_improvements = []
            for idx in range(len(vals) - 1):
                i_curr = vals[idx]
                i_next = vals[idx + 1]
                if i_curr <= 0:
                    rel = 0.0
                else:
                    rel = (i_curr - i_next) / i_curr
                rel_improvements.append(rel)

            # threshold for "small improvement" (tune if you like)
            threshold = 0.35

            # find first k where improvement drops below threshold
            elbow_k = None
            for idx, rel in enumerate(rel_improvements):
                # this k is the "last big improvement"
                if rel < threshold:
                    elbow_k = ks[idx]
                    break

            # if nothing below threshold, fall back to max k
            if elbow_k is None:
                elbow_k = ks[-1]

            elbow_k_holder["k"] = elbow_k

            # reuse ks/vals later for plotting
            valid_k = ks
            valid_inertias = vals
        else:
            valid_k = [k for k, i in zip(k_range, inertias) if i is not None]
            valid_inertias = [i for i in inertias if i is not None]



            # Subplot 1: Inertia (Elbow)
        fig = Figure(figsize=(8, 5))

        # Subplot 1: Inertia (Elbow)
        ax1 = fig.add_subplot(211)
        ax1.plot(valid_k, valid_inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax1.set_title('Elbow Method - Look for the "elbow" point')
        ax1.grid(True, alpha=0.3)

        # Mark elbow k if we found one
        if elbow_k_holder["k"] is not None:
            elbow_k = elbow_k_holder["k"]
            ax1.axvline(x=elbow_k, color='r', linestyle='--', alpha=0.7)
            ax1.annotate(
                f'Elbow k={elbow_k}',
                xy=(elbow_k, np.interp(elbow_k, valid_k, valid_inertias)),
                xytext=(elbow_k + 0.5, valid_inertias[0]),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red')
            )
        # Subplot 2: Silhouette scores
        ax2 = fig.add_subplot(212)
        valid_sil = [s for s in silhouettes if s is not None]
        ax2.plot(valid_k, valid_sil, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score - Higher is better')
        ax2.grid(True, alpha=0.3) 


        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        # Add button to apply best k

         # Add button to apply best k (based on elbow / inertia)
        def apply_best_k():
            elbow_k = elbow_k_holder["k"]
            if elbow_k is None:
                messagebox.showwarning(
                    "No elbow found",
                    "Could not determine an elbow point from the inertia curve."
                )
                return
            self.n_clusters_var.set(elbow_k)
            win.destroy()
            messagebox.showinfo(
                "Applied",
                f"Set n_clusters to {elbow_k} based on the inertia elbow.\nClick 'Run Clustering' to apply."
            )
        btn_frame = tk.Frame(win)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Apply Best k", command=apply_best_k).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=10)

    # ========================= EXPORT & ADD RECORD =========================

    def export_clustered_csv(self):
        """Export df_processed with cluster labels as a CSV."""
        if self.df_processed is None or self.labels is None:
            messagebox.showwarning(
                "No clustering",
                "Please run clustering before exporting."
            )
            return

        df = self.df_processed.copy()
        labels = np.array(self.labels)

        if len(df) != len(labels):
            messagebox.showerror(
                "Error",
                "Size mismatch between data and labels; cannot export."
            )
            return

        df["cluster"] = labels

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            df.to_csv(filepath, index=False)
            messagebox.showinfo("Export successful", f"Clustered data saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")

    def add_new_record(self):
        """Open a form to add a new row, then re-run clustering with same settings."""
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV before adding a record.")
            return
        if not self.selected_columns:
            messagebox.showwarning("No columns selected", "Select columns for clustering first.")
            return
        if self.labels is None:
            messagebox.showwarning(
                "No clustering yet",
                "Run clustering once before adding a new record.\n"
                "The same algorithm and parameters will be reused."
            )
            return

        win = tk.Toplevel(self)
        win.title("Add New Record")
        win.grab_set()

        info_label = tk.Label(
            win,
            text="Enter values for the new record.\n"
                 "Columns used for clustering (*) should be filled.",
            justify="left"
        )
        info_label.pack(side=tk.TOP, anchor="w", padx=10, pady=10)

        form_frame = tk.Frame(win)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(form_frame)
        scrollbar = tk.Scrollbar(form_frame, orient=tk.VERTICAL, command=canvas.yview)
        inner_frame = tk.Frame(canvas)

        inner_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        entry_widgets = {}
        for i, col in enumerate(self.df.columns):
            lbl_text = col
            if col in self.selected_columns:
                lbl_text += " *"
            tk.Label(inner_frame, text=lbl_text).grid(row=i, column=0, sticky="w", padx=5, pady=3)

            ent = tk.Entry(inner_frame, width=25)
            ent.grid(row=i, column=1, sticky="we", padx=5, pady=3)
            entry_widgets[col] = ent

        btn_frame = tk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        def on_cancel():
            win.destroy()

        def on_submit():
            new_row = {}
            for col, ent in entry_widgets.items():
                val = ent.get().strip()
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    if val == "":
                        new_row[col] = np.nan
                    else:
                        try:
                            new_row[col] = float(val)
                        except ValueError:
                            messagebox.showerror(
                                "Invalid value",
                                f"Column '{col}' expects a numeric value."
                            )
                            return
                else:
                    if val == "":
                        new_row[col] = None
                    else:
                        val_raw = val.strip()
                        existing = (
                            self.df[col]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .unique()
                        )
                        matches = [e for e in existing if e.lower() == val_raw.lower()]

                        if len(matches) == 1:
                            new_row[col] = matches[0]
                        else:
                            new_row[col] = val_raw.title()

            # Ensure all selected clustering columns have values
            for col in self.selected_columns:
                v = new_row.get(col, None)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    messagebox.showerror(
                        "Missing value",
                        f"Column '{col}' is used for clustering and must be filled."
                    )
                    return

            try:
                self.df = pd.concat(
                    [self.df, pd.DataFrame([new_row])],
                    ignore_index=True
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add new record:\n{e}")
                return

            win.destroy()
            self.run_clustering()

        tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        tk.Button(btn_frame, text="Add Record & Recluster", command=on_submit).pack(side=tk.RIGHT, padx=5)

    def save_report_as_pdf(self):
        """Export the Cluster Report text area as a PDF."""
        content = self.report_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning(
                "Empty report",
                "The Cluster Report is empty.\nRun clustering first to generate a report."
            )
            return

        if pdf_canvas is None:
            messagebox.showerror(
                "PDF export not available",
                "The 'reportlab' package is required to export PDF.\n\n"
                "Install it with:\n    pip install reportlab"
            )
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            c = pdf_canvas.Canvas(filepath, pagesize=letter)
            width, height = letter
            margin = 72  # 1 inch
            y = height - margin
            max_chars = 100
            line_height = 14

            for line in content.splitlines():
                while len(line) > max_chars:
                    part = line[:max_chars]
                    line = line[max_chars:]
                    c.drawString(margin, y, part)
                    y -= line_height
                    if y < margin:
                        c.showPage()
                        y = height - margin
                c.drawString(margin, y, line)
                y -= line_height
                if y < margin:
                    c.showPage()
                    y = height - margin

            c.save()
            messagebox.showinfo("PDF saved", f"Cluster report saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PDF:\n{e}")

    # ========================= VISUALIZATION =========================

    def clear_data_view(self):
        if self.data_tree is not None:
            self.data_tree.delete(*self.data_tree.get_children())
            self.data_tree["columns"] = ()

   
   

    def update_data_view(self):
        """Show clustered data table, with either clustering columns only or all CSV columns,
        optionally sorted by cluster label, and always with colored rows.
        """
        self.clear_data_view()

        if self.df_processed is None or self.labels is None:
            return

        # ---- 1. Get labels as clean Python ints ----
        labels_arr = np.asarray(self.labels)
        labels_int = []
        for x in labels_arr:
            try:
                # convert 0.0 -> 0, 1.0 -> 1, etc.
                xi = int(round(float(x)))
            except Exception:
                xi = 0
            labels_int.append(xi)

        labels_int = np.array(labels_int, dtype=int)

        # Safety check: rows used for clustering must match labels
        if len(self.df_processed) != len(labels_int):
            messagebox.showwarning(
                "Warning",
                "Size mismatch between data and labels; cannot show data table."
            )
            return

        # ---- 2. Decide which columns (features) to show ----
        if self.view_columns_var.get() == "all" and self.df is not None:
            try:
                base_df = self.df.loc[self.df_processed.index].copy()
            except Exception:
                base_df = self.df_processed.copy()
        else:
            base_df = self.df_processed.copy()

        feature_cols = list(base_df.columns)

        # ---- 3. Define Treeview columns: features + __cluster__ ----
        all_cols = feature_cols + ["__cluster__"]
        self.data_tree["columns"] = all_cols

        for col in all_cols:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120, anchor="center")

        # ---- 4. Define colors per cluster label ----
        unique_labels = sorted(set(labels_int.tolist()))
        colors = [
            "#FFCCCC", "#CCFFCC", "#CCCCFF", "#FFF2CC", "#EAD1DC",
            "#D9EAD3", "#CFE2F3", "#F4CCCC", "#D0E0E3", "#FCE5CD"
        ]

        for i, lbl in enumerate(unique_labels):
            tag_name = f"cluster_{lbl}"
            if lbl == -1:
                bg = "#EEEEEE"  # grey for noise
            else:
                bg = colors[i % len(colors)]
            self.data_tree.tag_configure(tag_name, background=bg)

        # ---- 5. Build row order (optionally sorted by cluster) ----
        indices = list(range(len(base_df)))
        if getattr(self, "sort_by_cluster_var", None) is not None and self.sort_by_cluster_var.get():
            indices = sorted(indices, key=lambda idx: labels_int[idx])

        # ---- 6. Insert rows with values + color tags ----
        for idx in indices:
            row = base_df.iloc[idx]
            lbl = labels_int[idx]

            # Features first, then cluster as last column
            values = [row[col] for col in feature_cols] + [int(lbl)]
            tag_name = f"cluster_{lbl}"
            self.data_tree.insert("", "end", values=values, tags=(tag_name,))







    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title("Clusters (2D projection)")
        self.ax.set_xlabel("Dim 1")
        self.ax.set_ylabel("Dim 2")
        self.canvas.draw_idle()

    def update_plot(self):
        if self.X is None or self.labels is None:
            return

        proj = self.projection_var.get()

        try:
            if proj == "UMAP":
                if umap is None:
                    messagebox.showerror(
                        "UMAP not available",
                        "UMAP requires the 'umap-learn' package.\n\n"
                        "Install it with:\n    pip install umap-learn"
                    )
                    return
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                X_2d = reducer.fit_transform(self.X)
                title = "Clusters (UMAP 2D projection)"
                xlab, ylab = "UMAP-1", "UMAP-2"
            else:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(self.X)
                title = "Clusters (PCA 2D projection)"
                xlab, ylab = "PC1", "PC2"
        except Exception as e:
            messagebox.showerror("Error", f"{proj} for plotting failed:\n{e}")
            return

        self.ax.clear()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlab)
        self.ax.set_ylabel(ylab)

        labels = np.array(self.labels)
        unique_labels = np.unique(labels)

        colors = [
            "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
            "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
        ]

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            label_name = f"Cluster {lbl}" if lbl != -1 else "Noise (-1)"
            color = colors[i % len(colors)]
            indices = np.where(mask)[0]

            sc = self.ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                label=label_name,
                alpha=0.7,
                s=30,
                edgecolors="none",
                c=color
            )
            sc._indices = indices  # for hover



        self.ax.legend(loc="best", fontsize=8)

        # Show centroids in the 2D projection (PCA/UMAP) if enabled
        if self.show_centroids_var.get():
            centroids_2d = []
            labels_arr = np.array(self.labels)

            for lbl in np.unique(labels_arr):
                # Skip noise label for DBSCAN
                if lbl == -1:
                    continue
                mask = labels_arr == lbl
                if np.sum(mask) == 0:
                    continue

                # Compute centroid directly in projected space
                cx = X_2d[mask, 0].mean()
                cy = X_2d[mask, 1].mean()
                centroids_2d.append((lbl, cx, cy))

            # Draw centroids
            for lbl, cx, cy in centroids_2d:
                self.ax.scatter(
                    cx,
                    cy,
                    marker="X",
                    s=80,
                    edgecolors="white",
                    linewidths=1.0,
                    c="black",
                    zorder=5,
                )


        if mplcursors is not None and self.df_processed is not None:
            if len(self.df_processed) == len(X_2d):
                label_cols = self.categorical_cols or self.selected_columns
                if label_cols:
                    hover_texts = [
                        ", ".join(str(self.df_processed.iloc[i][c]) for c in label_cols)
                        for i in range(len(self.df_processed))
                    ]

                    if self.hover_cursor is not None:
                        try:
                            self.hover_cursor.remove()
                        except Exception:
                            pass
                        self.hover_cursor = None

                    def on_add(sel):
                        artist = sel.artist
                        if not hasattr(artist, "_indices"):
                            return
                        idx_global = artist._indices[sel.index]
                        if 0 <= idx_global < len(hover_texts):
                            sel.annotation.set_text(hover_texts[idx_global])
                        else:
                            sel.annotation.set_text("")

                    self.hover_cursor = mplcursors.cursor(self.ax.collections, hover=True)
                    self.hover_cursor.connect("add", on_add)

        self.canvas.draw_idle()

    # ====================== METRICS & REPORT =========================

    def clear_metrics(self):
        self.metrics_text.delete("1.0", tk.END)
        if hasattr(self, 'metrics_fig'):
            self.metrics_fig.clear()
            self.metrics_canvas.draw()

    def generate_metrics_plot(self):
        """Generate metric visualization based on selection."""
        if self.X is None or self.labels is None:
            messagebox.showwarning("No clustering", "Run clustering first.")
            return

        plot_type = self.metrics_plot_var.get()
        labels = np.array(self.labels)
        unique_labels = np.unique(labels)

        self.metrics_fig.clear()

        try:
            if plot_type == "Silhouette Plot":
                # Silhouette plot per sample
                if len(unique_labels) < 2:
                    messagebox.showwarning("Cannot plot", "Need at least 2 clusters for silhouette plot.")
                    return

                sample_silhouette_values = silhouette_samples(self.X, labels)
                avg_score = silhouette_score(self.X, labels)

                ax = self.metrics_fig.add_subplot(111)
                y_lower = 10

                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                for i, lbl in enumerate(sorted(unique_labels)):
                    if lbl == -1:
                        continue
                    cluster_silhouette_values = sample_silhouette_values[labels == lbl]
                    cluster_silhouette_values.sort()

                    size_cluster = len(cluster_silhouette_values)
                    y_upper = y_lower + size_cluster

                    ax.fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        cluster_silhouette_values,
                        facecolor=colors[i],
                        edgecolor=colors[i],
                        alpha=0.7
                    )
                    ax.text(-0.05, y_lower + 0.5 * size_cluster, str(lbl), fontsize=9)
                    y_lower = y_upper + 10

                ax.axvline(x=avg_score, color="red", linestyle="--", label=f'Avg: {avg_score:.3f}')
                ax.set_xlabel("Silhouette Coefficient")
                ax.set_ylabel("Cluster")
                ax.set_title("Silhouette Plot per Sample")
                ax.legend(loc='best')
                ax.set_yticks([])

            elif plot_type == "Cluster Sizes":
                # Bar chart of cluster sizes
                ax = self.metrics_fig.add_subplot(111)

                cluster_sizes = []
                cluster_names = []
                for lbl in sorted(unique_labels):
                    size = np.sum(labels == lbl)
                    cluster_sizes.append(size)
                    name = f"Cluster {lbl}" if lbl != -1 else "Noise"
                    cluster_names.append(name)

                colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_sizes)))
                bars = ax.bar(cluster_names, cluster_sizes, color=colors, edgecolor='black')

                # Add value labels on bars
                for bar, size in zip(bars, cluster_sizes):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           str(size), ha='center', va='bottom', fontsize=9)

                ax.set_xlabel("Cluster")
                ax.set_ylabel("Number of Samples")
                ax.set_title("Cluster Size Distribution")
                ax.tick_params(axis='x', rotation=45)

            elif plot_type == "Metrics Summary":
                # Bar chart comparing metrics
                ax = self.metrics_fig.add_subplot(111)

                metrics = {}
                metric_names = []
                metric_values = []
                metric_colors = []

                # Calculate metrics
                if len(unique_labels) >= 2:
                    try:
                        sil = silhouette_score(self.X, labels)
                        metric_names.append("Silhouette\n(0-1, higher=better)")
                        metric_values.append(sil)
                        metric_colors.append('green' if sil > 0.5 else 'orange' if sil > 0.25 else 'red')
                    except:
                        pass

                    try:
                        dbi = davies_bouldin_score(self.X, labels)
                        # Normalize DBI for display (invert so higher = better)
                        dbi_norm = max(0, 1 - dbi/3)  # rough normalization
                        metric_names.append("Davies-Bouldin\n(inverted, higher=better)")
                        metric_values.append(dbi_norm)
                        metric_colors.append('green' if dbi < 0.7 else 'orange' if dbi < 1.5 else 'red')
                    except:
                        pass

                    try:
                        ch = calinski_harabasz_score(self.X, labels)
                        # Normalize CH for display
                        ch_norm = min(1, ch / 500)  # rough normalization
                        metric_names.append("Calinski-Harabasz\n(normalized, higher=better)")
                        metric_values.append(ch_norm)
                        metric_colors.append('green' if ch > 300 else 'orange' if ch > 100 else 'red')
                    except:
                        pass

                if metric_values:
                    bars = ax.bar(metric_names, metric_values, color=metric_colors, edgecolor='black', alpha=0.7)
                    ax.set_ylabel("Score (normalized)")
                    ax.set_title("Metrics Summary (Green=Good, Orange=Fair, Red=Poor)")
                    ax.set_ylim(0, 1.2)

                    for bar, val in zip(bars, metric_values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(0.5, 0.5, "Cannot calculate metrics\n(need ≥2 clusters)",
                           ha='center', va='center', transform=ax.transAxes)

            elif plot_type == "Silhouette by Cluster":
                # Box plot of silhouette scores by cluster
                if len(unique_labels) < 2:
                    messagebox.showwarning("Cannot plot", "Need at least 2 clusters.")
                    return

                sample_silhouette_values = silhouette_samples(self.X, labels)
                ax = self.metrics_fig.add_subplot(111)

                cluster_sil_data = []
                cluster_names = []
                for lbl in sorted(unique_labels):
                    if lbl == -1:
                        continue
                    cluster_sil = sample_silhouette_values[labels == lbl]
                    cluster_sil_data.append(cluster_sil)
                    cluster_names.append(f"Cluster {lbl}")

                bp = ax.boxplot(cluster_sil_data, labels=cluster_names, patch_artist=True)

                colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_sil_data)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                avg_score = silhouette_score(self.X, labels)
                ax.axhline(y=avg_score, color='red', linestyle='--', label=f'Avg: {avg_score:.3f}')

                ax.set_xlabel("Cluster")
                ax.set_ylabel("Silhouette Score")
                ax.set_title("Silhouette Score Distribution by Cluster")
                ax.legend(loc='best')
                ax.tick_params(axis='x', rotation=45)

            self.metrics_fig.tight_layout()
            self.metrics_canvas.draw()

        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate metrics plot:\n{e}")

    def update_metrics(self):
        self.clear_metrics()
        if self.X is None or self.labels is None:
            return

        labels = np.array(self.labels)
        unique_labels = np.unique(labels)

        algo = self.algo_var.get()
        total_samples = len(self.X)

        non_noise_labels = [lbl for lbl in unique_labels if lbl != -1]
        n_clusters_non_noise = len(non_noise_labels)
        n_noise = int(np.sum(labels == -1))

        algo_desc_map = {
            "kmeans": "Centroid-based (spherical clusters, sensitive to scale/outliers).",
            "minibatch_kmeans": "Faster KMeans variant for larger datasets.",
            "hierarchical": "Agglomerative hierarchical clustering (tree-like structure).",
            "gmm": "Probabilistic (Gaussian mixture), ellipsoidal clusters, soft assignments.",
            "spectral": "Graph-based clustering, can capture complex/curved shapes.",
            "birch": "Incremental clustering, suited for larger datasets.",
            "dbscan": "Density-based; can find arbitrary-shaped clusters and noise.",
        }

        text_lines = []

        # Overview
        text_lines.append("=============== CLUSTERING OVERVIEW ===============")
        text_lines.append(f"Algorithm           : {algo}")
        text_lines.append(f"Algorithm type      : {algo_desc_map.get(algo, 'N/A')}")
        if algo in ["kmeans", "minibatch_kmeans", "hierarchical", "gmm", "spectral", "birch"]:
            text_lines.append(f"Requested n_clusters: {self.n_clusters_var.get()}")
        elif algo == "dbscan":
            text_lines.append(f"eps, min_samples    : {self.eps_var.get()}, {self.min_samples_var.get()}")
        text_lines.append(f"Samples used        : {total_samples}")
        text_lines.append(f"Distinct labels     : {len(unique_labels)} (including noise if present)")
        if algo == "dbscan":
            text_lines.append(f"Non-noise clusters  : {n_clusters_non_noise}")
            text_lines.append(f"Noise points (label -1): {n_noise}")
        text_lines.append("===================================================\n")

        # Metric values
        sil_value = None
        dbi_value = None
        ch_value = None

        valid_for_silhouette = True
        if len(unique_labels) < 2:
            valid_for_silhouette = False
        else:
            counts = [np.sum(labels == lbl) for lbl in unique_labels if lbl != -1]
            if any(c < 2 for c in counts):
                valid_for_silhouette = False

        text_lines.append("----------- INTERNAL VALIDATION METRICS -----------")
        if valid_for_silhouette:
            try:
                sil_value = silhouette_score(self.X, labels)
                text_lines.append(f"Silhouette Score     : {sil_value:.4f}  (range: -1 to 1, higher is better)")
            except Exception as e:
                text_lines.append(f"Silhouette Score     : error calculating ({e})")
        else:
            text_lines.append("Silhouette Score     : N/A (need ≥2 clusters with ≥2 samples each)")

        if len(unique_labels) >= 2:
            try:
                dbi_value = davies_bouldin_score(self.X, labels)
                text_lines.append(f"Davies–Bouldin Index: {dbi_value:.4f}  (lower is better)")
            except Exception as e:
                text_lines.append(f"Davies–Bouldin Index: error calculating ({e})")

            try:
                ch_value = calinski_harabasz_score(self.X, labels)
                text_lines.append(f"Calinski–Harabasz   : {ch_value:.4f}  (higher is better)")
            except Exception as e:
                text_lines.append(f"Calinski–Harabasz   : error calculating ({e})")
        else:
            text_lines.append("Davies–Bouldin Index: N/A (need ≥2 clusters)")
            text_lines.append("Calinski–Harabasz   : N/A (need ≥2 clusters)")

        text_lines.append("---------------------------------------------------\n")

        # Interpretations
        text_lines.append("Metric interpretation (rule-of-thumb):")
        if sil_value is not None:
            if sil_value >= 0.60:
                qual = "strong, well-separated clusters"
            elif sil_value >= 0.40:
                qual = "clear structure with some overlap"
            elif sil_value >= 0.25:
                qual = "weak structure; clusters overlap substantially"
            else:
                qual = "very weak / no clear structure"
            text_lines.append(f"• Silhouette ({sil_value:.3f}): {qual}.")

        if dbi_value is not None:
            if dbi_value <= 0.70:
                qual = "excellent separation"
            elif dbi_value <= 1.00:
                qual = "good separation"
            elif dbi_value <= 1.50:
                qual = "moderate separation"
            else:
                qual = "poor separation (clusters not very distinct)"
            text_lines.append(f"• Davies–Bouldin ({dbi_value:.3f}): {qual}.")

        if ch_value is not None:
            text_lines.append(
                f"• Calinski–Harabasz ({ch_value:.1f}): higher is better; "
                "compare across different runs or k values."
            )

        # Cluster sizes
        text_lines.append("\n--------------- CLUSTER SIZE SUMMARY ---------------")
        for lbl in sorted(unique_labels):
            count = np.sum(labels == lbl)
            pct = 100.0 * count / total_samples if total_samples > 0 else 0
            name = f"Cluster {lbl}" if lbl != -1 else "Noise (-1)"
            text_lines.append(f"{name:<15}: {count:>4} samples ({pct:5.1f}%)")
        text_lines.append("----------------------------------------------------\n")

        # Simple heuristic suggestions
        text_lines.append("Suggested next steps:")
        if sil_value is None and dbi_value is None and ch_value is None:
            text_lines.append(
                "• Current solution has too few usable clusters; try a different algorithm or parameters."
            )
        else:
            if sil_value is not None and sil_value < 0.25:
                text_lines.append(
                    "• Clusters look weak; try changing the number of clusters or switching algorithm."
                )
            elif sil_value is not None and sil_value < 0.4:
                text_lines.append(
                    "• Structure exists but is not very strong; experiment with nearby k values or another algorithm."
                )

            if algo in ["kmeans", "minibatch_kmeans", "hierarchical", "gmm"]:
                text_lines.append(
                    "• For centroid/model-based methods, try k-1 and k+1 to see if the scores improve."
                )

            if algo == "dbscan":
                noise_ratio = n_noise / total_samples if total_samples > 0 else 0
                if noise_ratio > 0.3:
                    text_lines.append(
                        "• DBSCAN produced many noise points; consider increasing eps or decreasing min_samples."
                    )
                else:
                    text_lines.append(
                        "• You can fine-tune eps and min_samples to control how many points are marked as noise."
                    )

        text_lines.append(
            "\nTip: Use these metrics comparatively. Try different algorithms / parameters "
            "and prefer solutions with higher Silhouette and Calinski–Harabasz, and lower Davies–Bouldin."
        )

        self.metrics_text.insert(tk.END, "\n".join(text_lines))

    def clear_report(self):
        self.report_text.delete("1.0", tk.END)

    def update_report(self):
        self.clear_report()
        if self.df_processed is None or self.labels is None:
            return

        df = self.df_processed.copy()
        labels = np.array(self.labels)
        if len(df) != len(labels):
            self.report_text.insert(tk.END, "Size mismatch between data and labels; cannot build report.\n")
            return

        df["__cluster__"] = labels
        total = len(df)

        numeric_cols = [c for c in self.selected_columns if c in self.numeric_cols]
        categorical_cols = [c for c in self.selected_columns if c in self.categorical_cols]

        overall_stats = {}
        if numeric_cols:
            overall_stats["numeric_mean"] = df[numeric_cols].mean()

        unique_labels = sorted(df["__cluster__"].unique())
        non_noise_labels = [lbl for lbl in unique_labels if lbl != -1]
        n_clusters_non_noise = len(non_noise_labels)
        n_noise = int(np.sum(labels == -1))

        report_lines = []

        report_lines.append("================= CLUSTERING SUMMARY =================")
        report_lines.append(f"Algorithm            : {self.algo_var.get()}")
        if self.algo_var.get() in ["kmeans", "minibatch_kmeans", "hierarchical", "gmm", "spectral", "birch"]:
            report_lines.append(f"Requested n_clusters : {self.n_clusters_var.get()}")
        elif self.algo_var.get() == "dbscan":
            report_lines.append(
                f"DBSCAN params        : eps={self.eps_var.get()}, min_samples={self.min_samples_var.get()}"
            )

        report_lines.append(f"Rows used            : {total}")
        report_lines.append(
            f"Selected columns     : {', '.join(self.selected_columns)} "
            f"({len(self.selected_columns)} features)"
        )
        report_lines.append(f"Numeric features     : {numeric_cols or 'None'}")
        report_lines.append(f"Categorical features : {categorical_cols or 'None'}")
        report_lines.append(f"Non-noise clusters   : {n_clusters_non_noise}")
        if n_noise > 0:
            report_lines.append(f"Noise points (label -1): {n_noise}")
        report_lines.append("======================================================\n")

        # Per-cluster analysis
        for lbl in unique_labels:
            cluster_df = df[df["__cluster__"] == lbl]
            size = len(cluster_df)
            pct = 100.0 * size / total if total > 0 else 0

            cluster_name = f"Cluster {lbl}" if lbl != -1 else "Noise Cluster (-1)"
            report_lines.append("------------------------------------------------------")
            report_lines.append(f"{cluster_name}   |   {size} samples ({pct:.1f}% of data)")
            report_lines.append("------------------------------------------------------")

            # Numeric features summary
            if numeric_cols and size > 0:
                cluster_mean = cluster_df[numeric_cols].mean()
                cluster_min = cluster_df[numeric_cols].min()
                cluster_max = cluster_df[numeric_cols].max()
                report_lines.append("Numeric profile (cluster vs overall):")
                for col in numeric_cols:
                    cm = cluster_mean[col]
                    ov = overall_stats["numeric_mean"][col]
                    rel = "≈ overall"
                    if cm > ov * 1.1:
                        rel = "higher than overall"
                    elif cm < ov * 0.9:
                        rel = "lower than overall"
                    report_lines.append(
                        f"  • {col}: mean {cm:.3f} "
                        f"(overall {ov:.3f}, {rel}), "
                        f"range [{cluster_min[col]:.3f} – {cluster_max[col]:.3f}]"
                    )
            else:
                report_lines.append("Numeric profile: no numeric features in this cluster.")

            # Categorical features summary
            if categorical_cols and size > 0:
                report_lines.append("\nCategorical profile (dominant values):")
                for col in categorical_cols:
                    value_counts = cluster_df[col].value_counts(normalize=True)
                    if value_counts.empty:
                        report_lines.append(f"  • {col}: no data")
                        continue
                    top_val = value_counts.index[0]
                    top_pct = value_counts.iloc[0] * 100
                    n_distinct = len(value_counts)
                    variety = (
                        f"{n_distinct} distinct values"
                        if n_distinct > 1 else "single dominant value"
                    )
                    report_lines.append(
                        f"  • {col}: mostly '{top_val}' ({top_pct:.1f}%), {variety}"
                    )
            else:
                report_lines.append("\nCategorical profile: no categorical features in this cluster.")

            # Quick characterization
            if categorical_cols or numeric_cols:
                desc_parts = []
                if categorical_cols:
                    for col in categorical_cols:
                        vc = cluster_df[col].value_counts(normalize=True)
                        if not vc.empty:
                            top_val = vc.index[0]
                            top_pct = vc.iloc[0] * 100
                            desc_parts.append(f"{top_val} ({col}, {top_pct:.0f}%)")
                if numeric_cols:
                    for col in numeric_cols:
                        cm = cluster_df[col].mean()
                        desc_parts.append(f"{col}≈{cm:.1f}")
                if desc_parts:
                    report_lines.append("\nCluster characterization:")
                    report_lines.append("  " + "; ".join(desc_parts))
                else:
                    report_lines.append("\nCluster characterization: no strong distinguishing features.")
            report_lines.append("\n")

        report_lines.append("=============== HOW TO READ THIS REPORT ===============")
        report_lines.append(
            "- Start with the global summary to understand how many rows/features "
            "were used and which algorithm/parameters were applied."
        )
        report_lines.append(
            "- For each cluster, focus on numeric means vs overall and the dominant "
            "categorical values: this tells you which type of records it groups."
        )
        report_lines.append(
            "- Compare clusters to see which groups differ most on key variables "
            "(e.g., much higher values or clearly different category distributions)."
        )
        report_lines.append(
            "- Noise cluster (-1) in DBSCAN typically contains outliers or points "
            "that do not clearly belong to any dense region."
        )
        report_lines.append("=======================================================")

        self.report_text.insert(tk.END, "\n".join(report_lines))

    def generate_ai_report(self):
        """Generate an AI-enhanced report using Gemini API."""
        if self.df_processed is None or self.labels is None:
            messagebox.showwarning("No clustering", "Run clustering first.")
            return

        # First generate the base report
        self.update_report()

        # Check for API key
        api_key = 'AIzaSyDbcu8bXTDvjUPBn8gP5z4kVP7lXj8vXsA' #os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            messagebox.showinfo(
                "No API Key",
                "Gemini API key not found.\n\n"
                "To enable AI-enhanced reports:\n"
                "1. Get a free API key from:\n"
                "   https://aistudio.google.com/app/apikey\n"
                "2. Create a .env file in the project folder\n"
                "3. Add: GEMINI_API_KEY=your_key_here\n\n"
                "Using template-based report instead."
            )
            return

        # Prepare data summary for LLM
        try:
            df = self.df_processed.copy()
            labels = np.array(self.labels)
            df["cluster"] = labels

            # Build context for LLM
            context = self._build_llm_context(df, labels)

            # Call Gemini API
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            prompt = f"""You are a data science expert analyzing clustering results. Based on the following clustering analysis data, provide a comprehensive and insightful interpretation.

{context}

Please provide:
1. **Executive Summary** (2-3 sentences overview)
2. **Cluster Interpretations** (what each cluster represents, their characteristics)
3. **Quality Assessment** (based on the metrics, is this a good clustering?)
4. **Key Insights** (interesting patterns or findings)
5. **Recommendations** (what could be improved or investigated further)

Format your response in a clear, professional manner suitable for a technical report. Use bullet points where appropriate."""

            response = model.generate_content(prompt)

            # Append AI interpretation to report
            ai_text = "\n\n" + "=" * 60 + "\n"
            ai_text += "🤖 AI-GENERATED INTERPRETATION (Gemini)\n"
            ai_text += "=" * 60 + "\n\n"
            ai_text += response.text
            ai_text += "\n\n" + "=" * 60

            self.report_text.insert(tk.END, ai_text)
            messagebox.showinfo("Success", "AI-enhanced report generated successfully!")

        except Exception as e:
            messagebox.showerror("API Error", f"Failed to generate AI report:\n{e}")

    def _build_llm_context(self, df, labels):
        """Build context string for LLM from clustering results."""
        unique_labels = np.unique(labels)
        total = len(df)

        context_parts = []

        # Algorithm info
        algo = self.algo_var.get()
        context_parts.append(f"Algorithm: {algo}")
        if algo in ["kmeans", "hierarchical", "gmm"]:
            context_parts.append(f"Number of clusters: {self.n_clusters_var.get()}")
        elif algo == "dbscan":
            context_parts.append(f"DBSCAN params: eps={self.eps_var.get()}, min_samples={self.min_samples_var.get()}")

        context_parts.append(f"Total samples: {total}")
        context_parts.append(f"Features used: {', '.join(self.selected_columns)}")

        # Metrics
        try:
            if len(unique_labels) >= 2:
                sil = silhouette_score(self.X, labels)
                dbi = davies_bouldin_score(self.X, labels)
                ch = calinski_harabasz_score(self.X, labels)
                context_parts.append(f"\nValidation Metrics:")
                context_parts.append(f"- Silhouette Score: {sil:.4f} (range -1 to 1, higher is better)")
                context_parts.append(f"- Davies-Bouldin Index: {dbi:.4f} (lower is better)")
                context_parts.append(f"- Calinski-Harabasz Score: {ch:.4f} (higher is better)")
        except:
            pass

        # Cluster summaries
        context_parts.append(f"\nCluster Summaries:")
        numeric_cols = [c for c in self.selected_columns if c in self.numeric_cols]

        for lbl in sorted(unique_labels):
            cluster_df = df[df["cluster"] == lbl]
            size = len(cluster_df)
            pct = 100.0 * size / total

            cluster_name = f"Cluster {lbl}" if lbl != -1 else "Noise"
            context_parts.append(f"\n{cluster_name}: {size} samples ({pct:.1f}%)")

            if numeric_cols:
                for col in numeric_cols:
                    mean_val = cluster_df[col].mean()
                    overall_mean = df[col].mean()
                    diff = ((mean_val - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0
                    context_parts.append(f"  - {col}: mean={mean_val:.3f} ({diff:+.1f}% vs overall)")

        return "\n".join(context_parts)

    def export_professional_pdf(self):
        """Export report as a professionally formatted PDF."""
        content = self.report_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Empty report", "Generate a report first.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT

            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )

            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.darkblue
            )
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                leading=14
            )
            code_style = ParagraphStyle(
                'CodeStyle',
                parent=styles['Code'],
                fontSize=8,
                spaceAfter=6,
                leftIndent=20,
                backColor=colors.lightgrey
            )

            # Build PDF content
            story = []

            # Title
            story.append(Paragraph("Cluster Analysis Report", title_style))
            story.append(Spacer(1, 12))

            # Metadata
            from datetime import datetime
            meta_data = [
                ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Algorithm:", self.algo_var.get()],
                ["Samples:", str(len(self.df_processed)) if self.df_processed is not None else "N/A"],
            ]
            meta_table = Table(meta_data, colWidths=[1.5*inch, 3*inch])
            meta_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 20))

            # Process content
            lines = content.split('\n')
            current_section = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Detect section headers
                if line.startswith('===') and len(line) > 10:
                    continue
                elif line.startswith('---') and len(line) > 10:
                    continue
                elif line.isupper() and len(line) > 5:
                    # Section header
                    story.append(Paragraph(line, heading_style))
                elif line.startswith('•') or line.startswith('-'):
                    # Bullet point
                    bullet_text = line.lstrip('•-').strip()
                    story.append(Paragraph(f"• {bullet_text}", body_style))
                elif ':' in line and len(line.split(':')[0]) < 30:
                    # Key-value pair
                    story.append(Paragraph(line, body_style))
                else:
                    # Regular text
                    story.append(Paragraph(line, body_style))

            # Build PDF
            doc.build(story)
            messagebox.showinfo("PDF Exported", f"Professional PDF saved to:\n{filepath}")

        except ImportError as e:
            messagebox.showerror("Missing Package", f"Required package not found:\n{e}\n\nInstall with: pip install reportlab")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF:\n{e}")


if __name__ == "__main__":
    app = ClusterExplorerApp()
    app.mainloop()
