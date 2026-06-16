import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from db.database import get_engine, get_session, District, init_db
from ml.clustering import run_clustering_pipeline

# Page Configuration
st.set_page_config(
    page_title="Infrastructure Deficit Clustering Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Vanilla CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    
    /* Header Gradient */
    .title-gradient {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Glassmorphism Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 6px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        text-align: center;
    }
    .badge-severe {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.3));
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.3));
        color: #fde68a;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-well {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.3));
        color: #a7f3d0;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    /* Policy Box */
    .policy-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 5px solid #6366f1;
        border-radius: 4px 12px 12px 4px;
        padding: 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Database Engine Caching
@st.cache_resource
def get_db_engine():
    db_url = "sqlite:///db/districts.db"
    return get_engine(db_url)

engine = get_db_engine()

# Helper to fetch data into DataFrame
def load_data_from_db():
    session = get_session(engine)
    try:
        # Check if database has any records, if not, initialize it
        count = session.query(District).count()
        if count == 0:
            init_db(db_url="sqlite:///db/districts.db")
            run_clustering_pipeline(db_url="sqlite:///db/districts.db")
        
        # Verify cluster_tier is populated, if not, run clustering
        clustered_count = session.query(District).filter(District.cluster_tier != None).count()
        if clustered_count == 0:
            run_clustering_pipeline(db_url="sqlite:///db/districts.db")
            
        districts = session.query(District).all()
        data = [d.to_dict() for d in districts]
        return pd.DataFrame(data)
    finally:
        session.close()

# Load initial data
df = load_data_from_db()

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-education-school-and-education-flatart-icons-flat-flatarticons-1.png", width=70)
    st.markdown("### SDG-4 & SDG-10 Educational Allocation Tool")
    st.markdown("---")
    
    st.markdown("### 📊 Clustering Engine")
    
    if st.button("⚡ Re-run K-Means Clustering", use_container_width=True):
        with st.spinner("Re-training model and clustering districts..."):
            res = run_clustering_pipeline(db_url="sqlite:///db/districts.db")
            if res["status"] == "success":
                st.success("Clustering complete! Database updated.")
                st.rerun()
            else:
                st.error(f"Error: {res.get('message', 'Unknown error')}")
                
    st.markdown("---")
    
    st.markdown("### 🌐 SDG Alignment")
    st.info("""
    **SDG 4 (Quality Education):** Targets school resource shortfalls to foster equitable, basic learning conditions.
    
    **SDG 10 (Reduced Inequalities):** Directs infrastructure funding to the most underserved districts first.
    """)
    
    st.markdown("### 🗃️ System Metadata")
    st.write(f"**Districts Seeding:** {len(df)}")
    st.write(f"**Clustering Model:** K-Means (K=3)")
    st.write(f"**Database:** SQLite (SQLAlchemy)")
    st.write(f"**Dataset Source:** UDISE+ / Data.gov.in")

# ---- MAIN HEADER ----
st.markdown("<h1 class='title-gradient'>Infrastructure Deficit Clustering</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A Machine Learning-driven Educational Resource Allocation Framework for Indian Districts</p>", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 National Overview", 
    "🔍 District Profile & Analysis", 
    "🧠 Clustering Analysis & Math", 
    "📂 Data Explorer"
])

# ---- TAB 1: NATIONAL OVERVIEW ----
with tab1:
    # KPI Metrics
    total_districts = len(df)
    
    # Avoid errors if cluster_tier doesn't exist
    if 'cluster_tier' in df.columns and not df['cluster_tier'].isna().all():
        severe_df = df[df['cluster_tier'] == 'Severe Deficit']
        moderate_df = df[df['cluster_tier'] == 'Moderate Deficit']
        equipped_df = df[df['cluster_tier'] == 'Well-Equipped']
        
        severe_pct = (len(severe_df) / total_districts) * 100
        moderate_pct = (len(moderate_df) / total_districts) * 100
        equipped_pct = (len(equipped_df) / total_districts) * 100
    else:
        severe_pct = moderate_pct = equipped_pct = 0.0
        severe_df = moderate_df = equipped_df = pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Total Districts</div>
            <div class='metric-value'>{total_districts}</div>
            <div style='color: #38bdf8; font-weight: 600;'>National Coverage</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Severe Deficit</div>
            <div class='metric-value' style='color: #f87171;'>{len(severe_df)}</div>
            <div style='color: #f87171; font-weight: 600;'>{severe_pct:.1f}% of districts</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Moderate Deficit</div>
            <div class='metric-value' style='color: #fbbf24;'>{len(moderate_df)}</div>
            <div style='color: #fbbf24; font-weight: 600;'>{moderate_pct:.1f}% of districts</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Well-Equipped</div>
            <div class='metric-value' style='color: #34d399;'>{len(equipped_df)}</div>
            <div style='color: #34d399; font-weight: 600;'>{equipped_pct:.1f}% of districts</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Section
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        st.markdown("### 🍩 National Tier Distribution")
        # Deficit Tier Pie Chart
        tier_counts = df['cluster_tier'].value_counts().reset_index()
        tier_counts.columns = ['Deficit Tier', 'Count']
        
        fig_pie = px.pie(
            tier_counts, 
            values='Count', 
            names='Deficit Tier',
            color='Deficit Tier',
            color_discrete_map={
                'Severe Deficit': '#ef4444',
                'Moderate Deficit': '#f59e0b',
                'Well-Equipped': '#10b981'
            },
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=13),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.markdown("### 📊 State-wise Deficit Distribution (Top 10 States with Severe Deficits)")
        # Pivot table for states and tiers
        state_pivot = pd.crosstab(df['state'], df['cluster_tier']).reset_index()
        # Sort by severe deficit
        state_pivot = state_pivot.sort_values(by='Severe Deficit', ascending=False).head(10)
        
        fig_bar = px.bar(
            state_pivot,
            x='state',
            y=['Severe Deficit', 'Moderate Deficit', 'Well-Equipped'],
            title=None,
            labels={'value': 'Number of Districts', 'state': 'State', 'variable': 'Deficit Tier'},
            color_discrete_map={
                'Severe Deficit': '#ef4444',
                'Moderate Deficit': '#f59e0b',
                'Well-Equipped': '#10b981'
            },
            barmode='stack'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(showgrid=False, title='State'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Districts Count'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ---- TAB 2: DISTRICT PROFILE & RECOMMENDATIONS ----
with tab2:
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_state = st.selectbox("Select State/UT", sorted(df['state'].unique()))
    with col_sel2:
        state_districts = df[df['state'] == selected_state]
        selected_district = st.selectbox("Select District", sorted(state_districts['district'].unique()))
        
    # Get District Data
    district_data = df[(df['state'] == selected_state) & (df['district'] == selected_district)].iloc[0]
    
    st.markdown("---")
    
    # Row for District info
    col_d1, col_d2 = st.columns([1, 1.2])
    
    with col_d1:
        st.markdown("### 📌 District Overview")
        
        # Select badge based on tier
        tier = district_data['cluster_tier']
        if tier == 'Severe Deficit':
            badge_html = "<span class='badge badge-severe'>Severe Deficit</span>"
        elif tier == 'Moderate Deficit':
            badge_html = "<span class='badge badge-moderate'>Moderate Deficit</span>"
        else:
            badge_html = "<span class='badge badge-well'>Well-Equipped</span>"
            
        st.markdown(f"""
        <div class='metric-card' style='margin-bottom: 20px;'>
            <h2 style='margin:0 0 10px 0;'>{selected_district}</h2>
            <h5 style='margin:0 0 15px 0; color: #94a3b8;'>{selected_state}</h5>
            <div style='margin-bottom: 15px;'>{badge_html}</div>
            <p style='margin: 0;'>Total Schools: <b>{district_data['total_schools']}</b></p>
            <p style='margin: 0;'>Total Students: <b>{district_data['total_students']:,}</b></p>
            <p style='margin: 0;'>Total Teachers: <b>{district_data['total_teachers']:,}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Policy Recommendation Box
        st.markdown("### 📋 Actionable Resource Allocation Guidelines")
        
        recs = []
        if district_data['electricity_perc'] < 75:
            recs.append("⚡ **Grid & Solar Electrification:** Target the remaining schools with solar energy micro-grid installations, prioritizing schools with 0% electricity access.")
        if district_data['drinking_water_perc'] < 80:
            recs.append("💧 **Safe Drinking Water Campaign:** Build functional tap connections and install community RO/filtration systems under national sanitation funds.")
        if district_data['computer_perc'] < 40:
            recs.append("💻 **ICT Computer Labs Setup:** Budget for centralized computer laboratories under the PM SHRI or ICT @ School initiatives.")
        if district_data['internet_perc'] < 30:
            recs.append("🌐 **Telecom/VSAT Broadband Connectivity:** Coordinate with public/private ISPs to deploy broadband or low-cost satellite-internet (VSAT) nodes.")
        if district_data['student_teacher_ratio'] > 35:
            recs.append("👩‍🏫 **Primary/Secondary Teacher Recruitment:** Immediately allocate additional teaching posts to resolve the high pupil-teacher ratio and meet national RTE standards (< 30:1).")
            
        if tier == 'Well-Equipped' and not recs:
            recs.append("✅ **Regular Infrastructure Auditing:** Focus on standard equipment maintenance and technical upgrading. Direct new infrastructure funds to neighboring deficit districts.")
        elif not recs:
            recs.append("🛠️ **Asset Optimization:** Existing amenities are stable. Allocate minor budgets for technology maintenance and training teachers in digital curriculum.")
            
        recs_list_html = "".join([f"<li>{r}</li>" for r in recs])
        st.markdown(f"""
        <div class='policy-box'>
            <h5 style='margin: 0 0 10px 0; color: #818cf8; font-weight: 600;'>POLICY ACTIONS REQUIRED:</h5>
            <ul style='padding-left: 20px; margin: 0;'>
                {recs_list_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown("### 📊 Infrastructure Metrics vs State & National Averages")
        
        # Compute State Average and National Average
        state_avg = df[df['state'] == selected_state].mean(numeric_only=True)
        national_avg = df.mean(numeric_only=True)
        
        # Metrics to display
        metrics = ["Electricity (%)", "Drinking Water (%)", "Computers (%)", "Internet (%)", "Student-Teacher Ratio"]
        dist_vals = [
            district_data['electricity_perc'], 
            district_data['drinking_water_perc'], 
            district_data['computer_perc'], 
            district_data['internet_perc'],
            district_data['student_teacher_ratio']
        ]
        state_vals = [
            state_avg['electricity_perc'], 
            state_avg['drinking_water_perc'], 
            state_avg['computer_perc'], 
            state_avg['internet_perc'],
            state_avg['student_teacher_ratio']
        ]
        nat_vals = [
            national_avg['electricity_perc'], 
            national_avg['drinking_water_perc'], 
            national_avg['computer_perc'], 
            national_avg['internet_perc'],
            national_avg['student_teacher_ratio']
        ]
        
        fig_compare = go.Figure()
        
        # District bars
        fig_compare.add_trace(go.Bar(
            name='This District',
            x=metrics,
            y=dist_vals,
            marker_color='#6366f1'
        ))
        # State Average bars
        fig_compare.add_trace(go.Bar(
            name='State Average',
            x=metrics,
            y=state_vals,
            marker_color='#818cf8',
            opacity=0.6
        ))
        # National Average bars
        fig_compare.add_trace(go.Bar(
            name='National Average',
            x=metrics,
            y=nat_vals,
            marker_color='#94a3b8',
            opacity=0.4
        ))
        
        fig_compare.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Percentage / Ratio Value'),
            xaxis=dict(title=None)
        )
        st.plotly_chart(fig_compare, use_container_width=True)


# ---- TAB 3: CLUSTERING ANALYSIS & MATH ----
with tab3:
    st.markdown("### 🔬 Unsupervised Machine Learning Formulation")
    st.markdown("""
    To categorize districts objectively and mathematical, we utilize **K-Means Clustering**, a partitioning algorithm that groups similar data points.
    
    1. **Preprocessing (Standardization):** Features possess different ranges (percentages are $0-100\%$, Student-Teacher Ratios range $10-60$). To prevent large values from dominating, features are normalized using `StandardScaler`:
       $$z = \\frac{x - \\mu}{\\sigma}$$
    2. **Optimization:** We execute K-Means on the standardized array. The distance between points and centroids is minimized:
       $$J = \\sum_{i=1}^{k} \\sum_{x \\in C_i} ||x - \\mu_i||^2$$
    3. **Consistency:** Because initial centroids are randomized, cluster indexes shuffles on re-training. We define a custom **Readiness Score** to rank the resulting clusters deterministically:
       $$\\text{Readiness Score} = \\text{electricity\\%} + \\text{drinking\\_water\\%} + \\text{computer\\%} + \\text{internet\\%} - 2 \\times \\text{Student-Teacher Ratio}$$
       Tiers are ordered as **Severe Deficit** (lowest score), **Moderate Deficit** (middle), and **Well-Equipped** (highest).
    """)
    
    st.markdown("---")
    
    col_math1, col_math2 = st.columns([1.2, 1])
    
    # We will compute the elbow curve and centroids using the pipeline return structure.
    # To display it dynamically, we run a pipeline run in memory (which is cached/fast).
    res_pipeline = run_clustering_pipeline(db_url="sqlite:///db/districts.db")
    
    with col_math1:
        st.markdown("### 🕸️ Interactive 3D Cluster Visualization")
        st.markdown("*Use your mouse to rotate, zoom, and hover over individual districts.*")
        
        # 3D scatter plot of districts
        fig_3d = px.scatter_3d(
            df,
            x='electricity_perc',
            y='computer_perc',
            z='student_teacher_ratio',
            color='cluster_tier',
            hover_name='district',
            hover_data=['state', 'drinking_water_perc', 'internet_perc'],
            color_discrete_map={
                'Severe Deficit': '#ef4444',
                'Moderate Deficit': '#f59e0b',
                'Well-Equipped': '#10b981'
            },
            labels={
                'electricity_perc': 'Electricity (%)',
                'computer_perc': 'Computers (%)',
                'student_teacher_ratio': 'Student-Teacher Ratio',
                'cluster_tier': 'Deficit Tier'
            }
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)", showbackground=False),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)", showbackground=False),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)", showbackground=False),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=0),
            font=dict(color='#e2e8f0'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with col_math2:
        if res_pipeline.get("status") == "success":
            # Plot Elbow curve
            st.markdown("### 📐 The Elbow Method")
            st.write("The Elbow Method plots the sum of squared distances (inertia) versus the number of clusters. The 'elbow' point indicates the optimal trade-off between complexity and group cohesion.")
            
            elbow_df = pd.DataFrame(res_pipeline["elbow_data"])
            fig_elbow = px.line(
                elbow_df, 
                x='k', 
                y='inertia', 
                markers=True,
                color_discrete_sequence=['#818cf8']
            )
            # Annotate K=3
            fig_elbow.add_annotation(
                x=3, y=elbow_df.loc[elbow_df['k'] == 3, 'inertia'].values[0],
                text="Optimal K=3",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-30,
                font=dict(color="#fde68a", size=12),
                arrowcolor="#fde68a"
            )
            fig_elbow.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(showgrid=False, title='Number of Clusters (K)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Inertia (In-Group Variance)')
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
            
            # Show centroids details in a table
            st.markdown("### 🎯 Cluster Centroids Profiles")
            st.write("Average values of variables at cluster centers:")
            
            centroid_data = []
            for tier, center in res_pipeline["mapped_centroids"].items():
                center_row = {"Deficit Tier": tier}
                for f_name, f_val in center.items():
                    center_row[f_name.replace("_perc", " (%)").replace("_", " ").title()] = round(f_val, 1)
                centroid_data.append(center_row)
                
            st.dataframe(pd.DataFrame(centroid_data), use_container_width=True, hide_index=True)


# ---- TAB 4: DATA EXPLORER ----
with tab4:
    st.markdown("### 🔍 Search & Export Datasets")
    st.write("Filter the district dataset based on region and cluster classification, and export to CSV.")
    
    # Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        f_states = st.multiselect("Filter by State/UT", sorted(df['state'].unique()))
    with col_f2:
        f_tiers = st.multiselect("Filter by Deficit Tier", sorted(df['cluster_tier'].unique()))
    with col_f3:
        search_query = st.text_input("Search District Name")
        
    # Apply filters
    filtered_df = df.copy()
    if f_states:
        filtered_df = filtered_df[filtered_df['state'].isin(f_states)]
    if f_tiers:
        filtered_df = filtered_df[filtered_df['cluster_tier'].isin(f_tiers)]
    if search_query:
        filtered_df = filtered_df[filtered_df['district'].str.contains(search_query, case=False)]
        
    # Render table
    st.dataframe(
        filtered_df.rename(columns={
            "state": "State/UT",
            "district": "District",
            "electricity_perc": "Electricity (%)",
            "drinking_water_perc": "Drinking Water (%)",
            "computer_perc": "Computers (%)",
            "internet_perc": "Internet (%)",
            "student_teacher_ratio": "Student-Teacher Ratio",
            "total_schools": "Total Schools",
            "total_students": "Total Students",
            "total_teachers": "Total Teachers",
            "cluster_tier": "Deficit Tier"
        }).drop(columns=["id", "cluster_label"]),
        use_container_width=True
    )
    
    # Export csv button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="educational_deficit_clustered_data.csv",
        mime="text/csv",
    )
