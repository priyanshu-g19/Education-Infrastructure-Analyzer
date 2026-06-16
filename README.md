# Infrastructure Deficit Clustering for Educational Resource Allocation

A data-driven machine learning application to identify and cluster educational infrastructure deficits across Indian districts. This application assists policymakers, government bodies, and NGOs in prioritizing resource allocation and budget distribution to bridge the digital and infrastructural divide.

## 🌟 Sustainable Development Goals (SDG) Alignment

- **SDG 4: Quality Education** – Targets structural deficits in basic school environments (lack of electricity, drinking water) and technical shortcomings (computers, internet access) to secure inclusive, high-quality schooling.
- **SDG 10: Reduced Inequalities** – Employs an objective, unsupervised machine learning classifier to rank districts into resource deficit tiers, enabling fair, needs-based funding that bridges socio-economic disparities between rural and urban districts.

---

## 🛠️ Project Architecture & Components

The application is structured into modular components:

```text
f:\Priyanshu project\
│   requirements.txt              # Core dependencies
│   README.md                     # Documentation
│   app.py                        # Streamlit dashboard UI
│
├───data/
│       generate_mock_data.py     # Generates raw district profiles
│       districts_raw.csv         # Raw district infrastructure metrics (Generated)
│
├───db/
│       database.py               # Database layer (SQLite schema setup & seeding)
│       districts.db              # SQLite Database file (Generated)
│
└───ml/
        clustering.py             # Machine learning clustering pipeline (K-Means)
```

1. **Mock Data Generation (`data/generate_mock_data.py`)**: Simulates realistic educational profiles for 778 districts across 36 Indian states and UTs based on UDISE+ and data.gov.in metrics.
2. **Database Connector (`db/database.py`)**: Handles object-relational mapping (ORM) with SQLAlchemy and creates indexed SQLite tables to store district records.
3. **Machine Learning Pipeline (`ml/clustering.py`)**: 
   - Standardizes the features using `StandardScaler`.
   - Runs K-Means clustering ($K=3$).
   - Dynamically resolves K-Means label shuffling by computing a **Readiness Score** for each cluster centroid:
     $$\text{Readiness Score} = \text{Electricity\%} + \text{Drinking Water\%} + \text{Computers\%} + \text{Internet\%} - 2 \times \text{Student-Teacher Ratio}$$
   - Maps clusters to deterministic deficit tiers: **Severe Deficit** (lowest score), **Moderate Deficit** (medium), and **Well-Equipped** (highest).
4. **Interactive Streamlit Dashboard (`app.py`)**:
   - A highly aesthetic web interface using custom CSS styling, metric cards, and badge indicators.
   - Built-in interactive charts (Plotly 3D scatter plots, pie charts, state comparison bar charts).
   - Instant query, profile viewer, and dynamic policy action generation for individual districts.
   - Raw data filter, search, and CSV export.
   - Run-time clustering control panel in the sidebar.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ and `pip` installed.

### 2. Installation
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Generate Mock Data
Generate the district metrics dataset:
```bash
python data/generate_mock_data.py
```

### 4. Setup and Seed Database
Create the SQLite database and seed it with the raw data:
```bash
python db/database.py
```

### 5. Run Machine Learning Pipeline
Perform the K-Means clustering and save assignments:
```bash
python -m ml.clustering
```

### 6. Start the Dashboard
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
After running, open `http://localhost:8501` in your browser.

---

## 📈 Machine Learning Features
Clustering is performed using 5 key metrics:
- **Electricity (`electricity_perc`)**: Percentage of schools with functional electricity.
- **Drinking Water (`drinking_water_perc`)**: Percentage of schools with functional drinking water.
- **Computers (`computer_perc`)**: Percentage of schools with functional computers.
- **Internet (`internet_perc`)**: Percentage of schools with functional internet.
- **Student-Teacher Ratio (`student_teacher_ratio`)**: Average number of students per teacher.

### Cluster Tiers Definitions
- **Severe Deficit**: Lacks basic utilities (power/water) and tech tools. Has a high student-teacher ratio. Requires immediate emergency funding.
- **Moderate Deficit**: Stable basic utilities, but low digital amenities. Requires IT and digital laboratory funding.
- **Well-Equipped**: Excellent basic utilities, high digital presence, and low student-teacher ratio. Requires standard upkeep and maintenance.
