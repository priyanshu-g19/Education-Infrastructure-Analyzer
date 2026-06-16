import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from db.database import get_engine, get_session, District

def run_clustering_pipeline(db_url="sqlite:///db/districts.db", n_clusters=3):
    """
    Executes the K-Means clustering pipeline:
    1. Loads features from database
    2. Scales the features
    3. Fits KMeans
    4. Deterministically maps clusters to labels and tiers based on a "Readiness Score"
    5. Saves assignments back to database
    6. Returns metadata for visual plotting (inertias, centroids)
    """
    engine = get_engine(db_url)
    session = get_session(engine)
    
    try:
        # Load all districts
        districts = session.query(District).all()
        if not districts:
            raise ValueError("No districts found in the database. Please initialize the database first.")
            
        data = [d.to_dict() for d in districts]
        df = pd.DataFrame(data)
        
        # Select features for clustering
        feature_cols = [
            "electricity_perc", 
            "drinking_water_perc", 
            "computer_perc", 
            "internet_perc", 
            "student_teacher_ratio"
        ]
        X = df[feature_cols].copy()
        
        # Standardize features
        scaler = StandardScaler()
        # For clustering, we want to scale. Note: student_teacher_ratio has a negative relationship with quality.
        # We will scale everything normally. Standard K-Means handles this.
        X_scaled = scaler.fit_transform(X)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate centroids in original scale
        centroids = []
        for i in range(n_clusters):
            members = df[labels == i]
            centroids.append(members[feature_cols].mean().to_dict())
            
        # Deterministically map clusters to tiers using a "Readiness Score"
        # Readiness Score = Electricity + Water + Computer + Internet - 2 * STR
        readiness_scores = []
        for i in range(n_clusters):
            c = centroids[i]
            score = (c["electricity_perc"] + c["drinking_water_perc"] + 
                     c["computer_perc"] + c["internet_perc"] - 2.0 * c["student_teacher_ratio"])
            readiness_scores.append((score, i))
            
        # Sort readiness_scores: lowest is severe deficit, highest is well-equipped
        readiness_scores.sort(key=lambda x: x[0])
        
        # Create mapping dictionary
        # Mapping: old_label -> (new_label, tier_name)
        # 0: Severe Deficit, 1: Moderate Deficit, 2: Well-Equipped
        mapping = {}
        tiers_ordered = ["Severe Deficit", "Moderate Deficit", "Well-Equipped"]
        for new_label, (score, old_label) in enumerate(readiness_scores):
            mapping[old_label] = {
                "label": new_label,
                "tier": tiers_ordered[new_label]
            }
            
        # Update records in database
        for d in districts:
            old_lbl = labels[df[df['id'] == d.id].index[0]]
            d.cluster_label = mapping[old_lbl]["label"]
            d.cluster_tier = mapping[old_lbl]["tier"]
            
        session.commit()
        print("Successfully updated cluster labels in the database.")
        
        # Calculate Elbow inertias for dashboard visualization (K from 1 to 8)
        inertias = []
        k_values = list(range(1, 9))
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(float(km.inertia_))
            
        # Prepare metadata for return
        mapped_centroids = {}
        for old_label, map_info in mapping.items():
            mapped_centroids[map_info["tier"]] = centroids[old_label]
            
        return {
            "status": "success",
            "mapped_centroids": mapped_centroids,
            "elbow_data": {"k": k_values, "inertia": inertias},
            "cluster_counts": df['cluster_tier'].value_counts().to_dict() if 'cluster_tier' in df.columns else {}
        }
        
    except Exception as e:
        session.rollback()
        print(f"Clustering pipeline failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

if __name__ == "__main__":
    # Run pipeline with default database
    res = run_clustering_pipeline()
    print("Pipeline result:")
    print("Centroids:")
    for tier, center in res.get("mapped_centroids", {}).items():
        print(f"  {tier}: {center}")
