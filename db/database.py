import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Index
from sqlalchemy.orm import declarative_base, sessionmaker

# Base class for declarative models
Base = declarative_base()

class District(Base):
    """
    District model representing educational resource allocation and infrastructure statistics
    for a given district.
    """
    __tablename__ = 'districts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String(100), nullable=False)
    district = Column(String(100), nullable=False)
    
    # Infrastructure percentages (0 to 100)
    electricity_perc = Column(Float, nullable=False)
    drinking_water_perc = Column(Float, nullable=False)
    computer_perc = Column(Float, nullable=False)
    internet_perc = Column(Float, nullable=False)
    
    # Pupil-Teacher metrics
    student_teacher_ratio = Column(Float, nullable=False)
    total_schools = Column(Integer, nullable=False)
    total_students = Column(Integer, nullable=False)
    total_teachers = Column(Integer, nullable=False)
    
    # Machine Learning cluster fields (updated by clustering.py)
    cluster_label = Column(Integer, nullable=True)
    cluster_tier = Column(String(50), nullable=True)
    
    # Add indexes for efficient queries
    __table_args__ = (
        Index('idx_state_district', 'state', 'district', unique=True),
        Index('idx_cluster_tier', 'cluster_tier'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "state": self.state,
            "district": self.district,
            "electricity_perc": self.electricity_perc,
            "drinking_water_perc": self.drinking_water_perc,
            "computer_perc": self.computer_perc,
            "internet_perc": self.internet_perc,
            "student_teacher_ratio": self.student_teacher_ratio,
            "total_schools": self.total_schools,
            "total_students": self.total_students,
            "total_teachers": self.total_teachers,
            "cluster_label": self.cluster_label,
            "cluster_tier": self.cluster_tier
        }

def get_engine(db_url="sqlite:///db/districts.db"):
    """
    Creates and returns an SQLAlchemy engine. Creates local directory if SQLite database.
    """
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
    return create_engine(db_url, echo=False)

def get_session(engine):
    """
    Creates a new database session from the engine.
    """
    Session = sessionmaker(bind=engine)
    return Session()

def init_db(raw_csv_path="data/districts_raw.csv", db_url="sqlite:///db/districts.db", force=False):
    """
    Initializes the database schema and seeds it from the raw CSV data if empty or forced.
    """
    engine = get_engine(db_url)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    session = get_session(engine)
    
    try:
        # Check if table is empty
        count = session.query(District).count()
        
        if count == 0 or force:
            if force:
                print("Force flag set. Clearing existing data...")
                session.query(District).delete()
                session.commit()
                
            print(f"Seeding database from: {raw_csv_path} ...")
            if not os.path.exists(raw_csv_path):
                raise FileNotFoundError(f"Raw CSV file not found at {raw_csv_path}. Run generate_mock_data.py first.")
                
            df = pd.read_csv(raw_csv_path)
            
            districts_to_add = []
            for _, row in df.iterrows():
                district_obj = District(
                    state=row['state'],
                    district=row['district'],
                    electricity_perc=float(row['electricity_perc']),
                    drinking_water_perc=float(row['drinking_water_perc']),
                    computer_perc=float(row['computer_perc']),
                    internet_perc=float(row['internet_perc']),
                    student_teacher_ratio=float(row['student_teacher_ratio']),
                    total_schools=int(row['total_schools']),
                    total_students=int(row['total_students']),
                    total_teachers=int(row['total_teachers'])
                )
                districts_to_add.append(district_obj)
                
            session.bulk_save_objects(districts_to_add)
            session.commit()
            print(f"Successfully seeded database with {len(districts_to_add)} district records.")
        else:
            print(f"Database already contains {count} records. Seeding skipped.")
            
    except Exception as e:
        session.rollback()
        print(f"Database initialization failed: {e}")
        raise e
    finally:
        session.close()

if __name__ == "__main__":
    # If run directly, initialize database from default path
    init_db()
