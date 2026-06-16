import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Precise mapping of Indian States and UTs to their actual districts
STATE_TO_DISTRICTS = {
    "Andhra Pradesh": [
        "Alluri Sitharama Raju", "Anakapalli", "Ananthapuramu", "Annamayya", "Bapatla", 
        "Chittoor", "Dr. B.R. Ambedkar Konaseema", "East Godavari", "Eluru", "Guntur", 
        "Kakinada", "Krishna", "Kurnool", "Nandyal", "NTR", "Palnadu", 
        "Prakasam", "Sri Potti Sriramulu Nellore", "Sri Sathya Sai", "Srikakulam", 
        "Tirupati", "Visakhapatnam", "Vizianagaram", "West Godavari", "YSR Kadapa", "Parvathipuram Manyam"
    ],
    "Arunachal Pradesh": [
        "Anjaw", "Changlang", "Dibang Valley", "East Kameng", "East Siang", "Kamle", 
        "Kra Daadi", "Kurung Kumey", "Lepa Rada", "Lohit", "Longding", "Lower Dibang Valley", 
        "Lower Siang", "Lower Subansiri", "Namsai", "Pakke Kessang", "Papum Pare", 
        "Shi Yomi", "Siang", "Tawang", "Tirap", "Upper Siang", "Upper Subansiri", 
        "West Kameng", "West Siang"
    ],
    "Assam": [
        "Bajali", "Baksa", "Barpeta", "Biswanath", "Bongaigaon", "Cachar", "Charaideo", 
        "Chirang", "Darrang", "Dhemaji", "Dhubri", "Dibrugarh", "Dima Hasao", "Goalpara", 
        "Golaghat", "Hailakandi", "Hojai", "Jorhat", "Kamrup", "Kamrup Metropolitan", 
        "Karbi Anglong", "Karimganj", "Kokrajhar", "Lakhimpur", "Majuli", "Morigaon", 
        "Nagaon", "Nalbari", "Sivasagar", "Sonitpur", "South Salmara-Mankachar", 
        "Tinsukia", "Udalguri", "West Karbi Anglong", "Tamulpur"
    ],
    "Bihar": [
        "Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur", "Bhojpur", 
        "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj", "Jamui", "Jehanabad", 
        "Kaimur", "Katihar", "Khagaria", "Kishanganj", "Lakhisarai", "Madhepura", 
        "Madhubani", "Munger", "Muzaffarpur", "Nalanda", "Nawada", "Patna", "Purnia", 
        "Rohtas", "Saharsa", "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", 
        "Siwan", "Supaul", "Vaishali", "West Champaran"
    ],
    "Chhattisgarh": [
        "Balod", "Baloda Bazar-Bhatapara", "Balrampur-Ramanujganj", "Bastar", "Bemetara", 
        "Bijapur", "Bilaspur", "Dantewada", "Dhamtari", "Durg", "Gariaband", "Gaurela-Pendra-Marwahi", 
        "Janjgir-Champa", "Jashpur", "Kanker", "Kawardha", "Kondagaon", "Korba", "Koriya", 
        "Mahasamund", "Mungeli", "Narayanpur", "Raigarh", "Raipur", "Rajnandgaon", 
        "Sukma", "Surajpur", "Surguja", "Manendragarh-Chirmiri-Bharatpur", "Mohla-Manpur-Ambagarh Chowki",
        "Sakti", "Sarangarh-Bilaigarh", "Khairagarh-Chhuikhadan-Gandai"
    ],
    "Goa": ["North Goa", "South Goa"],
    "Gujarat": [
        "Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch", "Bhavnagar", 
        "Botad", "Chhota Udepur", "Dahod", "Dang", "Devbhumi Dwarka", "Gandhinagar", 
        "Gir Somnath", "Jamnagar", "Junagadh", "Kheda", "Kutch", "Mahisagar", "Mehsana", 
        "Morbi", "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar", "Rajkot", 
        "Sabarkantha", "Surat", "Surendranagar", "Tapi", "Vadodara", "Valsad"
    ],
    "Haryana": [
        "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad", "Gurugram", 
        "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Mahendragarh", 
        "Nuh", "Palwal", "Panchkula", "Panipat", "Rewari", "Rohtak", "Sirsa", 
        "Sonipat", "Yamunanagar"
    ],
    "Himachal Pradesh": [
        "Bilaspur", "Chamba", "Hamirpur", "Kangra", "Kinnaur", "Kullu", "Lahaul and Spiti", 
        "Mandi", "Shimla", "Sirmaur", "Solan", "Una"
    ],
    "Jharkhand": [
        "Bokaro", "Chatra", "Deoghar", "Dhanbad", "Dumka", "East Singhbhum", "Garhwa", 
        "Giridih", "Godda", "Gumla", "Hazaribagh", "Jamtara", "Khunti", "Koderma", 
        "Latehar", "Lohardaga", "Pakur", "Palamu", "Ramgarh", "Ranchi", "Sahibganj", 
        "Seraikela Kharsawan", "Simdega", "West Singhbhum"
    ],
    "Karnataka": [
        "Bagalkote", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", 
        "Bidar", "Chamarajanagara", "Chikkaballapura", "Chikkamagaluru", "Chitradurga", 
        "Dakshina Kannada", "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", 
        "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur", 
        "Ramanagara", "Shivamogga", "Tumakuru", "Udupi", "Uttara Kannada", "Vijayapura", 
        "Yadgir", "Vijayanagara"
    ],
    "Kerala": [
        "Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam", "Kottayam", 
        "Kozhikode", "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram", 
        "Thrissur", "Wayanad"
    ],
    "Madhya Pradesh": [
        "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", 
        "Betul", "Bhind", "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", 
        "Datia", "Dewas", "Dhar", "Dindori", "Guna", "Gwalior", "Harda", "Narmadapuram", 
        "Indore", "Jabalpur", "Jhabua", "Katni", "Khandwa", "Khargone", "Mandla", 
        "Mandsaur", "Morena", "Narsinghpur", "Neemuch", "Panna", "Raisen", "Rajgarh", 
        "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", "Seoni", "Shahdol", "Shajapur", 
        "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", "Ujjain", "Umaria", 
        "Vidisha", "Niwari"
    ],
    "Maharashtra": [
        "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", 
        "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", 
        "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", 
        "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad", 
        "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", 
        "Washim", "Yavatmal"
    ],
    "Manipur": [
        "Bishnupur", "Chandel", "Churachandpur", "Imphal East", "Imphal West", "Jiribam", 
        "Kakching", "Kamjong", "Kangpokpi", "Noney", "Pherzawl", "Senapati", "Tamenglong", 
        "Tengnoupal", "Thoubal", "Ukhrul"
    ],
    "Meghalaya": [
        "East Garo Hills", "East Jaintia Hills", "East Khasi Hills", "North Garo Hills", 
        "Ri Bhoi", "South Garo Hills", "South West Garo Hills", "South West Khasi Hills", 
        "West Garo Hills", "West Jaintia Hills", "West Khasi Hills", "Eastern West Khasi Hills"
    ],
    "Mizoram": [
        "Aizawl", "Champhai", "Hnahthial", "Khawzawl", "Kolasib", "Lawngtlai", 
        "Lunglei", "Mamit", "Saiha", "Saitual", "Serchhip"
    ],
    "Nagaland": [
        "Choumoukedima", "Dimapur", "Kiphire", "Kohima", "Longleng", "Mokokchung", 
        "Mon", "Niuland", "Noklak", "Peren", "Phek", "Shamator", "Tseminyu", 
        "Tuensang", "Wokha", "Zunheboto"
    ],
    "Odisha": [
        "Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh", "Cuttack", 
        "Deogarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghpur", "Jajpur", 
        "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara", "Keonjhar", "Khordha", 
        "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada", 
        "Puri", "Rayagada", "Sambalpur", "Subarnapur", "Sundargarh"
    ],
    "Punjab": [
        "Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Fazilka", 
        "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", 
        "Malerkotla", "Mansa", "Moga", "Muktsar", "Pathankot", "Patiala", "Rupnagar", 
        "Sahibzada Ajit Singh Nagar", "Sangrur", "Shahid Bhagat Singh Nagar", "Tarn Taran"
    ],
    "Rajasthan": [
        "Ajmer", "Alwar", "Banswara", "Baran", "Barmer", "Bharatpur", "Bhilwara", 
        "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Dholpur", "Dungarpur", 
        "Hanumangarh", "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", 
        "Jodhpur", "Karauli", "Kota", "Nagaur", "Pali", "Pratapgarh", "Rajsamand", 
        "Sawai Madhopur", "Sikar", "Sirohi", "Sri Ganganagar", "Tonk", "Udaipur"
    ],
    "Sikkim": [
        "Gangtok", "Gyalshing", "Mangan", "Namchi", "Pakyong", "Soreng"
    ],
    "Tamil Nadu": [
        "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", 
        "Dindigul", "Erode", "Kallakurichi", "Kanchipuram", "Kanyakumari", "Karur", 
        "Krishnagiri", "Madurai", "Mayiladuthurai", "Nagapattinam", "Namakkal", 
        "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem", 
        "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", 
        "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", 
        "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar"
    ],
    "Telangana": [
        "Adilabad", "Bhadradri Kothagudem", "Hanamkonda", "Hyderabad", "Jagtial", 
        "Jangaon", "Jayashankar Bhupalpally", "Jogulamba Gadwal", "Kamareddy", 
        "Karimnagar", "Khammam", "Kumuram Bheem Asifabad", "Mahabubabad", "Mahabubnagar", 
        "Mancherial", "Medak", "Medchal-Malkajgiri", "Mulugu", "Nagarkurnool", 
        "Nalgonda", "Nirmal", "Nizamabad", "Peddapalli", "Rajanna Sircilla", 
        "Rangareddy", "Sangareddy", "Siddipet", "Suryapet", "Vikarabad", "Wanaparthy", 
        "Warangal", "Yadadri Bhuvanagiri"
    ],
    "Tripura": [
        "Dhalai", "Gomati", "Khowai", "North Tripura", "Sepahijala", "South Tripura", 
        "Unakoti", "West Tripura"
    ],
    "Uttar Pradesh": [
        "Agra", "Aligarh", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya", "Azamgarh", 
        "Baghpat", "Bahraich", "Ballia", "Balrampur", "Banda", "Barabanki", "Bareilly", 
        "Basti", "Bhadohi", "Bijnor", "Budaun", "Bulandshahr", "Chandauli", "Chitrakoot", 
        "Deoria", "Etah", "Etawah", "Ayodhya", "Farrukhabad", "Fatehpur", "Firozabad", 
        "Gautam Buddha Nagar", "Ghaziabad", "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur", 
        "Hapur", "Hardoi", "Hathras", "Jalaun", "Jaunpur", "Jhansi", "Kannauj", 
        "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi", "Kheri", "Kushinagar", 
        "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri", "Mathura", "Mau", 
        "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar", "Pilibhit", "Pratapgarh", 
        "Raebareli", "Rampur", "Saharanpur", "Sambhal", "Sant Kabir Nagar", "Shahjahanpur", 
        "Shamli", "Shravasti", "Siddharthnagar", "Sitapur", "Sonbhadra", "Sultanpur", 
        "Unnao", "Varanasi"
    ],
    "Uttarakhand": [
        "Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar", 
        "Nainital", "Pauri Garhwal", "Pithoragarh", "Rudraprayag", "Tehri Garhwal", 
        "Udham Singh Nagar", "Uttarkashi"
    ],
    "West Bengal": [
        "Alipurduar", "Bankura", "Birbhum", "Cooch Behar", "Dakshin Dinajpur", "Darjeeling", 
        "Hooghly", "Howrah", "Jalpaiguri", "Jhargram", "Kalimpong", "Kolkata", 
        "Malda", "Murshidabad", "Nadia", "North 24 Parganas", "Paschim Bardhaman", 
        "Paschim Medinipur", "Purba Bardhaman", "Purba Medinipur", "Purulia", 
        "South 24 Parganas", "Uttar Dinajpur"
    ],
    "Delhi": [
        "Central Delhi", "East Delhi", "New Delhi", "North Delhi", "North East Delhi", 
        "North West Delhi", "Shahdara", "South Delhi", "South East Delhi", "South West Delhi", 
        "West Delhi"
    ],
    "Jammu & Kashmir": [
        "Anantnag", "Bandipora", "Baramulla", "Budgam", "Doda", "Ganderbal", "Jammu", 
        "Kathua", "Kishtwar", "Kulgam", "Kupwara", "Poonch", "Pulwama", "Rajouri", 
        "Ramban", "Reasi", "Samba", "Shopian", "Srinagar", "Udhampur"
    ],
    "Ladakh": ["Kargil", "Leh"],
    "Puducherry": ["Karaikal", "Mahe", "Puducherry", "Yanam"],
    "Chandigarh": ["Chandigarh"],
    "Andaman & Nicobar Islands": ["Nicobar", "North and Middle Andaman", "South Andaman"],
    "Dadra & Nagar Haveli and Daman & Diu": ["Dadra and Nagar Haveli", "Daman", "Diu"],
    "Lakshadweep": ["Lakshadweep"]
}

# State developmental profiles to specify ratios of developed, developing, underdeveloped districts
STATES_PROFILES = {
    "Andhra Pradesh": {"developed_ratio": 0.4, "developing_ratio": 0.5, "underdeveloped_ratio": 0.1},
    "Arunachal Pradesh": {"developed_ratio": 0.1, "developing_ratio": 0.3, "underdeveloped_ratio": 0.6},
    "Assam": {"developed_ratio": 0.2, "developing_ratio": 0.5, "underdeveloped_ratio": 0.3},
    "Bihar": {"developed_ratio": 0.05, "developing_ratio": 0.35, "underdeveloped_ratio": 0.6},
    "Chhattisgarh": {"developed_ratio": 0.15, "developing_ratio": 0.45, "underdeveloped_ratio": 0.4},
    "Goa": {"developed_ratio": 0.9, "developing_ratio": 0.1, "underdeveloped_ratio": 0.0},
    "Gujarat": {"developed_ratio": 0.5, "developing_ratio": 0.4, "underdeveloped_ratio": 0.1},
    "Haryana": {"developed_ratio": 0.6, "developing_ratio": 0.3, "underdeveloped_ratio": 0.1},
    "Himachal Pradesh": {"developed_ratio": 0.4, "developing_ratio": 0.5, "underdeveloped_ratio": 0.1},
    "Jharkhand": {"developed_ratio": 0.1, "developing_ratio": 0.4, "underdeveloped_ratio": 0.5},
    "Karnataka": {"developed_ratio": 0.45, "developing_ratio": 0.45, "underdeveloped_ratio": 0.1},
    "Kerala": {"developed_ratio": 0.85, "developing_ratio": 0.15, "underdeveloped_ratio": 0.0},
    "Madhya Pradesh": {"developed_ratio": 0.15, "developing_ratio": 0.5, "underdeveloped_ratio": 0.35},
    "Maharashtra": {"developed_ratio": 0.5, "developing_ratio": 0.4, "underdeveloped_ratio": 0.1},
    "Manipur": {"developed_ratio": 0.15, "developing_ratio": 0.5, "underdeveloped_ratio": 0.35},
    "Meghalaya": {"developed_ratio": 0.1, "developing_ratio": 0.4, "underdeveloped_ratio": 0.5},
    "Mizoram": {"developed_ratio": 0.2, "developing_ratio": 0.6, "underdeveloped_ratio": 0.2},
    "Nagaland": {"developed_ratio": 0.1, "developing_ratio": 0.5, "underdeveloped_ratio": 0.4},
    "Odisha": {"developed_ratio": 0.2, "developing_ratio": 0.5, "underdeveloped_ratio": 0.3},
    "Punjab": {"developed_ratio": 0.6, "developing_ratio": 0.35, "underdeveloped_ratio": 0.05},
    "Rajasthan": {"developed_ratio": 0.25, "developing_ratio": 0.5, "underdeveloped_ratio": 0.25},
    "Sikkim": {"developed_ratio": 0.3, "developing_ratio": 0.6, "underdeveloped_ratio": 0.1},
    "Tamil Nadu": {"developed_ratio": 0.6, "developing_ratio": 0.35, "underdeveloped_ratio": 0.05},
    "Telangana": {"developed_ratio": 0.4, "developing_ratio": 0.5, "underdeveloped_ratio": 0.1},
    "Tripura": {"developed_ratio": 0.2, "developing_ratio": 0.6, "underdeveloped_ratio": 0.2},
    "Uttar Pradesh": {"developed_ratio": 0.2, "developing_ratio": 0.45, "underdeveloped_ratio": 0.35},
    "Uttarakhand": {"developed_ratio": 0.3, "developing_ratio": 0.5, "underdeveloped_ratio": 0.2},
    "West Bengal": {"developed_ratio": 0.35, "developing_ratio": 0.45, "underdeveloped_ratio": 0.2},
    "Delhi": {"developed_ratio": 0.95, "developing_ratio": 0.05, "underdeveloped_ratio": 0.0},
    "Jammu & Kashmir": {"developed_ratio": 0.2, "developing_ratio": 0.5, "underdeveloped_ratio": 0.3},
    "Ladakh": {"developed_ratio": 0.1, "developing_ratio": 0.6, "underdeveloped_ratio": 0.3},
    "Puducherry": {"developed_ratio": 0.8, "developing_ratio": 0.2, "underdeveloped_ratio": 0.0},
    "Chandigarh": {"developed_ratio": 0.95, "developing_ratio": 0.05, "underdeveloped_ratio": 0.0},
    "Andaman & Nicobar Islands": {"developed_ratio": 0.3, "developing_ratio": 0.6, "underdeveloped_ratio": 0.1},
    "Dadra & Nagar Haveli and Daman & Diu": {"developed_ratio": 0.6, "developing_ratio": 0.4, "underdeveloped_ratio": 0.0},
    "Lakshadweep": {"developed_ratio": 0.4, "developing_ratio": 0.6, "underdeveloped_ratio": 0.0}
}

def generate_district_profile_data(tier):
    if tier == "underdeveloped":
        electricity = np.clip(np.random.normal(loc=55.0, scale=12.0), 10.0, 85.0)
        drinking_water = np.clip(np.random.normal(loc=65.0, scale=10.0), 25.0, 90.0)
        computers = np.clip(np.random.normal(loc=12.0, scale=6.0), 1.0, 30.0)
        internet = np.clip(np.random.normal(loc=8.0, scale=5.0), 0.5, 25.0)
        str_ratio = np.clip(np.random.normal(loc=42.0, scale=6.0), 30.0, 65.0)
    elif tier == "developing":
        electricity = np.clip(np.random.normal(loc=88.0, scale=6.0), 70.0, 98.0)
        drinking_water = np.clip(np.random.normal(loc=90.0, scale=5.0), 75.0, 100.0)
        computers = np.clip(np.random.normal(loc=45.0, scale=10.0), 20.0, 75.0)
        internet = np.clip(np.random.normal(loc=35.0, scale=10.0), 15.0, 65.0)
        str_ratio = np.clip(np.random.normal(loc=28.0, scale=4.0), 20.0, 38.0)
    else:  # developed
        electricity = np.clip(np.random.normal(loc=98.5, scale=1.5), 90.0, 100.0)
        drinking_water = np.clip(np.random.normal(loc=99.0, scale=1.0), 92.0, 100.0)
        computers = np.clip(np.random.normal(loc=85.0, scale=8.0), 60.0, 100.0)
        internet = np.clip(np.random.normal(loc=80.0, scale=9.0), 55.0, 100.0)
        str_ratio = np.clip(np.random.normal(loc=18.0, scale=3.0), 8.0, 25.0)
        
    total_schools = int(np.clip(np.random.normal(loc=1200, scale=400), 150, 4500))
    avg_students_per_school = int(np.clip(np.random.normal(loc=120, scale=30), 40, 300))
    total_students = total_schools * avg_students_per_school
    total_teachers = int(total_students / str_ratio)
    actual_str = round(total_students / max(total_teachers, 1), 1)

    return {
        "electricity_perc": round(electricity, 2),
        "drinking_water_perc": round(drinking_water, 2),
        "computer_perc": round(computers, 2),
        "internet_perc": round(internet, 2),
        "student_teacher_ratio": actual_str,
        "total_schools": total_schools,
        "total_students": total_students,
        "total_teachers": total_teachers
    }

def main():
    print("Starting mock data generation...")
    records = []
    
    for state, districts in STATE_TO_DISTRICTS.items():
        count = len(districts)
        
        # Get profile ratios for state
        profile = STATES_PROFILES.get(state, {"developed_ratio": 0.3, "developing_ratio": 0.5, "underdeveloped_ratio": 0.2})
        
        dev_count = int(count * profile["developed_ratio"])
        undev_count = int(count * profile["underdeveloped_ratio"])
        devp_count = count - dev_count - undev_count
        
        dist_tiers = (["developed"] * dev_count) + (["developing"] * devp_count) + (["underdeveloped"] * undev_count)
        # Shuffle tiers so that underdeveloped/developed status is distributed randomly among the districts of a state
        np.random.shuffle(dist_tiers)
        
        for i, district in enumerate(districts):
            tier = dist_tiers[i]
            metrics = generate_district_profile_data(tier)
            metrics["state"] = state
            metrics["district"] = district
            records.append(metrics)
            
    df = pd.DataFrame(records)
    cols = ["state", "district", "electricity_perc", "drinking_water_perc", "computer_perc", "internet_perc", 
            "student_teacher_ratio", "total_schools", "total_students", "total_teachers"]
    df = df[cols]
    
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "districts_raw.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Generated data for {len(df)} districts across {len(STATE_TO_DISTRICTS)} states/UTs.")
    print(f"File saved to: {output_path}")

if __name__ == "__main__":
    main()
