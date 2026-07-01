import pandas as pd

df = pd.read_csv("classroom_energy_dataset.csv")
df.columns = df.columns.str.strip()

def classify_occupancy(count):
    if count == 0:
        return "Empty"
    elif count <= 2:
        return "Low Occupancy"
    elif count <= 5:
        return "Medium Occupancy"
    else:
        return "High Occupancy"

def recommend_room(status):
    if status == "Empty":
        return "No classroom needed / keep appliances OFF"
    elif status == "Low Occupancy":
        return "Move to Small Classroom tomorrow"
    elif status == "Medium Occupancy":
        return "Use Small or Medium Classroom tomorrow"
    else:
        return "Keep Same Classroom"

df["occupancy_status"] = df["person_count"].apply(classify_occupancy)
df["room_recommendation"] = df["occupancy_status"].apply(recommend_room)

df.to_csv("classroom_final_dataset.csv", index=False)

print("Created classroom_final_dataset.csv")
print(df[["time", "person_count", "occupancy_status", "room_recommendation"]].head(20))