import pandas as pd

# Read your corrected CSV
df = pd.read_csv("crowd_counts.csv")

# If your column is named differently, change final_count below
# Example required columns: image, time, yolo_count, final_count

def time_to_minutes(t):
    t = str(t).strip().upper()
    time_part, ampm = t.split()

    hour, minute = time_part.split(".")
    hour = int(hour)
    minute = int(minute)

    if ampm == "PM" and hour != 12:
        hour += 12

    if ampm == "AM" and hour == 12:
        hour = 0

    return hour * 60 + minute


def crowd_level(count):
    if count == 0:
        return "Empty"
    elif count <= 10:
        return "Less Crowded"
    elif count <= 29:
        return "Moderate"
    elif count <= 50:
        return "Crowded"
    else:
        return "Very Crowded"


def mess_session(minutes):
    if 7*60 <= minutes <= 9*60:
        return "Breakfast"
    elif 12*60 <= minutes <= 14*60:
        return "Lunch"
    elif 19*60 <= minutes <= 21*60:
        return "Dinner"
    else:
        return "Closed/Empty Time"


df["minutes"] = df["time"].apply(time_to_minutes)
df["crowd_level"] = df["final_count"].apply(crowd_level)
df["mess_session"] = df["minutes"].apply(mess_session)

df = df.sort_values("minutes")

df.to_csv("prepared_crowd_dataset.csv", index=False)

print("Created prepared_crowd_dataset.csv")
print(df.head(10))