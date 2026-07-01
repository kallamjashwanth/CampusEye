import streamlit as st
import pandas as pd
import os
from datetime import time
import matplotlib.pyplot as plt
from smart_dustbin.read_live_fill import read_current_fill
st.set_page_config(
    page_title="Smart Campus Monitoring & Management System",
    layout="wide"
)

st.sidebar.title("Smart Campus Modules")

module = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Smart Classroom",
        "Smart Mess",
        "Smart Dustbin"
    ]
)

# ---------------- HOME ----------------

if module == "Home":
    st.title("Smart Campus Monitoring & Management System")

    st.markdown("""
    ### Welcome to the Smart Campus Dashboard

    This system helps campus users make better decisions by showing real-time and predicted information about classroom usage, mess crowd levels, and dustbin collection planning.

    ---
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Smart Classroom")
        st.write("""
        Helps in reducing unnecessary electricity usage in classrooms.

        **What it shows:**
        - Whether the classroom is occupied or empty
        - Light and fan status
        - Energy saved during the day
        - Future energy-saving prediction
        - Recommended time slots to shift to a smaller classroom
        """)

    with col2:
        st.subheader("Smart Mess")
        st.write("""
        Helps students choose a better time to visit the mess.

        **What it shows:**
        - Current mess crowd level
        - Queue waiting time
        - Best time to visit
        - Future crowd forecast
        - Mess opening and closing alerts
        """)

    with col3:
        st.subheader("Smart Dustbin")
        st.write("""
        Helps campus staff plan waste collection efficiently.

        **What it shows:**
        - Live dustbin fill percentage
        - Predicted future fill level
        - Days remaining until full
        - Collection priority
        - Recommended collection action
        """)

    st.markdown("---")

    st.subheader("Project Goal")

    st.success("""
    To reduce crowding, save energy, improve waste collection, and optimize campus resources using smart prediction and monitoring.
    """)

    st.subheader("How to Use")

    st.write("""
    Use the sidebar to open:
    - **Smart Classroom** for classroom energy and room allocation insights
    - **Smart Mess** for crowd forecast and best visit time
    - **Smart Dustbin** for live fill monitoring and collection prediction
    """)

# ---------------- SMART CLASSROOM ----------------

elif module == "Smart Classroom":
    st.title("AI-Based Smart Classroom Energy Management System")

    df = pd.read_csv("smart_classroom/dataset/classroom_final_dataset.csv")
    future = pd.read_csv("smart_classroom/dataset/future_energy_prediction.csv")

    df.columns = df.columns.str.strip()
    future.columns = future.columns.str.strip()

    total_energy_saved = df["energy_saved"].sum()
    total_energy_used = df["energy_used"].sum()
    avg_occupancy = round(df["person_count"].mean(), 2)
    tomorrow_saving = future["predicted_energy_saved"].sum()
    weekly_saving = tomorrow_saving * 7

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Occupancy", avg_occupancy)
    col2.metric("Today's Energy Saved", f"{total_energy_saved:.2f} Wh")
    col3.metric("Tomorrow Prediction", f"{tomorrow_saving:.2f} Wh")
    col4.metric("Weekly Prediction", f"{weekly_saving / 1000:.3f} kWh")

    st.subheader("AI Video Detection Output")

    videos = {
        "Empty Classroom": "smart_classroom/videos/iot_empty_classroom_web.mp4",
        "Students Entering": "smart_classroom/videos/iot_output_students_entering_web.mp4",
        "Students Leaving": "smart_classroom/videos/iot_output_students_leaving_web.mp4"
    }

    choice = st.selectbox("Select Demo Video", list(videos.keys()))
    video_path = videos[choice]

    if os.path.exists(video_path):
        with open(video_path, "rb") as video_file:
            st.video(video_file.read())
    else:
        st.warning(f"Video file not found: {video_path}")

    st.subheader("Classroom Occupancy and Energy Analysis")

    df_plot = df.copy()
    df_plot["time"] = df_plot["time"].astype(str)

    st.write("### Occupancy Trend")
    st.line_chart(df_plot.set_index("time")[["person_count"]])

    st.write("### Temperature and Humidity Trend")
    st.line_chart(df_plot.set_index("time")[["temperature", "humidity"]])

    st.write("### Energy Used vs Energy Saved")
    st.bar_chart(df_plot[["energy_used", "energy_saved"]])

    st.subheader("Future Energy Saving Prediction")
    st.line_chart(future[["predicted_energy_saved"]])

    st.subheader("Smart Classroom Allocation Recommendation")

    df_rec = df.copy()
    df_rec["time"] = pd.to_datetime(df_rec["time"], format="%H:%M", errors="coerce")

    df_rec["occupancy_status"] = df_rec["person_count"].apply(
        lambda x: "Empty" if x == 0
        else "Low" if x <= 2
        else "Medium" if x <= 5
        else "High"
    )

    current_status = None
    start_time = None
    found_recommendation = False

    for i, row in df_rec.iterrows():
        status = row["occupancy_status"]

        if current_status is None:
            current_status = status
            start_time = row["time"]

        elif status != current_status:
            end_time = df_rec.iloc[i - 1]["time"]
            duration_minutes = (end_time - start_time).total_seconds() / 60

            if current_status in ["Low", "Medium"] and duration_minutes >= 45:
                recommendation = (
                    "Move to Small Classroom"
                    if current_status == "Low"
                    else "Move to Medium Classroom"
                )

                st.warning(
                    f"{start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')} "
                    f"({int(duration_minutes)} min): {recommendation}"
                )
                found_recommendation = True

            current_status = status
            start_time = row["time"]

    end_time = df_rec.iloc[-1]["time"]
    duration_minutes = (end_time - start_time).total_seconds() / 60

    if current_status in ["Low", "Medium"] and duration_minutes >= 45:
        recommendation = (
            "Move to Small Classroom"
            if current_status == "Low"
            else "Move to Medium Classroom"
        )

        st.warning(
            f"{start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')} "
            f"({int(duration_minutes)} min): {recommendation}"
        )
        found_recommendation = True

    if not found_recommendation:
        st.success("No classroom change required for long duration. Current classroom allocation is suitable.")

# ---------------- SMART MESS ----------------

elif module == "Smart Mess":
    st.title("AI-Powered Smart Mess Crowd Forecasting System")

    df = pd.read_csv("mess_project/timeseries_dataset.csv")

    trend = df.groupby("minutes")["count"].mean().reset_index()
    trend = trend.sort_values("minutes")

    def minutes_to_label(m):
        m = m % (24 * 60)
        h = int(m // 60)
        minute = int(m % 60)
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return f"{h12}:{minute:02d} {suffix}"

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

    def recommendation(level):
        if level == "Empty":
            return "Mess is almost empty. Best time to visit."
        elif level == "Less Crowded":
            return "Good time to visit mess."
        elif level == "Moderate":
            return "Crowd is manageable. You can go now."
        elif level == "Crowded":
            return "Mess is crowded. Better to wait if possible."
        else:
            return "Mess is very crowded. Prefer going later."

    def get_count_for_time(input_minutes):
        trend["diff"] = abs(trend["minutes"] - input_minutes)
        nearest_row = trend.loc[trend["diff"].idxmin()]
        return int(round(nearest_row["count"]))

    def next_mess_opening(input_minutes):
        timings = [
            ("Breakfast", 7 * 60, 9 * 60),
            ("Lunch", 12 * 60, 14 * 60),
            ("Dinner", 19 * 60, 21 * 60)
        ]

        for name, start, end in timings:
            if input_minutes < start:
                return name, start, end

        return "Breakfast", 7 * 60, 9 * 60

    def find_best_time_after(input_minutes):
        timings = [
            ("Breakfast", 7 * 60, 9 * 60),
            ("Lunch", 12 * 60, 14 * 60),
            ("Dinner", 19 * 60, 21 * 60)
        ]

        for name, start, end in timings:
            if start <= input_minutes <= end:
                future_times = trend[
                    (trend["minutes"] >= input_minutes) &
                    (trend["minutes"] <= end)
                ].copy()

                if future_times.empty:
                    return f"{minutes_to_label(input_minutes)}\nGo now"

                good_times = future_times[future_times["count"] <= 29]

                if not good_times.empty:
                    best = good_times.iloc[0]
                else:
                    best = future_times.loc[future_times["count"].idxmin()]

                wait_minutes = int(best["minutes"] - input_minutes)

                if wait_minutes <= 0:
                    return f"{minutes_to_label(input_minutes)}\nGo now"
                else:
                    return f"{minutes_to_label(best['minutes'])}\nWait about {wait_minutes} min"

        for name, start, end in timings:
            if input_minutes < start:
                return f"{minutes_to_label(start)}\n{name} opens"

        return "7:00 AM\nBreakfast opens tomorrow"

    def estimate_queue_time(count):
        if count <= 30:
            return "No wait time"
        elif count <= 50:
            return "Around 5 minutes"
        elif count <= 80:
            return "Around 10 minutes"
        else:
            return "Around 15 minutes"
        
    day_type = st.sidebar.selectbox(
        "Select Day Type",
        ["Weekday", "Weekend", "Event Day", "Rainy Day"]
    )

    def mess_timing_message(input_minutes):
        timings = [
            ("Breakfast", 7 * 60, 9 * 60),
            ("Lunch", 12 * 60, 14 * 60),
            ("Dinner", 19 * 60, 21 * 60)
        ]
    
        for name, start, end in timings:
            if start - 15 <= input_minutes < start:
                mins_left = start - input_minutes
                return f"{name} will open in {mins_left} minutes."

            if end - 20 <= input_minutes <= end:
                mins_left = end - input_minutes
                return f"Hurry up! {name} will close in {mins_left} minutes."

            if start <= input_minutes <= end:
                return f"{name} is currently open."

        for name, start, end in timings:
            if input_minutes < start:
                return (
                    f"Mess is closed now. Next timing: {name} "
                    f"from {minutes_to_label(start)} to {minutes_to_label(end)}."
                )

        return "Mess is closed now. Next timing: Breakfast from 7:00 AM to 9:00 AM tomorrow."

    st.sidebar.header("Mess User Input")

    selected_time = st.sidebar.time_input(
        "Enter time",
        value=time(12, 30)
    )

    input_minutes = selected_time.hour * 60 + selected_time.minute

    current_count = get_count_for_time(input_minutes)

    if day_type == "Weekend":
        current_count = int(current_count * 0.75)
    elif day_type == "Event Day":
        current_count = int(current_count * 1.35)
    elif day_type == "Rainy Day":
        current_count = int(current_count * 0.85)

    queue_time = estimate_queue_time(current_count)
    current_level = crowd_level(current_count)
    current_rec = recommendation(current_level)
    timing_msg = mess_timing_message(input_minutes)

    def adjust_by_day_type(count, day_type):
        if day_type == "Weekend":
            return int(count * 0.75)
        elif day_type == "Event Day":
            return int(count * 1.35)
        elif day_type == "Rainy Day":
            return int(count * 0.85)
        else:
            return int(count)

    current_count = adjust_by_day_type(get_count_for_time(input_minutes), day_type)
    future_15 = adjust_by_day_type(get_count_for_time(input_minutes + 15), day_type)
    future_30 = adjust_by_day_type(get_count_for_time(input_minutes + 30), day_type)
    future_45 = adjust_by_day_type(get_count_for_time(input_minutes + 45), day_type)
    future_60 = adjust_by_day_type(get_count_for_time(input_minutes + 60), day_type)

    best_time = find_best_time_after(input_minutes)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected Time", minutes_to_label(input_minutes))
        st.info(timing_msg)

    with col2:
        st.metric("Current Crowd Level", current_level)

    with col3:
        st.write("**Best Time to Go**")
        st.info(best_time)

    st.subheader("Queue Information")

    if queue_time == "No wait time":
        st.success(f"Estimated Queue Time: {queue_time}")
    else:
        st.warning(f"Estimated Queue Time: {queue_time}")

    st.subheader("Smart Suggestions")

    if "closed now" in timing_msg:
        st.info(timing_msg)
    elif "open in" in timing_msg:
        st.warning(timing_msg)
    elif "close in" in timing_msg:
        st.error(timing_msg)
    elif current_level == "Very Crowded":
        st.error(f"Current crowd is {current_level}. Better to wait before visiting.")
    elif current_level == "Crowded":
        st.warning(f"Current crowd is {current_level}. Waiting is recommended.")
    else:
        st.success(f"Current crowd is {current_level}. Good time to visit.")

    st.subheader("Future Crowd Forecast")

    forecast_df = pd.DataFrame({
        "Time": [
            minutes_to_label(input_minutes),
            minutes_to_label(input_minutes + 15),
            minutes_to_label(input_minutes + 30),
            minutes_to_label(input_minutes + 45),
            minutes_to_label(input_minutes + 60)
        ],
        "Predicted Count": [
            current_count,
            future_15,
            future_30,
            future_45,
            future_60
        ]
    })

    forecast_df["Crowd Level"] = forecast_df["Predicted Count"].apply(crowd_level)

    st.dataframe(forecast_df)

    st.line_chart(
        forecast_df.set_index("Time")["Predicted Count"]
    )

    st.subheader("One Day Mess Crowd Trend")

    trend_plot = trend.copy()
    trend_plot = trend_plot.sort_values("minutes")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        trend_plot["minutes"],
        trend_plot["count"]
    )

    ax.set_xticks(trend_plot["minutes"][::5])
    ax.set_xticklabels(
        trend_plot["minutes"][::5].apply(minutes_to_label),
        rotation=45
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Crowd Count")
    ax.set_title("One Day Mess Crowd Trend")

    st.pyplot(fig)

elif module == "Smart Dustbin":
    
    st.title("AI-Based Smart Dustbin Fill Prediction System")

    import joblib
    from smart_dustbin.read_live_fill import read_current_fill

    dustbin_df = pd.read_csv("smart_dustbin/dustbin_30day_dataset.csv")
    model = joblib.load("smart_dustbin/dustbin_stack_model.pkl")

    dustbin_df.columns = dustbin_df.columns.str.strip()
    dustbin_df = dustbin_df.loc[:, ~dustbin_df.columns.str.contains("Unnamed")]

    slot_order_map = {
        "Morning": 0,
        "Afternoon": 1,
        "Evening": 2,
        "Night": 3
    }

    reverse_slot_map = {
        0: "Morning",
        1: "Afternoon",
        2: "Evening",
        3: "Night"
    }

    dustbin_df["slot_order"] = dustbin_df["time_slot"].map(slot_order_map)
    dustbin_df = dustbin_df.sort_values(
        ["bin_name", "day", "slot_order"]
    ).reset_index(drop=True)

    dustbin_df["prev_fill"] = dustbin_df.groupby("bin_name")["fill_percentage"].shift(1)
    dustbin_df["fill_rate"] = dustbin_df["fill_percentage"] - dustbin_df["prev_fill"]
    dustbin_df = dustbin_df.dropna()

    st.subheader("Live Dustbin Reading")

    selected_bin = st.selectbox(
        "Select Dustbin",
        dustbin_df["bin_name"].unique()
    )

    selected_slot = st.selectbox(
        "Select Current Time Slot",
        ["Morning", "Afternoon", "Evening", "Night"]
        )
    
    com_port = st.text_input("ESP32 COM Port", "COM3")

    if st.button("Read Current Fill from Sensor"):
        live_fill = read_current_fill(com_port)

        if live_fill is not None:
            st.session_state["live_fill"] = live_fill
            st.success(f"Live Sensor Fill: {live_fill:.2f}%")
        else:
            st.error("Could not read sensor value. Check COM port and close Arduino Serial Monitor.")

    bin_history = dustbin_df[dustbin_df["bin_name"] == selected_bin].copy()
    latest = bin_history.iloc[-1]

    display_current_fill = st.session_state.get(
        "live_fill",
        float(latest["fill_percentage"])
    )

    current_fill = display_current_fill
    prev_fill = float(latest["fill_percentage"])
    fill_rate = current_fill - float(latest["prev_fill"])

    if fill_rate <= 0:
        positive_rates = bin_history[bin_history["fill_rate"] > 0]["fill_rate"]
        fill_rate = positive_rates.mean() if len(positive_rates) > 0 else 5

    location = latest["location"]
    current_day = int(latest["day"])

    st.subheader("Current Dustbin Status")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Bin", selected_bin)
    c2.metric("Location", location)
    c3.metric("Current Fill", f"{display_current_fill:.2f}%")
    c4.metric("Current Slot", selected_slot)

    st.subheader("Model-Based Future Fill Prediction")

    future_rows = []
    model_days_remaining = "-"

    start_slot = slot_order_map[selected_slot]

    for future_day in range(1, 31):
        actual_day = current_day + future_day

        if future_day == 1:
            slot_range = range(start_slot+1, 4)
        else:
            slot_range = range(0, 4)

        for slot_order in slot_range:
            X_future = pd.DataFrame([{
                "day": actual_day,
                "slot_order": slot_order,
                "bin_name": selected_bin,
                "location": location,
                "fill_percentage": current_fill,
                "prev_fill": prev_fill,
                "fill_rate": fill_rate
            }])

            predicted_fill = model.predict(X_future)[0]

            if predicted_fill < current_fill:
                predicted_fill = current_fill + max(fill_rate, 1)

            predicted_fill = min(predicted_fill, 100)

            new_rate = predicted_fill - current_fill
            prev_fill = current_fill
            current_fill = predicted_fill
            fill_rate = new_rate if new_rate > 0 else fill_rate

            if current_fill >= 90:
                row_priority = "Urgent"
            elif current_fill >= 75:
                row_priority = "High"
            elif current_fill >= 50:
                row_priority = "Medium"
            else:
                row_priority = "Low"

            if model_days_remaining == "-" and current_fill >= 100:
                model_days_remaining = future_day

            future_rows.append({
                "future_day": future_day,
                "time_slot": reverse_slot_map[slot_order],
                "predicted_fill": round(current_fill, 2),
                "priority": row_priority
            })

        if model_days_remaining != "-":
            break

    future_df_live = pd.DataFrame(future_rows)

    positive_rates = bin_history[bin_history["fill_rate"] > 0]["fill_rate"]
    avg_rate_per_slot = positive_rates.mean() if len(positive_rates) > 0 else 5
    avg_rate_per_day = avg_rate_per_slot * 4

    estimated_days = round((100 - display_current_fill) / avg_rate_per_day, 1)
    if estimated_days < 0:
        estimated_days = 0

    if display_current_fill >= 90 or estimated_days <= 0.5:
        latest_priority = "Urgent"
        collection_time = "Collect Immediately"
    elif display_current_fill >= 75 or estimated_days <= 1:
        latest_priority = "High"
        collection_time = "Collect Today"
    elif display_current_fill >= 50 or estimated_days <= 2:
        latest_priority = "Medium"
        collection_time = "Collect Tomorrow"
    else:
        latest_priority = "Low"
        collection_time = "No Immediate Action"

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Current Fill", f"{display_current_fill:.1f}%")
    k2.metric("Predicted Full In", f"{estimated_days} days")
    k3.metric("Priority", latest_priority)
    k4.metric("Action", collection_time)

    if latest_priority == "Urgent":
        st.error(f"{selected_bin} should be collected immediately or very soon.")
    elif latest_priority == "High":
        st.warning(f"{selected_bin} is filling quickly. Plan collection soon.")
    elif latest_priority == "Medium":
        st.info(f"{selected_bin} is moderately filled. Monitor regularly.")
    else:
        st.success(f"{selected_bin} does not require immediate collection.")


    st.subheader("Future Prediction Preview")
    st.dataframe(future_df_live.head(12))

    st.subheader("Historical Fill Trend")
    st.line_chart(
        bin_history.set_index("day")["fill_percentage"]
    )