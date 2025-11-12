import pandas as pd
import matplotlib.pyplot as plt

def analyze_erg(file_path):
    # Load the CSV
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns for consistency
    df.rename(columns={
        'time (s)': 'time',
        'pace (sec/500m)': 'pace',
        'watts': 'watts',
        'stroke rate': 'stroke_rate',
        'heart rate': 'heart_rate'
    }, inplace=True)

    print("Columns found:", df.columns.tolist())
    print("\n--- Workout Summary ---")

    # Core metrics
    avg_split = df["pace"].mean()
    avg_power = df["watts"].mean()
    consistency = df["watts"].std()
    drift = df["watts"].iloc[-1] - df["watts"].iloc[0]

    # Print summary
    print(f"Average Split: {avg_split:.2f} sec/500m")
    print(f"Average Power: {avg_power:.1f} W")
    print(f"Power Consistency (lower is better): {consistency:.2f}")
    print(f"Drift (change in watts from start to end): {drift:.1f}")

    if consistency < 10:
        print("✅ Very consistent pacing — excellent control.")
    elif consistency < 20:
        print("⚙️ Some variability — good, but could be smoother.")
    else:
        print("⚠️ High variability — work on steady power output.")

    if drift < 0:
        print("⬇️ Power faded — endurance training might help.")
    else:
        print("💡 Slight fade — try to pace more evenly next time.")

    # --- Visualization Section ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(df["time"], df["watts"], label="Watts (Power)", color="orange", linewidth=2)
    plt.plot(df["time"], df["pace"], label="Pace (sec/500m)", color="blue", linestyle="--", linewidth=2)

    if "heart_rate" in df.columns:
        plt.plot(df["time"], df["heart_rate"], label="Heart Rate (bpm)", color="red", alpha=0.6)

    plt.title("Erg Workout Analysis")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Performance Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig("workout_plot.png")
    print("\n📊 Plot saved as workout_plot.png in your erg-ai folder.")

if __name__ == "__main__":
    analyze_erg("workout.csv")
