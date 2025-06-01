import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

# Read FSR sensor data from CSV
df = pd.read_csv("new_gait_window.csv")

# Convert a row of FSR readings into a 2x2 grid
def get_pressure_grid(row):
    return np.array([
        [row['fsr2'], row['fsr3']],  # Mid Left & Mid Right
        [row['fsr1'], row['fsr4']]   # Heel & Toes
    ])

# Set up the plot
fig, ax = plt.subplots()

# Display the first frame of heatmap
heatmap = ax.imshow(
    get_pressure_grid(df.iloc[0]),
    cmap='jet',
    interpolation='nearest',
    vmin=0,
    vmax=df[['fsr1','fsr2','fsr3','fsr4']].values.max()
)

# Load right foot outline image with transparency
foot_img = mpimg.imread('foot_right.png')

# Position the image to match the heatmap grid (adjust as needed)
img_extent = [-0.5, 1.5, 1.5, -0.5]
img_plot = ax.imshow(foot_img, extent=img_extent, alpha=0.5)

# Add labels and styling
plt.colorbar(heatmap, ax=ax, label='Pressure (FSR)')
plt.title("Foot Sole Pressure Animation with Right Foot Outline")
plt.xticks([0, 1], ['Mid Left', 'Mid Right'])
plt.yticks([0, 1], ['Heel', 'Toes'])
ax.invert_yaxis()  # Show heel at bottom

# Animation update function
def update(frame):
    row = df.iloc[frame]
    heatmap.set_data(get_pressure_grid(row))
    return [heatmap, img_plot]

# Run the animation
ani = FuncAnimation(fig, update, frames=len(df), interval=500, blit=True)
plt.show()