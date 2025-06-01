import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from PIL import Image

class FootPressureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foot Pressure Heatmap Overlay")
        self.setGeometry(100, 100, 600, 850)

        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        main_layout = QVBoxLayout()
        self.widget.setLayout(main_layout)

        # Matplotlib Figure and Canvas
        self.fig = Figure(figsize=(5, 7))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # Buttons layout
        btn_layout = QHBoxLayout()
        main_layout.addLayout(btn_layout)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)

        # Connect buttons
        self.prev_btn.clicked.connect(self.prev_row)
        self.next_btn.clicked.connect(self.next_row)

        # Load data and image
        self.data = pd.read_csv('new_gait_window.csv')
        self.foot_img = Image.open('foot_right.png')

        self.current_index = 0

        self.plot_heatmap()

    def plot_heatmap(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)

        # Background foot image
        ax.imshow(self.foot_img, extent=[0, 5, 0, 10], aspect='auto')

        # Get sensor values
        row = self.data.iloc[self.current_index]
        fsr_values = [row['fsr1'], row['fsr2'], row['fsr3'], row['fsr4']]

        # Clip negatives, amplify
        fsr_values = [max(0, v)*5 for v in fsr_values]

        pressure_grid = np.array([
            [0, fsr_values[0], 0],
            [fsr_values[1], 0, fsr_values[2]],
            [0, fsr_values[3], 0]
        ])

        norm = Normalize(vmin=0, vmax=max(fsr_values) if max(fsr_values)>0 else 1)

        heatmap = ax.imshow(pressure_grid, cmap='jet', alpha=0.8,
                            extent=[0.8, 4.2, 1.5, 8.5], interpolation='nearest',
                            norm=norm)

        self.fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label='Pressure')

        ax.axis('off')
        ax.set_title(f'Foot Pressure Heatmap (Row {self.current_index+1} of {len(self.data)})')

        self.canvas.draw()

    def next_row(self):
        if self.current_index < len(self.data)-1:
            self.current_index += 1
            self.plot_heatmap()

    def prev_row(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.plot_heatmap()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FootPressureApp()
    window.show()
    sys.exit(app.exec_())