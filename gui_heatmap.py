import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from PIL import Image

class FootPressureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foot Pressure Heatmap Overlay")
        self.setGeometry(100, 100, 600, 800)

        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        self.fig = Figure(figsize=(5, 7))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Load data and image
        self.data = pd.read_csv('new_gait_window.csv')
        self.foot_img = Image.open('foot_right.png')

        self.current_index = 0  # to iterate rows if nak animasi

        self.plot_heatmap()

    def plot_heatmap(self):
        self.fig.clf()  # Clear figure
        ax = self.fig.add_subplot(111)

        # Plot foot image as background
        ax.imshow(self.foot_img, extent=[0, 5, 0, 10], aspect='auto')

        # Ambil nilai fsr dari current row
        row = self.data.iloc[self.current_index]
        fsr_values = [row['fsr1'], row['fsr2'], row['fsr3'], row['fsr4']]

        # Amplify nilai tekanan supaya warna jelas (cuba 5 kali)
        fsr_values = [v * 5 for v in fsr_values]

        # Susun pressure ke grid 3x3 (nilai 0 untuk tempat kosong)
        pressure_grid = np.array([
            [0,          fsr_values[0], 0],
            [fsr_values[1], 0,          fsr_values[2]],
            [0,          fsr_values[3], 0]
        ])

        # Normalize ikut nilai tekanan baru
        norm = Normalize(vmin=0, vmax=max(fsr_values))

        # Overlay heatmap pressure dengan alpha pekat
        heatmap = ax.imshow(pressure_grid, cmap='jet', alpha=0.8,
                            extent=[0.8, 4.2, 1.5, 8.5], interpolation='nearest',
                            norm=norm)

        # Colorbar
        self.fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label='Pressure')

        ax.axis('off')
        ax.set_title(f'Foot Pressure Heatmap (Row {self.current_index+1})')

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FootPressureApp()
    window.show()
    sys.exit(app.exec_())