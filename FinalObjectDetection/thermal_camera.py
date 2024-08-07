import numpy as np
import cv2
import adafruit_mlx90640
import busio
import board
import cmapy
import matplotlib.pyplot as plt
import time

class ThermalCamera:
    def __init__(self, colormap='inferno'):
        self.colormap = colormap
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
            self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
            self.frame = np.zeros((24 * 32,))
            self.emissivity = 0.95
            self._temp_min = 0
            self._temp_max = 0
            self.filter_image = False
            print("Thermal camera initialized successfully.")
        except Exception as e:
            print(f"Error initializing thermal camera: {e}")

    def set_colormap(self, colormap):
        self.colormap = colormap
        
    def toggle_filter(self):
        self.filter_image = not self.filter_image

    def get_frame(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                frame = np.zeros((24 * 32,))
                self.mlx.getFrame(frame)
                frame = frame.reshape((24, 32))
                self._temp_min = np.min(frame)
                self._temp_max = np.max(frame)
                print(f"Frame retrieved successfully on attempt {attempt + 1}")
                return frame
            except RuntimeError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Too many retries")

    def process_frame(self, frame):
        image = np.reshape(frame, (24, 32))
        image = np.uint8(np.interp(image, (image.min(), image.max()), (0, 255)))
        colormap_frame = cv2.applyColorMap(image, cmapy.cmap(self.colormap))
        if self.filter_image:
            colormap_frame = cv2.bilateralFilter(colormap_frame, 9, 75, 75)
        colormap_frame = cv2.resize(colormap_frame, (800, 600), interpolation=cv2.INTER_LINEAR)
        return colormap_frame
        
    def generate_legend(self, width=800, height=200):
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        cbar = fig.colorbar(ax.imshow(gradient, aspect='auto', cmap=self.colormap), ax=ax, orientation='horizontal')
        cbar.set_label('Temperature (°C)')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels([f'{self._temp_min:.1f}', '', '', '', f'{self._temp_max:.1f}'])
        ax.remove()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'static/colormap_legends/{self.colormap}_legend.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
