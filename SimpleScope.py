#!/bin/python3

import numpy as np
import pandas as pd
import sys 
import matplotlib.pyplot as plt 
import os
from matplotlib.backends.backend_pdf import PdfPages

class SimpleACDC:
    def __init__(self, filepath, convert_to_voltage=False, baseline_correction=False):
        self.filepath = filepath
        self.convert_to_voltage = convert_to_voltage
        self.baseline_correction = baseline_correction
        self.data = self.process_waveform()

    def process_waveform(self):
        kwargs = {'header': None, 'delimiter': ' ', 'usecols': range(1, 31)}
        d = pd.read_csv(self.filepath, **kwargs).values
        d = d.reshape(int(len(d) / 256), 256, 30) 
        baseline = np.mean(d[:, :10, :], axis=1)
        if self.baseline_correction:
            corrected_data = d - baseline[:, np.newaxis, :]  
        else:
            corrected_data = d 
        if self.convert_to_voltage:
            corrected_data = corrected_data * 0.3 
        return corrected_data

    def event_display(self, event_id, filename, show_event):
        plt.switch_backend('Agg')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) 

        # Heatmap plot
        e_id = slice(event_id, event_id + 1)
        im = ax1.imshow(self.data[e_id, :, 0:30].T, cmap='viridis', aspect=7.0)
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Amplitude [mV]' if self.convert_to_voltage else 'Amplitude [ADC counts]')
        ax1.grid(True)
        ax1.set_title(f'Event {event_id}')
        ax1.set_xlabel('Sampling #')
        ax1.set_ylabel('Channels')

        # Waveform plot
        for i in range(self.data.shape[2]):
            ax2.plot(self.data[e_id, :, i].swapaxes(0, 1), linewidth=2)
        ax2.grid(True, linestyle='--')
        ax2.set_title(f'Event {event_id}')
        ax2.set_xlabel('Sampling #')
        ax2.set_ylabel('Amplitude [mV]' if self.convert_to_voltage else 'Amplitude [ADC counts]')

        plt.tight_layout()
        plt.style.use('dark_background')

        if show_event == 1:
            plt.show()
        if show_event == 2 and filename:
            plt.savefig(filename)
        plt.close()
        return fig 

    def event_display_pdf(self, nevent, output_name):
        with PdfPages(output_name + ".pdf") as pdf:
            for i in range(min(nevent, self.data.shape[0])):  # Don't exceed available events
                fig = self.event_display(i, None, 0)
                pdf.savefig(fig)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python SimpleScope.py /path/to/data.txt [--voltage] [--baseline] [--plot NEVENT]")
        print("  NEVENT: number of events to plot (required with --plot)")
        sys.exit(1)

    filepath = sys.argv[1]
    convert_to_voltage = "--voltage" in sys.argv
    baseline_correction = "--baseline" in sys.argv

    processor = SimpleACDC(filepath, convert_to_voltage=convert_to_voltage, baseline_correction=baseline_correction)
    #print("Data shape:", processor.data.shape)
    #print("Sample of first event (first 5 samples, first 5 channels):")
    #print(processor.data[0, :5, :5])

    # Handle plotting if requested
    if "--plot" in sys.argv:
        try:
            plot_idx = sys.argv.index("--plot")
            if plot_idx + 1 >= len(sys.argv):
                print("Error: --plot requires NEVENT")
                print("Example: python SimpleScope.py data.txt --plot 5")
                sys.exit(1)
    
            nevent = int(sys.argv[plot_idx + 1]) 
            output_name = "output"
    
            if nevent <= 0:
                print("Error: NEVENT must be a positive integer")
                sys.exit(1)
    
            processor.event_display_pdf(nevent, output_name)
            #os.system("xdg-open ./output.pdf") 
            os.system("open ./output.pdf") 
            print(f"Saved PDF with {min(nevent, processor.data.shape[0])} events to {output_name}.pdf")
        except ValueError:
            print("Error: NEVENT must be an integer")
            print("Example: python SimpleScope.py data.txt --plot 5")
            sys.exit(1)
~                          
