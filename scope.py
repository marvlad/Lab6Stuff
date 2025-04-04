# M. Ascencio-Sosa, ISU, Jun 2024
# get_corrected_wf function from A. Mastbaum U. Rutgers

from cProfile import label
from itertools import count
from re import I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
from matplotlib.colors import LogNorm
from tqdm import tqdm
import warnings
from matplotlib import MatplotlibDeprecationWarning
import os
import sys 
import concurrent.futures

# Suppress the specific MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

BASE_PATH = '/Users/marvinascenciososa/Desktop/mezzanine/data/'
ACDC_name = 'ACDC'
OUTPUT = './plots/'
REPORT = './report/'
PEDESTAL = 'pedestal'
CHANNELS = 'channels'


mezzanine_id = [24,25,26,27,28,29,18,19,20,21,22,23,12,13,14,15,16,17,6,7,8,9,10,11,0,1,2,3,4]

def create_latex_file(number, sections, plots_directory, output):
    content = f"""\\documentclass{{beamer}} 
\\usepackage{{hyperref}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{xcolor}}
\\usepackage{{pgffor}} 
\\begin{{document}} 
\\title{{ACDC {number} Report}} 
\\date{{\\today}} 
\\frame{{\\titlepage}}
\\begin{{frame}}{{Outline}}
    \\tableofcontents
\\end{{frame}}
\\def\\sections{{{','.join(map(str, range(29)))}}}
\\def\\eventdisplay{{{','.join(map(str, sections))}}}
\\section{{Total RMS and Average Min}}
\\begin{{frame}}{{ACDC {number}, Total RMS and avergage min}}
    \\includegraphics[scale=0.32]{{{plots_directory}/Total_RMS.pdf}}
    \\includegraphics[scale=0.32]{{{plots_directory}/Total_AvMin.pdf}}
\\end{{frame}}
\\foreach \\var in \\sections {{
    \\section{{Mezzanine channel \\var}}
    \\subsection{{RMS histograms}}
    \\begin{{frame}}{{RMS, Mezzanine channel \\var}}
        \\includegraphics[scale=0.21]{{{plots_directory}/RMS_ACDC_{number}_MCh_\\var.pdf}} 
    \\end{{frame}}
    \\subsection{{Average Min histograms}}
    \\begin{{frame}}{{Av Min, Mezzanine channel \\var}}
        \\includegraphics[scale=0.21]{{{plots_directory}/Min_ACDC_{number}_MCh_\\var.pdf}} 
    \\end{{frame}}
    \\subsection{{MultiEvent 1D display, 10 events per plot}}
    \\begin{{frame}}{{MultiEvents, Mezzanine channel \\var}}
        \\includegraphics[scale=0.21]{{{plots_directory}/MultiEvents_ACDC_{number}_MCh_\\var.pdf}} 
    \\end{{frame}}
    \\foreach \\ed in \\eventdisplay {{
        \\begin{{frame}}{{Mezzanine channel \\var, event \\ed}}
            \\includegraphics[scale=0.3]{{{plots_directory}/EventDisplay_\\ed_ACDC_{number}_MCh_\\var.pdf}} 
        \\end{{frame}}
    }}
}}
\\end{{document}}
"""
    with open(output, "w") as file:
        file.write(content)

def make_report(OUTPUT, ACDC_id):
    contents = os.listdir(OUTPUT)
    if not contents:
        print("The %s directory is empty, run declare the object ACDC_analysis and use get_plots." % OUTPUT)
        sys.exit()

    create_latex_file(ACDC_id, {1, 2, 3, 4, 5, 6}, "../plots","./report/report.tex")
    os.system(r"cd ./report && ls -lrt && pdflatex report.tex > /dev/null 2>&1 && find . -maxdepth 1 -name 'report.*' ! -name 'report.pdf' -exec rm {} \;")


def Plot1D(data, title, xl, yl, filename, show):
    plt.plot(data)
    plt.grid(True, linestyle='--')
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    if show == 1:
        plt.show()

    if show == 2:
        plt.savefig(filename, dpi=150)

    plt.close()
    return plt 

def Plot2D(data, title, xl, yl, zl, ed, filename):
    print("print...")
    if ed == 0:
        plt.imshow(data, cmap='viridis', interpolation='nearest')
    else:
        plt.imshow(data, cmap='viridis', aspect=7.0) 
    cbar = plt.colorbar()
    cbar.set_label(zl)
    plt.grid(True, linestyle='--')
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.savefig(filename)
    plt.close()
    #plt.show()

def Plot1D_hist(arr, nbins, title, xl, yl, filename, show):
    plt.hist(arr, bins=nbins, edgecolor='black')  # Adjust bins as needed
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.grid(True, linestyle='--')  # Optional grid
    if show == 1:
        plt.show()

    if show == 2:
        plt.savefig(filename)
    
    plt.close()
    return plt 

class ACDC:
    def __init__(self, pedestal_path, data_path)-> None:
        self.pedestal_path = pedestal_path
        self.data_path = data_path # ID on injected signal in the mezzanine phyical board
        self.data = self.get_corrected_wf()

    def get_corrected_wf(self):
        # Calculate pedestals
        pedestal_files = sorted(glob(self.pedestal_path + '/*.txt'))
        print(self.pedestal_path + '/*.txt')
        kwargs = {'header': None, 'delimiter': ' ', 'usecols': range(1,31)}
        d = np.vstack([pd.read_csv(x, **kwargs).values for x in pedestal_files[:1]])
        d = d.reshape(int(len(d) / 256), 256, 30) 
        peds = np.mean(d, axis=0) 

        # Channel data
        input_files = sorted(glob(self.data_path + '/*.txt' ))
        print(self.data_path + '/*.txt')
        d = pd.read_csv(input_files[0], **kwargs).values
        #print(input_files[self.channel])
        d.shape = (int(len(d) / 256), 256, 30) 

        # Pedestal and baseline subtraction (see notes below on algorithm)
        d = d - peds
        dr = d.reshape(d.shape[0], 16, 16, d.shape[2])
        idxe = np.repeat(np.arange(d.shape[0]), d.shape[2])
        idxc = np.tile(np.arange(d.shape[2]), d.shape[0])
        idxs = np.argmin(np.std(dr, axis=2), axis=1).flatten()
        ds = d - np.mean(dr[idxe, idxs, :, idxc], axis=1).reshape(d.shape[0], 1, d.shape[2])

        # Align the waveforms, approximately
        dst = ds[:,:,5].copy()
        dt = dst - np.roll(dst, 1, axis=1)
        dst[(dt<-10)] = 0 
        r = 128 - np.argmax(np.abs(dst - 0.35 * np.min(dst, axis=1)[:,np.newaxis]), axis=1)
        ri, ci = np.ogrid[:dst.shape[0], :dst.shape[1]]
        r[r<0] += dst.shape[1]
        ci = ci - r[:,np.newaxis]
        dss = ds[ri,ci]

        return dss 

    def plot_event(self, event_id):
        e_id = slice(event_id, event_id + 1)
        Plot2D(self.data[e_id,:,0:30].T, 'Event %i' % event_id, 'Sampling #', 'ACDC Channels', 'Amplitude [mV]', 1)
    
    def plot_event_ACDCchannel(self, event_id, acdc_channel, filename, show):
        e_id = slice(event_id, event_id + 1)
        Plot1D(self.data[e_id,:,acdc_channel].swapaxes(0,1), plt.title('Event %i' % event_id), 'Sampling #', 'Amplitude [mV]', filename, show) 
    
    def plot_events_ACDCchannel(self, event1, event2, acdc_channel, filename, show):
        e_id = slice(event1, event2)
        Plot1D(self.data[e_id,:,acdc_channel].swapaxes(0,1), 'ACDC %i, %i, Events from %i to %i' % (self.board, acdc_channel, event1,event2), 'Sampling #', 'Amplitude [mV]', filename, show)

    def event_display(self, event_id, filename, show_event):
        plt.switch_backend('Agg')
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) 

        # Plot for function 'a' on the first subplot
        e_id = slice(event_id, event_id + 1)
        im = ax1.imshow(self.data[e_id, :, 0:30].T, cmap='viridis', aspect=7.0)
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Amplitude [mV]')
        ax1.grid(True)
        ax1.set_title('Event %i' % event_id)
        ax1.set_xlabel('Sampling #')
        ax1.set_ylabel('ACDC Channels')

        # Plot for function 'b' on the second subplot
        for i in range(self.data.shape[2]):
            ax2.plot(self.data[e_id, :, i].swapaxes(0, 1), linewidth=4)
        ax2.grid(True, linestyle='--')
        ax2.set_title('Event %i' % event_id)
        ax2.set_xlabel('Sampling #')
        ax2.set_ylabel('Amplitude [mV]')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.style.use('dark_background')

        # Show the plot
        if show_event == 1:
            plt.show()

        if show_event == 2:
           plt.savefig(filename) 
        plt.close()
        return fig 
    
    def event_displaypdf(self, nevent, output_name):
        with PdfPages(output_name+".pdf") as pdf:
            progress_bar = tqdm(total=nevent, desc="Progress", unit="iteration")
            for i in range(nevent):
                fig = self.event_display(i, 'none', 0)
                pdf.savefig(fig)
                progress_bar.update(1)
            progress_bar.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:  # Check if there are fewer than 3 arguments
        print("Usage: python scope.py <ped_path> <data_path> <nevents>")
        print("Example: python3 scope.py /the/path/pedestal /the/path/data 100")
        print("WARNING: Do NOT add the files '.txt' only the paths")
        sys.exit(1)  # Exit with an error code
        
    ped_path = str(sys.argv[1])
    data_path = str(sys.argv[2])
    nevents = int(sys.argv[3])

    myACDC = ACDC(ped_path, data_path)
    myACDC.event_displaypdf(nevents, './scope')
    os.system("xdg-open ./scope.pdf")
