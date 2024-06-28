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

BASE_PATH = '/data/'
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
    def __init__(self, board, channel)-> None:
        self.board = board
        self.channel = channel # ID on injected signal in the mezzanine phyical board
        self.data = self.get_corrected_wf()

    def get_corrected_wf(self):
        global BASE_PATH, ACDC_name, PEDESTAL
        if self.board < 10:
            ACDC_name = 'ACDC0'

        ACDC_PATH = BASE_PATH + ACDC_name + str(self.board)

        try:
            cnt = os.listdir(ACDC_PATH)
            print(f"Contents of the directory: {cnt}")

            if 'peds' in cnt:
                PEDESTAL = 'peds'
            elif 'pedestal' in cnt:
                PEDESTAL = 'pedestal'

            print(f"PEDESTAL variable is set to: {PEDESTAL}")
            
        except FileNotFoundError:
            print(f"The directory {ACDC_PATH} does not exist.")

        print("Loadding data from %s, ACDC %s, channel %s " % (BASE_PATH, self.board, self.channel)) 

        # Calculate pedestals
        pedestal_files = sorted(glob(BASE_PATH + '%s%i/%s/*.txt' % (ACDC_name, self.board, PEDESTAL)))
        print(BASE_PATH + '%s%i/%s/*.txt' % (ACDC_name, self.board, PEDESTAL))
        kwargs = {'header': None, 'delimiter': ' ', 'usecols': range(1,31)}
        d = np.vstack([pd.read_csv(x, **kwargs).values for x in pedestal_files[:1]])
        d = d.reshape(int(len(d) / 256), 256, 30)
        peds = np.mean(d, axis=0) 

        # Channel data
        input_files = sorted(glob(BASE_PATH + '%s%i/%s/*.txt' % (ACDC_name,self.board, CHANNELS)))
        print(BASE_PATH + '%s%i/%s/*.txt' % (ACDC_name,self.board, CHANNELS))
        d = pd.read_csv(input_files[self.channel], **kwargs).values
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
        Plot2D(self.data[e_id,:,0:30].T, 'Event %i' % event_id, 'Time [ns]', 'ACDC Channels', 'Amplitude [mV]', 1)
    
    def plot_event_ACDCchannel(self, event_id, acdc_channel, filename, show):
        e_id = slice(event_id, event_id + 1)
        Plot1D(self.data[e_id,:,acdc_channel].swapaxes(0,1), plt.title('Event %i' % event_id), 'Time [ns]', 'Amplitude [mV]', filename, show) 
     
    def plot_events_ACDCchannel(self, event1, event2, acdc_channel, filename, show):
        e_id = slice(event1, event2)
        Plot1D(self.data[e_id,:,acdc_channel].swapaxes(0,1), 'ACDC %i, %i, Events from %i to %i' % (self.board, acdc_channel, event1,event2), 'Time [ns]', 'Amplitude [mV]', filename, show)

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
        ax1.set_xlabel('Time [ns]')
        ax1.set_ylabel('ACDC Channels')

        # Plot for function 'b' on the second subplot
        for i in range(self.data.shape[2]):
            ax2.plot(self.data[e_id, :, i].swapaxes(0, 1))
        ax2.grid(True, linestyle='--')
        ax2.set_title('Event %i' % event_id)
        ax2.set_xlabel('Time [ns]')
        ax2.set_ylabel('Amplitude [mV]')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        if show_event == 1:
            plt.show()

        if show_event == 2:
           plt.savefig(filename) 
        plt.close()
        return fig
    
    def event_displaypdf(self, nevent, output_name):
        with pdfPages(output_name+".pdf") as pdf:
            progress_bar = tqdm(total=nevent, desc="Progress", unit="iteration")
            for i in range(nevent):
                fig = self.event_display(i, 'none', 0)
                pdf.savefig(fig)
                progress_bar.update(1)
            progress_bar.close()

    def get_RMS_channel(self, channel_id):
        #rms = np.sqrt(np.mean(np.square(self.data[e_id, :, 24])))
        rms = np.sqrt(np.mean(np.square(self.data[:, :, channel_id])))
        #print(rms)
        return rms

    def get_RMS_channel_hist(self, channel_id, filename, show):
        #rms = np.sqrt(np.mean(np.square(self.data[e_id, :, 24])))
        arr = []
        for i in range(1,self.data.shape[0]):
            rms = np.sqrt(np.mean(np.square(self.data[slice(i, i+1), :, channel_id])))
            arr.append(rms)
        # Plotting the histogram
        Plot1D_hist(arr, 100, 'RMS_B%i_ACh%i_MCh%i' %(self.board, channel_id, self.channel), 'RMS', 'Events', filename, show)
    
    def get_RMS(self):
        arr = []
        for i in range(self.data.shape[2]):
            arr.append(self.get_RMS_channel(i))
        return arr

    def get_min_list(self, channel_id):
        min_arr = []
        for ev in range(1, self.data.shape[0]):
            e_id = slice(ev , ev + 1)
            min = np.min(self.data[e_id, :, channel_id])
            min_arr.append(min)
        return min_arr

    def get_min_av(self, channel_id):
        min_arr = self.get_min_list(channel_id)
        av_min = np.mean(min_arr)
        return av_min

    def get_Min_channel_hist(self, channel_id, filename, show):
        arr = self.get_min_list(channel_id)
        Plot1D_hist(arr, 80, 'Min_B%i_ACh%i_MCh%i' %(self.board, channel_id, self.channel), 'Min', 'Events', filename, show) 

    def get_min(self):
        arr = []
        for i in range(self.data.shape[2]):
            arr.append(self.get_min_av(i))
        return arr

    def full_channel_RMS(self, filename):
        print(filename)
        fig, axs = plt.subplots(5, 6, figsize=(20, 15))  # Adjust figsize as needed
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            plot_obj = self.get_RMS_channel_hist(i, 'none', 0)
            plt.sca(ax)  # Set the current axis to the one we want to plot on
            
        fig.tight_layout()
        plt.savefig(filename, transparent=False)
        plt.close()

    def full_channel_Min(self, filename):
        print(filename)
        fig, axs = plt.subplots(5, 6, figsize=(20, 15))  # Adjust figsize as needed
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            plot_obj = self.get_Min_channel_hist(i, 'none', 0)
            plt.sca(ax)  # Set the current axis to the one we want to plot on
            
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()

    def full_channel_eventScan(self, filename, nevents):
        print(filename)
        fig, axs = plt.subplots(5, 6, figsize=(20, 15))  # Adjust figsize as needed
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            plot_obj = self.plot_events_ACDCchannel(1,nevents,i, 'none', 0)
            plt.sca(ax)  # Set the current axis to the one we want to plot on
            
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()  
        
    def generate_report_input(self, nevents):
        print(' We are using input inform plots for ACDC %i, mezzanine channel %i' % (self.board, self.channel))
        output_rms_names = OUTPUT + 'RMS_ACDC_%i_MCh_%i.pdf' %(self.board, self.channel)
        output_min_names = OUTPUT + 'Min_ACDC_%i_MCh_%i.pdf' %(self.board, self.channel)
        name_ch_events = OUTPUT + 'MultiEvents_ACDC_%i_MCh_%i.pdf' %(self.board, self.channel)
        self.full_channel_RMS(output_rms_names)
        self.full_channel_Min(output_min_names)
        self.full_channel_eventScan(name_ch_events, nevents)
        print('ploting %i event display, mezzanine channel %i ' % (nevents, self.channel))
        for ev_id in range(1, 20):
            name_evD = OUTPUT + 'EventDisplay_%i_ACDC_%i_MCh_%i.pdf' %(ev_id, self.board, self.channel)
            self.event_display(ev_id, name_evD, 2) 
            

class ACDC_analysis:
    def __init__(self, board) -> None:
        self.board = board
        self.rms_list, self.min_list = self.get_arrays() 

    def get_acdc_data(self, b):
        c_acdc = ACDC(self.board, b)
        rms = c_acdc.get_RMS()
        min_val = c_acdc.get_min()
        c_acdc.generate_report_input(10)
        del c_acdc
        return rms, min_val

    def get_arrays(self):
        rms_array = []
        min_array = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(self.get_acdc_data, range(29)))
        for rms, min_val in results:
            rms_array.append(rms)
            min_array.append(min_val)
        return rms_array, min_array
    
    def plot_rms(self):
        Plot2D(self.rms_list, 'ACDC %i' % self.board, 'ACDC channels', 'Mezzanine channels', 'RMS', 0, './plots/Total_RMS.pdf')

    def plot_min(self):
        Plot2D(self.min_list, 'ACDC %i' % self.board, 'ACDC channels', 'Mezzanine channels', 'Average Min', 0, './plots/Total_AvMin.pdf')
    
    def get_plots(self):
        self.plot_rms()
        self.plot_min()


def main(ACDC_id) -> None:

    global ACDC_name

    if os.path.isdir(BASE_PATH):
        print(f"The input directory '{BASE_PATH}' exists ")
    else:
        print(f"The input directory '{BASE_PATH}' does not exist. Please fix it to continue...")
        sys.exit()

    if not os.path.isdir(OUTPUT):
        os.makedirs(OUTPUT, exist_ok=True)
        print(f"The directory '{OUTPUT}' was created.")

    if not os.path.isdir(REPORT):
        os.makedirs(REPORT, exist_ok=True)
        print(f"The directory '{REPORT}' was created.")
    
    if ACDC_id < 10:
        ACDC_name = 'ACDC0'

    ACDC_PATH = BASE_PATH + ACDC_name + str(ACDC_id) 

    if os.path.isdir(ACDC_PATH):
        print(f"The input directory '{ACDC_PATH}' exists ")
    else:
        print(f"The input directory '{ACDC_PATH}' does not exist. Please fix it to continue...")
        sys.exit()


    # Examples for simple data set

    # analyze a single data set
    # ------------------------------------------------------------------------
    #output_n = "./Scan_events"
    # Get the ACDC data <arg> (ACDC board_id, signal_in_mezzanine_channel)
    #my_acdc = ACDC(ACDC_id,0)

    # Plot a single event <arg> (event_id)
    #my_acdc.plot_event(1)

    # Plot an events and an ACDC channel <arg> (event_id, channel_id)
    #my_acdc.plot_event_ACDCchannel(1,5)

    # Plot multple events same ACDC channel, <arg> (event_1, event_2, channel_id) 
    #my_acdc.plot_events_ACDCchannel(1,50,5)

    # Event display <arg> (event id, show_plot yes, 1 or no 0)
    #my_acdc.event_display(1, 1)

    # Print multiple event displays <arg> (number of events, output path)
    #my_acdc.event_displaypdf(50, output_n)

    # Get RMS channel hist
    #my_acdc.get_RMS_channel_hist(12)

    # Get Min channels hist
    #my_acdc.get_Min_channel_hist(5)

    # Generate report v1
    #my_acdc.generate_report_input(10)

    #my_acdc.full_channel_Min('./test.pdf')
    
    #my_acdc.full_channel_eventScan('./test.pdf', 10)

    # analyze the full data set
    # ------------------------------------------------------------------------
    ana = ACDC_analysis(ACDC_id)
    ana.get_plots()
    make_report(OUTPUT, ACDC_id)

if __name__ == "__main__":
    arg = int(sys.argv[1])
    main(arg)
