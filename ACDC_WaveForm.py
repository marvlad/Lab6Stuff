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

BASE_PATH = 'the_path_here'
ACDC_name = 'acdc_'

mezzanine_id = [24,25,26,27,28,29,18,19,20,21,22,23,12,13,14,15,16,17,6,7,8,9,10,11,0,1,2,3,4]

class ACDC:
    def __init__(self, board, channel)-> None:
        self.board = board
        self.channel = channel # ID on injected signal in the mezzanine phyical board
        self.data = self.get_corrected_wf()

    def get_corrected_wf(self):
        print("Loadding data from %s, ACDC %s and channel %s " % (BASE_PATH, self.board, self.channel)) 

        # Calculate pedestals
        pedestal_files = sorted(glob(BASE_PATH + '%s%i/pedestal/*.txt' % (ACDC_name,self.board)))
        kwargs = {'header': None, 'delimiter': ' ', 'usecols': range(1,31)}
        d = np.vstack([pd.read_csv(x, **kwargs).values for x in pedestal_files[:1]])
        d = d.reshape(int(len(d) / 256), 256, 30)
        peds = np.mean(d, axis=0) 

        # Channel data
        input_files = sorted(glob(BASE_PATH + '%s%i/channels/*.txt' % (ACDC_name,self.board)))
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
        plt.imshow(self.data[e_id,:,0:30].T, cmap='viridis', aspect=7.0)
        cbar = plt.colorbar()
        cbar.set_label('Amplitude [mV]')
        plt.grid(True)
        plt.title('Event %i' % event_id)
        plt.xlabel('Time [ns]')
        plt.ylabel('ACDC Channels')
        plt.show()
    
    def plot_event_ACDCchannel(self, event_id, acdc_channel):
        e_id = slice(event_id, event_id + 1)
        plt.plot(self.data[e_id,:,acdc_channel].swapaxes(0,1))
        plt.grid(True, linestyle='--')
        plt.title('Event %i' % event_id)
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [mV]')
        plt.show()
     
    def plot_events_ACDCchannel(self, event1, event2, acdc_channel):
        e_id = slice(event1, event2)
        plt.plot(self.data[e_id,:,acdc_channel].swapaxes(0,1))
        plt.grid(True, linestyle='--')
        plt.title('Events from %i to %i' % (event1,event2))
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [mV]')
        plt.show()

    def event_display(self, event_id, show_event):
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

        plt.close()
        return fig
    
    def event_displayPDF(self, nevent, output_name):
        plt.switch_backend('Agg')
        with PdfPages(output_name+".pdf") as pdf:
            progress_bar = tqdm(total=nevent, desc="Progress", unit="iteration")
            for i in range(nevent):
                fig = self.event_display(i, 0)
                pdf.savefig(fig)
                progress_bar.update(1)
            progress_bar.close()

    def get_RMS_channel(self, channel_id):
        #rms = np.sqrt(np.mean(np.square(self.data[e_id, :, 24])))
        rms = np.sqrt(np.mean(np.square(self.data[:, :, channel_id])))
        #print(rms)
        return rms
    
    def get_RMS(self):
        num_rows = self.data.shape[2]
        arr = []
        for i in range(num_rows):
            arr.append(self.get_RMS_channel(i))
        return arr

    #def get_min():
       

class ACDC_analysis():
    def __init__(self, board)-> None:
        self.board = board
        self.rms_list = self.get_rms() 

    def get_rms(self):
        rms_array = []
        for b in range(29):
            c_acdc = ACDC(self.board, b)
            rms_array.append(c_acdc.get_RMS())
            del c_acdc
        return rms_array
    
    def plot_rms(self):
        plt.imshow(self.rms_list, cmap='viridis', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label('RMS')
        plt.grid(True, linestyle='--')
        plt.title('ACDC %i' % self.board)
        plt.xlabel('ACDC channels')
        plt.ylabel('Mezzanine channels')
        plt.show()


def main() -> None:
    # arg events, = int(sys.argv[1])

    # Examples for simple data set

    # analize single data set
    # ------------------------------------------------------------------------
    #output_n = "./Scan_events"
    # Get the ACDC data <arg> (ACDC board_id, signal_in_mezzanine_channel)
    #my_acdc = ACDC(35,0)

    # Plot a single event <arg> (event_id)
    #my_acdc.plot_event(1)

    # Plot an events and an ACDC channel <arg> (event_id, channel_id)
    #my_acdc.plot_event_ACDCchannel(1,5)

    # Plot multple events same ACDC channel, <arg> (event_1, event_2, channel_id) 
    #my_acdc.plot_events_ACDCchannel(1,10,5)

    # Event display <arg> (event id, show_plot yes, 1 or no 0)
    #my_acdc.event_display(1, 1)

    # Print multiple event displays <arg> (number of events, output path)
    #my_acdc.event_displayPDF(50, output_n)

    # Get RMS
    #my_acdc.get_RMS()

    # analize full data set
    # ------------------------------------------------------------------------
    ana = ACDC_analysis(35)
    ana.plot_rms()

if __name__ == "__main__":
    main()
