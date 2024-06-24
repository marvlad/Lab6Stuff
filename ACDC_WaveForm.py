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

BASE_PATH = 'you_path_here'
ACDC_name = 'acdc_0'

mezzanine_id = [24,25,26,27,28,29,18,19,20,21,22,23,12,13,14,15,16,17,6,7,8,9,10,11,0,1,2,3,4]

def Plot1D():
    print("plot..")

def Plot2D(data, title, xl, yl, zl):
    print("print...")
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label(zl)
    plt.grid(True, linestyle='--')
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()

class ACDC:
    def __init__(self, board, channel)-> None:
        self.board = board
        self.channel = channel # ID on injected signal in the mezzanine phyical board
        self.data = self.get_corrected_wf()

    def get_corrected_wf(self):
        print("Loadding data from %s, ACDC %s, channel %s " % (BASE_PATH, self.board, self.channel)) 

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

    def get_RMS_channel_hist(self, channel_id):
        #rms = np.sqrt(np.mean(np.square(self.data[e_id, :, 24])))
        arr = []
        for i in range(1,self.data.shape[0]):
            rms = np.sqrt(np.mean(np.square(self.data[slice(i, i+1), :, channel_id])))
            arr.append(rms)
        # Plotting the histogram
        plt.hist(arr, bins=100, edgecolor='black')  # Adjust bins as needed
        plt.xlabel('RMS')
        plt.ylabel('Events')
        plt.title('RMS, ACDC %i, ACDC Ch. %i, Mezzanine Ch. %i' %(self.board, channel_id,  (self.channel + 1)))
        plt.grid(True, linestyle='--')  # Optional grid

        # Display the plot
        plt.show()
    
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

    def get_Min_channel_hist(self, channel_id):
        #rms = np.sqrt(np.mean(np.square(self.data[e_id, :, 24])))
        arr = self.get_min_list(channel_id)
        plt.hist(arr, bins=80, edgecolor='black')  # Adjust bins as needed
        plt.xlabel('Min')
        plt.ylabel('Events')
        plt.title('Min, ACDC %i, ACDC Ch. %i, Mezzanine Ch. %i' %(self.board, channel_id,  (self.channel + 1)))
        plt.grid(True, linestyle='--')  # Optional grid

        # Display the plot
        plt.show()

    def get_min(self):
        arr = []
        for i in range(self.data.shape[2]):
            arr.append(self.get_min_av(i))
        return arr 
       

class ACDC_analysis():
    def __init__(self, board)-> None:
        self.board = board
        self.rms_list, self.min_list = self.get_arrays() 

    def get_arrays(self):
        rms_array = []
        min_array = []
        for b in range(29):
            c_acdc = ACDC(self.board, b)
            rms_array.append(c_acdc.get_RMS())
            min_array.append(c_acdc.get_min())
            del c_acdc
        return rms_array, min_array
    
    def plot_rms(self):
        Plot2D(self.rms_list, 'ACDC %i' % self.board, 'ACDC channels', 'Mezzanine channels', 'RMS')

    def plot_min(self):
        Plot2D(self.min_list, 'ACDC %i' % self.board, 'ACDC channels', 'Mezzanine channels', 'Average Min')


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

    # Get RMS channel hist
    #my_acdc.get_RMS_channel_hist(12)

    # Get Min channels hist
    #my_acdc.get_Min_channel_hist(5)

    # analize full data set
    # ------------------------------------------------------------------------
    ana = ACDC_analysis(5)

    # Plot the RMS of the full set (all 29 files)
    ana.plot_rms()

    # Plot the Min of the full set (all 29 files)
    ana.plot_min()

if __name__ == "__main__":
    main()