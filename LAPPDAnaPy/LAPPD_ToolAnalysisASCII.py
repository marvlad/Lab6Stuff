import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import time
import sys


# Define the sequence of numbers (mapping ACDC to strip)
full_index0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
strip_index = [0, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 18, 17, 16, 15, 45, 44, 55, 54, 53, 52, 51, 50, 61, 60, 59, 58, 57, 56, 36, 31, 6, 25, 26, 27, 28, 29, 30, 19, 20, 21, 22, 23, 24, 13, 14, 46, 47, 48, 49, 38, 39, 40, 41, 42, 43, 32, 33, 34, 35, 37, 62, 63]

# Create a NumPy array from the sequence
index_original = np.array(full_index0)
index_strip    = np.array(strip_index)

def get_event(data, event_id, side, spare_channels):
    if side == 0:
        if spare_channels == 0:
            return data[event_id,:,2:30]
        else:
            return data[event_id,:,1:31]
    else:
        if spare_channels == 0:
            return data[event_id,:,33:61] 
        else:
            return data[event_id,:,32:62]

def plot2D_event(data, event_id, side, spare_channels):
   plt.imshow(get_event(data, event_id, side, spare_channels).astype(float).T,cmap='viridis', aspect='auto')
   plt.colorbar(label='Amplitude [mV]')  # Add colorbar for reference
   plt.gca().invert_yaxis()
   plt.title("Event {}, side {}".format(event_id, side))
   plt.xlabel('Time [ns]')
   if spare_channels == 0:
       plt.ylabel('Strip #')
   else:
       plt.ylabel('Strip # + spare channels')
   plt.show()

def plot1D_event(data, event_id, side, spare_channels):
    plt.plot(get_event(data, event_id, side, spare_channels))
    if spare_channels == 0:
        plt.title('Projetion Event {}, side {}'.format(event_id, side))
    else:
        plt.title('Projetion Event {}, side {}, + spare channels'.format(event_id, side))
    plt.xlabel('Time [ns]')
    plt.ylabel('Amplitude [mV]')
    plt.show()

def plot2D_eventAx(data, event_id, side, spare_channels, ax):
   im = ax.imshow(get_event(data, event_id, side, spare_channels).astype(float).T,cmap='viridis', aspect='auto')
   plt.colorbar(im, ax=ax).set_label('Amplitude [mV]')  # Add colorbar for reference
   ax.invert_yaxis()
   ax.set_title("Side {}".format(side))
   ax.set_xlabel('Time [ns]')
   if spare_channels == 0:
       ax.set_ylabel('Strip #')
   else:
       ax.set_ylabel('Strip # + spare channels')

def plot1D_eventAx(data, event_id, side, spare_channels, ax):
    ax.plot(get_event(data, event_id, side, spare_channels))
    if spare_channels == 0:
        ax.set_title('Projetion , side {}'.format(side))
    else:
        ax.set_title('Projetion, side {}, + spare channels'.format(side))
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Amplitude [mV]')

def display_event(data, event_id, spare_channels, show_display):
    if show_display == 0:
        plt.switch_backend('Agg')

    # Create a figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    
    # Plot data on each subplot
    plot2D_eventAx(data, event_id, 0, spare_channels, ax1)
    plot2D_eventAx(data, event_id, 1, spare_channels, ax2)
    plot1D_eventAx(data, event_id, 0, spare_channels, ax3)
    plot1D_eventAx(data, event_id, 1, spare_channels, ax4)

    # Add a general title to the canvas
    fig.suptitle("Event {}".format(event_id), fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    if show_display == 1:
        plt.show()
    
    plt.close()
    return fig

def ReadData(data_path, pedestal_path1, pedestal_path2, substract_ped):   
    # --------------------------------------------------------------------------------------------------
    # Read the event data into a pandas DataFrame
    data = pd.read_csv(data_path, sep=' ', header=None)

    # Read pedestal1 data into a pandas DataFrame
    pedestal1_data = pd.read_csv(pedestal_path1, sep=' ', header=None)

    # Read pedestal2 data into a pandas DataFrame
    pedestal2_data = pd.read_csv(pedestal_path2, sep=' ', header=None)

    # Assuming each event contains 256 lines
    lines_per_event = 256

    # Calculate the number of events
    num_events = len(data) // lines_per_event

    # Reshape the pedestal data to match the shape of the event data
    pedestal1_values = pedestal1_data.values.reshape(1, -1, 30).astype(float)
    pedestal2_values = pedestal2_data.values.reshape(1, -1, 30).astype(float)

    # Reshape the event data to have one row per event
    event_data = data.values.reshape((num_events, lines_per_event, -1))

    if substract_ped == 1:
        # sub pedestal 1
        event_data[:, :, 1:31] -= pedestal1_values 
        # sub pedestal 2
        event_data[:, :, 32:62] -= pedestal2_values
    
    return event_data 

def ADC2Voltage(event_data):
    # Multiply by 0.3 to convert ADC to Voltage
    # -1 to 0.3 is to invert the waveforms
    event_data[:, :, 1:31]  = np.multiply(event_data[:, :, 1:31],-0.3) 
    event_data[:, :, 32:62] = np.multiply(event_data[:, :, 32:62],-0.3)
    return event_data 

def ACDCmetaCorrection(event_data):
    # --------------------------------------------------------------------------------------------------
    # Reorder ACDC meta 
    num_events = event_data.shape[0]

    bs1 = int("0000000000000111", 2)
    shift_values = np.array([bs1 & int(x, 16) for x in event_data[:,10,31]])
    shift_values = np.multiply(shift_values, 32)

    shift_global = np.full(num_events, 80)

    # Create a range of indices along the rows axis
    row_indices = np.arange(event_data.shape[1])
    # Expand the meta array to match the shape of event_data along the rows axis
    expanded_meta = np.expand_dims(-shift_values, axis=1)
    # Calculate the rolled indices using broadcasting
    rolled_indices = (row_indices - expanded_meta) % event_data.shape[1]
    # Use advanced indexing to roll the event_data array
    event_data = event_data[np.arange(len(-shift_values))[:, None], rolled_indices]

    # Shift 80 time units (Matt?)
    # Expand the meta array to match the shape of event_data along the rows axis
    expanded_meta = np.expand_dims(-shift_global, axis=1)
    # Calculate the rolled indices using broadcasting
    rolled_indices = (row_indices - expanded_meta) % event_data.shape[1]
    # Use advanced indexing to roll the event_data array
    event_data = event_data[np.arange(len(-shift_global))[:, None], rolled_indices]
    return event_data

def ACDC2Strip(event_data):
    # --------------------------------------------------------------------------------------------------
    # ACDC channels index to Strip index
    event_data = event_data[:, :, strip_index]
    return event_data 

def BaselineCorrection(event_data):
    # --------------------------------------------------------------------------------------------------
    # Baseline correction
    # Calculate the baseline for each row
    #baseline1 = np.mean(event_data[:,0:255,1:31], axis=1, keepdims=True)
    baseline1 = np.mean(event_data[:,100:120,1:31], axis=1, keepdims=True)
    event_data[:, :, 1:31] -= baseline1 

    #baseline2 = np.mean(event_data[:,0:255,32:62], axis=1, keepdims=True)
    baseline2 = np.mean(event_data[:,100:120,32:62], axis=1, keepdims=True)
    event_data[:, :, 32:62] -= baseline2 
    return event_data

def Display_eventsPDF(event_data, spare_chn, nevent, output_name):
    print("Plotting the events....")
    # Create a PDF file to save the plots
    plt.switch_backend('Agg')
    with PdfPages(output_name+".pdf") as pdf:
        # Initialize the progress bar
        progress_bar = tqdm(total=nevent, desc="Progress", unit="iteration")
        for i in range(nevent):
            fig = display_event(event_data, i, spare_chn, 0)
            pdf.savefig(fig)
            progress_bar.update(1)
        progress_bar.close()
        print("Plotting done....")

def main():
    nevents  = sys.argv[1]
    datapath = sys.argv[2]
    datapath = '/Users/marvinascenciososa/Desktop/pnfs_mrvn/mac/Desktop/Lab6_tests/python_proj/input/data/Ascii20241903_141434_100ev.txt'
    ped1     = '/Users/marvinascenciososa/Desktop/pnfs_mrvn/mac/Desktop/Lab6_tests/python_proj/input/pedestal/ACDC31_needCheck.txt'
    ped2     = '/Users/marvinascenciososa/Desktop/pnfs_mrvn/mac/Desktop/Lab6_tests/python_proj/input/pedestal/ACDC26_needCheck.txt'

    output   = './Event_display'
    pedestalsubstraction = 1
    event_data = ReadData(datapath, ped1, ped2, pedestalsubstraction)

    event_data = ADC2Voltage(event_data)
    event_data = ACDCmetaCorrection(event_data)
    event_data = ACDC2Strip(event_data)
    event_data = BaselineCorrection(event_data)
    
    #print(event_data[6])
    #plot1D_event(event_data, 4, 1, 1)
    #plot2D_event(event_data, 4, 1, 1)

    #display_event(event_data, 4, 1, 1)
    #Display_eventsPDF(event_data, 1, 10, output)
    Display_eventsPDF(event_data, 1, nevents, output)


    #plt.plot(event_data[:,:,32].T)
    #plt.show()

    #plt.imshow(event_data[:,:,32].astype(float).T,cmap='viridis', aspect='auto')
    #plt.imshow(event_data[:,:,6].astype(float),cmap='viridis', aspect='auto')
    #plt.colorbar(label='Amplitude [mV]')  # Add colorbar for reference
    #plt.gca().invert_yaxis()
    #plt.show()

if __name__ == "__main__":
    main()
