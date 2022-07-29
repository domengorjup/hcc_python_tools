#!/usr/bin/env python

"""telops_hcc_tools.py: Common tools to interface with Telops .hcc files."""

__author__ = "Domen Gorjup"


import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm


from TelopsToolbox.hcc.readIRCam import read_ircam


def read_hcc(file, crop_tblr=None):
    """
    Load a single .hcc file.
    """
    measurement_name = os.path.split(os.path.split(file)[0])[-1]
    _, header, _, _ = read_ircam(file, headers_only=True)
    widths = header['Width']
    heights = header['Height']
    time_true = header['POSIXTime'] + header['SubSecondTime']*(10**(-7))
    time_true -= time_true[0]
    w = widths[0]
    h = heights[0]
    convert_offset = header['DataOffset']
    fps = header['AcquisitionFrameRate'][0]
    dt = 1 / fps
    N = widths.shape[0]
    time = np.arange(N) * dt + dt
    data, header, _, _ = read_ircam(file, frames=np.arange(N))
    
    if all(header["Height"] == header["Height"]) and all(header["Width"] == header["Width"]):
        data_im_raw = np.reshape(data, (N, h, w))[:, ::-1, ::-1]
    else:
        data_im_raw = np.array([np.reshape(data[i, :], (header["Height"][i], header["Width"][i])) for i in range(N)])[:, ::-1, ::-1]

    if crop_tblr is None:
        frame_data = data_im_raw
    elif len(crop_tblr) == 4:
        top, bottom, left, right = crop_tblr
        frame_data = data_im_raw[:, top:bottom, left:right]
    
    data_im = frame_data - convert_offset[:, None, None]
    return time, data_im, header


def pickle_dump(path, **kwargs):
    """
    Dump a dictionary of all keyword arguments to pickle.
    """
    with open(path, 'wb') as file:
        pickle.dump(kwargs, file)


def read_segmented(files, crop_tblr=None, save_pickle=True):
    """
    Load Telops footage, segmented into multiple .hcc `files`, into a single array.
    """
    if len(files) > 1:
        frames = None
        time = None
        for f in tqdm(files):
            t, data_im, header, = read_hcc(f, crop_tblr=crop_tblr)
            if frames is None:
                frames = data_im
                time = t
            else:
                frames = np.append(frames, data_im, axis=0)
                time = np.append(time, t+time[-1])
    
    elif len(files) == 1:
        time, frames, header = read_hcc(files[0], crop_tblr=crop_tblr)

    if save_pickle:
        pickle_file = os.path.splitext(files[0])[0] + '.pkl'
        pickle_dump(pickle_file, time=time, frames=frames, header=header)
        print(f'Data (time, frames, header) saved to {pickle_file:s}.')

    return time, frames, header



def convert_to_pickle(files, crop_tblr=None):
    """
    Read .hcc file / multiple files and save the data into a .pkl file.
    """
    if type(files) == str:
        files = [files]

    for file in files:
        print(f'Converting file:\n\t{file:s}')
        time, frames, header = read_hcc(file, crop_tblr=crop_tblr)

        pickle_file = os.path.splitext(file)[0] + '.pkl'
        pickle_dump(pickle_file, time=time, frames=frames, header=header)
        print(f'Data (time, frames, header) saved to {pickle_file:s}.')
        print()



if __name__ == '__main__':
    import napari
    from PyQt5.QtWidgets import QFileDialog, QApplication
    import matplotlib.pyplot as plt

    def gui_folder(dir=None):
        """Select a file via a dialog and return the file name."""
        if dir is None: dir ='./'
        app = QApplication([])
        fname = QFileDialog.getExistingDirectory(None, "Izberi mapo s posnetki.")
        return fname

    root_dir = gui_folder()
    measurement_name = os.path.split(root_dir)[-1]

    print('Found .hcc files:\n')

    hcc_files = []
    i = 0
    this_dir = ''
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if os.path.splitext(f)[1] == '.hcc':
                
                if os.path.split(dirpath)[-1] != this_dir:
                    this_dir = os.path.split(dirpath)[-1]
                    print(this_dir)
                    
                hcc_files.append(os.path.join(dirpath, f))
                print(f'\t{i:d}: {f:s}')  
                i += 1

    print('Loading files.')
    time, frames, header = read_segmented(hcc_files, save_pickle=False)

    print('Convert files to pickle.')
    convert_to_pickle(hcc_files)

    plt.figure()
    plt.imshow(frames[0], cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('T [°C]', rotation=90)
    plt.title('frame 0')
    plt.axis('off')
    plt.show()

    # napari viewer
    print('View loaded footage.')
    def update_t_annotation(event):
        step = viewer.dims.current_step[0]
        viewer.text_overlay.text = f"{time[step]:.5f} s"
        
    viewer = napari.view_image(data=frames, colormap='viridis', title=measurement_name)
    viewer.text_overlay.visible = True
    viewer.dims.events.current_step.connect(update_t_annotation)
    napari.run()