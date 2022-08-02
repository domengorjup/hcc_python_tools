#!/usr/bin/env python

"""telops_hcc_tools.py: Common tools to interface with Telops .hcc files."""

__author__ = "Domen Gorjup, Klemen Zaletelj"


import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm


from TelopsToolbox.hcc.readIRCam import read_ircam


def read_hcc(file, crop_tblr=None, allow_pickle=True):
    """
    Load a single .hcc file.

    :param file: str, path to .hcc file to read.
    :param crop_tblr: tuple, (top, bottom, left, right), parameters used to crop the read footage. 
        If None, the images are not cropped. Defaults to None.
    :param allow_pickle: bool, whether or not to use pickle to load / save the read data. If True,
        the data is read from a .pkl file with the same name, if one exists, and the read data
        is saved to a pickle file if one does not yet exist. Defaults to True.
    :return time: array of shape (n,), the time vector corresponding to read image frames.
    :return frames: aray of shape (n, h, w), the read image data.
    :return header: dict, the .hcc file info header.
    """
    pickle_file = os.path.splitext(file)[0] + '.pkl'
    if os.path.exists(pickle_file) and allow_pickle:
        return pickle_load(pickle_file)
    
    else:
        #measurement_name = os.path.split(os.path.split(file)[0])[-1]
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

        frames = frame_data - convert_offset[:, None, None]

        if allow_pickle:
            pickle_dump(pickle_file, time=time, frames=frames, header=header)

        return time, frames, header


def pickle_dump(path, time, frames, header):
    """
    Dump a dictionary of read .hcc file to pickle.
    """
    with open(path, 'wb') as file:
        pickle.dump({'time':time, 'frames':frames, 'header':header}, file)


def pickle_load(path):
    """
    Read a pickled dictionary of .hcc data.

    :return time: array of shape (n,), the time vector corresponding to read image frames.
    :return frames: aray of shape (n, h, w), the read image data.
    :return header: dict, the .hcc file info header.
    """
    with open(path, 'rb') as f:
        out = pickle.load(f)
    
    return out['time'], out['frames'], out['header']


def read_segmented(files, crop_tblr=None, allow_pickle=True):
    """
    Load Telops footage, segmented into multiple .hcc `files`, and concatenate it into a 
    single array.

    :param files: list or str, list of multiple filenames, constituting a single segmented .hcc
        measurement. If the list contains the path to a single .hcc file, the return of `read_hcc`
        is returned. If a sting is passed instead, it is assumed to contain the path to a single
        .hcc file.
    :param crop_tblr: tuple, (top, bottom, left, right), parameters used to crop the read footage. 
        If None, the images are not cropped. Defaults to None.
    :param allow_pickle: bool, whether or not to use pickle to load / save the read data. If True,
        the data is read from a .pkl file with the same name, if one exists, and the read data
        is saved to a pickle file if one does not yet exist. Defaults to True.
    :return time: array of shape (n,), the time vector corresponding to read image frames.
    :return frames: aray of shape (n, h, w), the read image data.
    :return header: dict, the .hcc file info header.

    """
    if type(files) == str:
        files = [files]

    pickle_file = os.path.splitext(files[0])[0] + '_all_segments.pkl'

    if os.path.exists(pickle_file) and allow_pickle:
        return pickle_load(pickle_file)

    else:
        if len(files) > 1:
            frames = None
            time = None
            for f in tqdm(files):
                t, data_im, header, = read_hcc(f, crop_tblr=crop_tblr, allow_pickle=False)
                if frames is None:
                    frames = data_im
                    time = t
                else:
                    frames = np.append(frames, data_im, axis=0)
                    time = np.append(time, t+time[-1])
        
        elif len(files) == 1:
            time, frames, header = read_hcc(files[0], crop_tblr=crop_tblr, allow_pickle=allow_pickle)

        if allow_pickle:
            pickle_dump(pickle_file, time=time, frames=frames, header=header)
            print(f'Data (time, frames, header) saved to {pickle_file:s}.')

        return time, frames, header


def convert_to_pickle(files, crop_tblr=None):
    """
    Read .hcc file / multiple files and save the data into a .pkl file.
    Does not return the read data. To read the data use `read_segmented`.

    :param files: list or str, list of multiple filenames, constituting a single segmented .hcc
        measurement. If the list contains the path to a single .hcc file, the return of `read_hcc`
        is returned. If a sting is passed instead, it is assumed to contain the path to a single
        .hcc file.
    :param crop_tblr: tuple, (top, bottom, left, right), parameters used to crop the read footage. 
        If None, the images are not cropped. Defaults to None.
    """
    if type(files) == str:
        files = [files]

    for file in files:
        print(f'Converting file:\n\t{file:s}')

        # allow_pickle is False to force the overwrite of existing pickle files.
        time, frames, header = read_hcc(file, crop_tblr=crop_tblr, allow_pickle=False)

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