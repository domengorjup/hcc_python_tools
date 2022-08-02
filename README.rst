
==================
`hcc` Python tools
==================

This repository contains various tools, used to work with Telops .hcc files in `LADISK`_.

To read the `.hcc` file data, the `PythonTelopsToolbox` package, provided by Telops, is used. This is not included with this repository.

Other external packages, required to use these tools, can be installed using `pip` with the provided requirements file:

.. code::

    pip install -r requirements.txt


Basic usage
-----------

A simple example of basic usage is provided below:

.. code:: python
    
    from telops_hcc_tools import read_hcc, read_segmented
    import numpy as np
    import napari
    
    
    # Read a single .hcc file
    file_single = 'hcc_example/example_1.hcc'
    
    time, frames, header = read_hcc(file_single, allow_pickle=False)
    print(f'Read single `.hcc` file. Image data shape: {frames.shape}.')
    
    # Read a segmented .hcc measurement
    files_segmented = [
        'hcc_example/example_1.hcc', 
        'hcc_example/example_2.hcc'
        ]
    
    time_s, frames_s, header_s = read_segmented(files_segmented, allow_pickle=False)
    print(f'Read segmented `.hcc` file. Image data shape: {frames_s.shape}.')
    
    # Display the read data using napari
    
    def update_t_annotation(event):
            step = viewer.dims.current_step[0]
            viewer.text_overlay.text = f"{time[step]:.5f} s"
            
    viewer = napari.view_image(data=frames, colormap='viridis', title='exmaple footage')
    viewer.text_overlay.visible = True
    viewer.dims.events.current_step.connect(update_t_annotation)
    napari.run()

.. _LADISK: http://ladisk.si/

