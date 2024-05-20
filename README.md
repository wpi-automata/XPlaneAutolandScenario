# XPlaneAutolandScenario
This is an autolanding scenario using the X-Plane simulator for the Grant County airport Runway 04.
The module is intended as a research tool for verification with learning-enabled components in the loop.

The `xpc3_helper.py` code is an extension of the Stanford [NASA_ULI_Xplane_Simulator](https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator/tree/sim_v2/src) code.

## Quickstart
Create a Python virtual environment using at least Python 3.7.
```
python3 -m venv ./env
```
Activate your environment and install the module (run from the top-level of this repository):
```
source ./env/bin/activate
pip install -e .
```

Download and install X-Plane 11 from [here](https://www.x-plane.com/desktop/try-it/older/).

### XPlaneConnect
Set up `XPlaneConnect` for communicating with `X-Plane` via `Python`.
Download `XPlaneConnect` version 1.2.0 from the [releases](https://github.com/nasa/XPlaneConnect/releases).
Navigate to `X-Plane 11/Resources/plugins` and extract the downloaded zip to this location.

### XCamera
Download `XCamera` from [here](https://stickandrudderstudios.com/x-camera/download-x-camera/) and extract to `X-Plane 11/Resources/plugins`.
From the plugins drop down menu, you can select XCamera and use the control panel to create and place different camera views.

### Run Autolander

Launch X-Plane 11 and start a flight with a Cessna Skyhawk at Grant Co Airport (KMWH).
Run `run_autoland.py` from within the Python virtual environment. This will land the plane using perfect state information.

### Additional Information (WPI)

To run the autolander with the NN, set the --model arg to the path leading to the desired vision model.

Set the --monitor arg to the correct monitor in which to take screenshots (the one in which XPlane is running). Can be either 0 or 1 on the WPI system, default is 1. 0 is the RIGHT monitor and 1 is the LEFT monitor (If I remember correctly). Also refer to sticky notes left on my desk. Will also need to be done with place_and_collect.py as well.

Note: Most up-to-date data generation script is in colette-dev_error-plot branch. Make sure the x-center (max distance) argument is set when run. 

Due to changes in the vision_driver.py script in order to run previously generated datasets through the NN, auto_lander.py and train.py are always run in the colette_dev branch only.

Only in colette-dev_error-plot branch: 
error_generator.py runs a dataset through the NN and records the output of the NN to a csv file called generated_states.csv. 
error_plot.py takes the true state information of the airplane for a particular image using the states.csv file in the dataset and uses it and generated_states.csv to calculate the error in output of the NN. Contrary to it's name, it doesn't actually plot anything.... Make sure the datasets are also cut over to the repo prior to running. 
image_check.py was a test program created to sanity check the tensor images. 




