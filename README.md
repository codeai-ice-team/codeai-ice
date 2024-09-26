# ICE: Industrial Condition Evaluation
Python library for training, evaluating, using deep learning for industrial monitoring tasks.

## Tasks
The library solves the following industrial monitoring tasks:
|Task|Setting|Type|Description|
|----|-------|----|-----------|
|Fault diagnosis|Supervised|Multiclass classification|For a given sample of sensor data, to classify the type of fault|
|Fault detection|Unsupervised|Binary classification|For a given sample of sensor data, to define if it is a fautly state or the normal state|
|Remaining Useful Life Estimation|Supervised|Regression|For a given sample of sensor data, to estimate the number of machine operation cycles until a breakdown|
|Health Index Estimation|Supervised|Regression|For a given sample of sensor data, to measure the deviation of the device state from a fully functional state|

## Industrial applications
1. __Chemical industry__: Monitoring and optimisation of chemical processes,
management and prevention of emergency situations.
2. __Oil and gas industry__: Prediction and prevention of equipment failures, optimisation of production and processing. Equipment failure prediction and prevention, production and processing optimisation.
3. __Energy__: Monitoring and control of energy systems, including
power plants, wind turbines and solar power plants.
4. __Automotive__: Diagnosis and prediction of potential failures and breakdowns in automotive manufacturing processes and the production processes of automotive plants.
5. __Electronics manufacturing__: Quality control and defect detection in the manufacturing of electronic components. Production process of electronic components.
6. __Food production__: Monitoring of production parameters to ensure food safety and quality. Food safety and quality assurance.
7. __Pharmaceutical industry__: Monitoring and optimising the production of pharmaceuticals.

## Benchmarking
The library provides benchmarks for industrial monitoring tasks:
1. Tennessee Eastman Process (TEP) — senosor data of the chemical process consisting of 52 sensors and 28 faulty states for evaluation fault detection/diagnosis systems.
2. NASA Milling — the dataset with 16 sub-datasets in which the cutting depths, feed rates and material types were changed. The sensors used to record the milling process are two vibroaccelerometers, two “AE” sensors, and two “CTA” current sensors. The dataset is used for evaluation Health Index Estimation methods.
3. NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) — 
dataset with four subsets of data. Each subset contains readings from 21 sensors located in different parts of the degrading engine, controlled by three operational settings characteristics: height, speed, and acceleration in terms of altitude, Mach number, and throttle resolver angle. The dataset is used for evaluation Remaining Useful Lifetime Estimation methods.

## Installation
To install ICE, run the following command:
```
pip install git+https://github.com/codeai-ice-team/codeai-ice
```
## Minimum system requirements
* OS: Windows 11 or higher, MacOS 14 or higher, Ubuntu 18 or higher
* Processor: 1xCPU 2.2GHz
* RAM: 13 GB
* HDD Space: 100 GB
* Video Card for training: 1xGPU NVIDIA GTX1080

## Testing
To test ICE, download the repository by pressing on "Code" and "Download ZIP" in the main page. Then, unzip the files and go to the root folder by the following command (for linux, macos):
```
cd codeai-ice-main
```
Finally, run the following command:
```
pytest ice/tests
```

## Documentation
The documentation is available here: https://codeai-ice-team.github.io/codeai-ice-docs/.
