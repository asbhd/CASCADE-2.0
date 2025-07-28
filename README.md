# CASCADE-2.0: Real-time Prediction of <sup>13</sup>C NMR Shifts with sub-ppm Accuracy
This repository contains the code, data and trained models to reproduce the results for the work: 

The data folder contains the molecular geometries and the chemical shifts organised in the respective model folders. The models folder contains the code to process the data and the train the models published in the study. Pretrained model weights are also available to make direct predictions.

## Setting up the Environment

To correctly setup the environment for the model, the following dependencies are required

* rdkit==2023.09.05
* tensorflow=2.11.0
* tensorflow-probability==0.19.0
* KGCNN==2.2.1
* pandas==1.5.3

To set up the environment one could pip install the above package in a virtual environment (Python 3.10 recommended), or use the cascadeV2.yaml file to setup the required environment using ```conda env create -f cascadeV2.yml```

Using the above version of the packages is recommended, as some import paths and codes have been known to be substantially changed in the subsequent versions of the packages.

## Training Models

To train a model go to the respective directory and run ```python preprocess.py```. **The generated files need to be moved to a directory named data in the same directory** . Finally, the model can be trained with ```python train.py```

## Using Pre-trained Models

Within each directory trained model weights are provided to make predictions from .sdf files. Notebooks containing examples to make predictions using the pre-trained models have been included

## Webserver
The model has also been implemented as a webserver to enable quick predictions from SMILES strings. The webserver can be accessed at: https://nova.chem.colostate.edu/v2/cascade/

## Citation
If you use CASCADE-2.0 in your work, please cite the following work: 
