# DESCRIPTION
This repository contains code to reproduce results from the paper:

**NAPCTest: Neuron Activation Path Coverage Guided White-Box Testing Framework for Deep Learning Systems**


## REQUIREMENTS

- Python 3.6.13
- Tensorflow 1.12.0 
- Numpy 1.15.4 
- cv2 3.4.2
- scipy 1.1.0

## EXPERIMENTS

The repository consists of two Python scripts and two folders. The folder utils includes the required util functions for obtaining the activation from the training set and using NAPCTest to generate test cases. The folder metric contains the six evaluation metrics in our paper. You should download the [data](https://www.image-net.org/challenges/LSVRC/2012/index.php) and [pretrained models](https://github.com/google-research/tf-slim) before running the code. 

### Running the code

- `python obhtain_activation.py`:  compute the mean activation values for each neurons in the network.
- `python NAPCTest.py`:  generate test cases by our proposed NAPCRTest method.
- `python metric/MR.py`:  compute the MR metric for generated test cases.
- `python metric/ACTC.py`:  compute the ACTC metric for generated test cases.
- `python metric/ALD.py`:  compute the ALD metric for generated test cases.
- `python metric/ASS.py`:  compute the ASS metric for generated test cases.
- `python metric/RGB.py`:  compute the RGB metric for generated test cases.
- `python metric/RIC.py`:  compute the RIC metric for generated test cases.

### Example usage

After cloning the repository you can run the giving code to generate test cases and then evaluate the quality by six metrics.

- Generate test cases:

```
python obhtain_activation.py
python NAPCTest.py
```

- Evaluate the quality of generated test cases：

```
python metric/MR.py
python metric/ACTC.py
python metric/ALD.py
python metric/ASS.py
python metric/RGB.py
python metric/RIC.py
```
