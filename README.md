Synthetic Promoter Design in Escherichia coli based on Deep Generative Network
=====================================

Code for computational models in ["Synthetic Promoter Design in Escherichia coli based on Deep Genera-tive Network"](https://doi.org/10.1101/563775).

We report a novel AI-based framework for de novo promoter design in E. coli, which could design brand new synthetic promoters in silico. We combined deep generative model that guides the search, and a prediction model to pre-select the most promising promoters. 

From the experimental results, up to 70.8% of the AI-designed promoters were experimentally demonstrated to be functional and shared no significant sequence similarity with E. coli genome. Here, we introduced the code used for promoter sequences generation, then the promoters could be used for experimental tests.



## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib, Keras
- A recent NVIDIA GPU
- TensorFlow == 1.10.0
- Cuda == 9.0
- Python == 2.7.0

## Installation
Our computational models could be directly downloaded by:
`git clone https://github.com/HaochenW/Deep_promoter.git`
Installation has been successfully tested in a Linux platform.

## Training WGAN model on sequence data

- Note that this code uses python2
- Procdure:
- 1. Store the sequence data in .\seq\sequence_data.txt
- 2. Change the setting in python_language.py, which may include BATCH_SIZE, SEQ_LEN (must be changed, based on your sequence length), MAX_N_EXAMPLES(must be changed, based on the number of all the sequences), etc
- 3. Train the model by `python gan_language.py`
- The generated sequences will be saved in the current folder

## References