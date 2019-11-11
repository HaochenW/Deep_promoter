Synthetic Promoter Design in Escherichia coli based on Deep Generative Network
=====================================

Code for computational models in ["Synthetic Promoter Design in Escherichia coli based on Deep Genera-tive Network"](https://doi.org/10.1101/563775).

We report a novel AI-based framework for de novo promoter design in E. coli, which could design brand new synthetic promoters in silico. We combined deep generative model that guides the search, and a prediction model to pre-select the most promising promoters. 

<p align="center">
  <img width="650" height="300" src="https://github.com/HaochenW/Deep_promoter/blob/master/structure.png">
</p>

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

## Using GAN model to generate promoter sequence data
- Procdure:
- 1. Store the sequence data in .\seq\sequence_data.txt
- 2. Change the setting in python_language.py, which may include BATCH_SIZE, SEQ_LEN (must be changed, based on your sequence length), MAX_N_EXAMPLES(must be changed, based on the number of all the sequences), etc
- 3. Train the model by `python gan_language.py`
- The generated sequences will be saved in the current folder

## Using Convoluational neural network(CNN)/Suppoer Vector Regression (SVR) model to pre-select high-expression promoter sequences
The CNN model was trained by the dataset from the Thomason et al which contains 14098 promoters with corresponding gene expression level measured by dRNA-seq, and the SVR model was trained by the first round experimental results.
- Procdure:
- 1. Store the predicted promoter sequence in .\seq\predicted_promoters.fa
- 2. Use the predictor by `python predictor.py`
- The predicted results will be saved in `seq_exp_SVR.txt` and `seq_exp_CNN.txt`

## Citation
Wang Y, Wang H, Liu L, et al. Synthetic Promoter Design in Escherichia coli based on Generative Adversarial Network[J]. BioRxiv, 2019: 563775.
@article{liu2019hicgan,
  title={hicGAN infers super resolution Hi-C data with generative adversarial networks},
  author={Liu, Qiao and Lv, Hairong and Jiang, Rui},
  journal={Bioinformatics},
  volume={35},
  number={14},
  pages={i99--i107},
  year={2019},
  publisher={Oxford University Press}
}

## References
[1]  Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., and Courville, A. C. (2017) Improved training of wasserstein gans, In Advances in Neural Information Processing Systems, pp 5767-5777.
[2]  Thomason, M. K., Bischler, T., Eisenbart, S. K., Forstner, K. U., Zhang, A., Herbig, A., Nieselt, K., Sharma, C. M., and Storz, G. (2015) Global transcriptional start site mapping using differential RNA sequencing reveals novel antisense RNAs in Escherichia coli, Journal of bacteriology 197, 18-28.


## License
This project is licensed under the MIT License - see the LICENSE.md file for details


