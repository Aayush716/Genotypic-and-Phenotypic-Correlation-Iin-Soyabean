# Genomic selection using deep learning and saliency map

We provide a deep-learning method to predict five 	quantitative traits (Yield, Protein, Oil, Moisture and Plant height) of Soyabean dataset.
We also applied saliency map approach measure phenotype contribution for genome wide association study. 
The program is implemented using Keras2.0 and Tensorflow

### Prerequisites

Python packages are required,

```
numpy
pandas
tensorflow
keras
scipy
sklearn
matplotlib
```
## Running the program

The scripts train and test model with 10 fold cross validation and plot a comparison of genotype contribution using saliency map value and Wald test value.

Below is the code for CNN
MY ML Approaches/cnn_v2.ipynb

Similarly code for Random Forest and SVR is inside folder MY ML Approaches.


```
cd HEIGHT
python height.py

```

## Authors

* **Aayush Agrawal** - *Indian Institute of Technology Indore*
* **Email** - *aayush113.aa@gmail.com* 





