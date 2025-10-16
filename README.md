# EECE-490 Project Repository  

This repository is maintained by the following team members:  

- **Bahaa Hamdan**  
- **Jad Eido**  
- **Khaled Kanawati**  

We will be collaborating here to develop and maintain the project’s code....


## To Run Code Locally:
#### Prerequisite:
- Make sure you can run Jupyter Notebook locally, either through VS Code or other means

**Importing Procedure**:
- On cmd, navigate to your chosen directory using cd commadn
- Once in the directory, use `git clone URL`, where URL is the URL of this github page, foudn by pressign the big green **CODE** button in the main page and copying
- Open the files usign an IDE, they should be in the directory you cloned to, if not, retry previous 2 steps
- Run `pip install -r requirements.txt` to download all needed dependencies
- Navigate to the **Models** file, which will include all models we used, each as a file with its results
- Upon entering a file, you will notice there exists soem CSV files there with certain names, the names make sense with respect to the IPYNB file that has the model testing on it
- They include results of testing in case user wants to view resutls without running teh code themselves
- The results of best model are already extensively seen, in case someone wants to see how other models with different parameters fared, at least in cross validation stage, they can view this file
- To run the code, you can simply run the IPYNB fiel yourself, or look at results that are already there from previous run, they will produce the same output due to random seed

# Initial Results
We started with cleaning the feature set of the dataset found in Models, logistic, and finding most relevant features to work on

So far only *Logistic Regression* was tested, with it giving 0.8718 accuracy :
<img width="506" height="165" alt="image" src="https://github.com/user-attachments/assets/54e37e89-c6a5-415b-ac59-b53a712c507f" />

Multiple degrees and hyperparamter combinations were tried, including different solvers, and regularization techniques.
By using PCA we determined our data and output are not linearly seperable:


<img width="433" height="335" alt="image" src="https://github.com/user-attachments/assets/37d7c34e-1f11-433e-a3f5-59e060e24b81" />









We will be seeking to work on more complex, non-linear models such as **Random Forests**, **SVM**, **XGBoost**, and **Neural Networks**

We will also explore more data sets, such as the ones below:
#### Raw data:
##### Full parkinsons (not early)
- https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127
- https://github.com/SJTU-YONGFU-RESEARCH-GRP/Parkinson-Patient-Speech-Dataset?

##### Preprocessed:
**For early PD**:
- https://archive.ics.uci.edu/dataset/301/parkinson%2Bspeech%2Bdataset%2Bwith%2Bmultiple%2Btypes%2Bof%2BAudio%2Brecordings 
- https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
- https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification
Diagnosed: 
- https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set
- https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features

With this much data, and promising results from the first dataset, we can be almost sure that our project is feasable. With the rich features of the preprocessed data, we are able to find the most relevant results, in which we will be able to use those to determine what we extract from the voice samples to see if our extraction and prediction process is working as intended.

Will keep this updated as we go.
