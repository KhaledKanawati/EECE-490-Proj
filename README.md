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
- Open the files usign an IDE, they should be in the directory you cloned to, if not, retry steps 2 and 3
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

Will keep this updated as we go.
