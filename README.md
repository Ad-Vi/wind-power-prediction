# wind-power-prediction

This is a project given in the context of [Introduction To Machine Learning course of the University of Liège](https://www.programmes.uliege.be/cocoon/20222023/cours/ELEN0062-1.html) The [statement](https://github.com/Ad-Vi/wind-power-prediction/blob/main/statement.md) is available in this repo.  
This project is about predicting the wind power production of a wind farm in Australia. A dataset is given over 10 zones and the goal is to predict the power production of the wind farm for the next 24 hours in those 10 zones.  
A contest is held between the students on gradescope and the goal is to have the smaller error possible. The error is calculated using the mean absolute error (MAE) between the predicted and the real values over all the zones - *MAE_GLOB* - and on zone one only -*MAE_Z1*- .

## Run the code

The code is located in the file `submission.py`. It performs the following steps:

1. Load the data
2. Perform or not feature extraction
3. Split the data
4. Train the model
5. Predict the values
6. Calculate the error
7. Write the predictions in submission files

**Usage**:
Run it with python3 with the following parameters.

| Parameter   | Short name  | Description | Values | Default |
| ----------- | ----------- | ----------- | ----------- | --------- |
| `--display` | `-d` | Whether information are print in the terminal during the computing | Boolean | False |
| `--test_size` | `-t` | Proportion of the data used to test our model | float | 0.0 |

## Submissions on gradescope

Before the submission ʕ•ᴥ•ʔ ʕ•ᴥ•ʔ models were not trained on all the data, but only on the last zone due to an error of implementation.  
Expected error is Mean Absolute error (MAE), exepted when MSE is written.  

| Position |     name      |     MAE_GLOB         |      MAE_Z1          |   method                      | calculation time (s)| Expected error (%)|
|:---------|:--------------|:--------------------:|:--------------------:|:--------------------------:|--------------------:|------------------:|
|    27    |    ʕ•ᴥ•ʔ      | 0.2780296804948635   | 0.2707220813681427   |  [mean](https://github.com/Ad-Vi/wind-power-prediction/commit/e66df5a3bb3429b5176c79e960009aba96769c51)                        | | |
|   23     | (｡◕‿◕｡)       | 0.20197477113320383  |0.16355695113016427   |  [kNN with k=10](https://github.com/Ad-Vi/wind-power-prediction/commit/5a5865a4be8c86ada9f4448f34f1df1dcee06f02)               | | |
|22        | ʕ•ᴥ•ʔ         | 0.19558727990167069   |0.17447651350699664  |  [kNN with k = 100](https://github.com/Ad-Vi/wind-power-prediction/commit/b9dda65d8e58a1123ce24bc95e00bba33c18ad51)            | | |
|23        | ʕ•ᴥ•ʔ         | 0.19589805746468192   |0.17604443622797977  |  [10 bagging kNN with k = 100](https://github.com/Ad-Vi/wind-power-prediction/commit/2514582fcd6bae5990b791552fca75906df3f4b7) | | |
|13        | ʕ•ᴥ•ʔ         | 0.18537202141293346   |0.14236585017869863  |  [Random forest 100 trees](https://github.com/Ad-Vi/wind-power-prediction/commit/a5ec8b259edc48e0fa71416344295b5a6e02da73)     | 169.53| |
|29        | should be shitty| 0.20496786870346928 | 0.1665694056025948 | [RF, 100 trees, with univariate Feature extraction](https://github.com/Ad-Vi/wind-power-prediction/commit/b651a4961b01af1c68197c22656379f0f890afdc) | 85.498 | |
| 38 | ʕ•ᴥ•ʔ | 0.21508922377477843 | 0.1813066592111945 | [RF, 100 trees, test 10%, correlation FE](https://github.com/Ad-Vi/wind-power-prediction/commit/640a4d267f49b9e14db8884820abef1d5dc08538) | 46.8149 |5.247 (MSE)|
| 35 | ʕ•ᴥ•ʔ | 0.20947656094651754 | 0.17472517235573073 | [RF, 100 trees, correlation FE](https://github.com/Ad-Vi/wind-power-prediction/commit/de034dbcf3b020e13c0446706c180f096b62802f) | 49.909 | |
|38 | ʕ•ᴥ•ʔ |0.2138090495818373 | 0.17895749096035105 | [RF, 500 trees, test 10%, correlation FE](https://github.com/Ad-Vi/wind-power-prediction/commit/20baa45a03f6de5639ef46c9bec5c5e6355093be) | 244.576 | 5.1641 (MSE)|
|36 | ʕ•ᴥ•ʔ | 0.20955215554930665 | 0.172885499857228 |[RF, 500 trees, test 10%](https://github.com/Ad-Vi/wind-power-prediction/commit/b290df373bf306117ab3825918756cf18c237d55) | 346.45 | 5.15 (MSE)|
|35 | ʕ•ᴥ•ʔ | 0.20488767291036022 | 0.16671282241238644 | [RF, 500 trees](https://github.com/Ad-Vi/wind-power-prediction/commit/81890a7362821aa8ca47a12e868f9695ada8a077) | 404.994 | |
|27 |ʕ•ᴥ•ʔ ʕ•ᴥ•ʔ|0.1855171110453776|0.14418710077268204 | [RF, 500 trees regressor on all zones](https://github.com/Ad-Vi/wind-power-prediction/commit/5b22d107c31b1a325305914676629c77097fd6f8) |  704.69 | |
|41|ʕ•ᴥ•ʔ|0.24377415359099888|0.23252893098076502| [epsSVR, rbf kernel, test 10%](https://github.com/Ad-Vi/wind-power-prediction/commit/b3c6cd75b3a7d2102d4fe6979dc821bf79818638) | 1015.973 |  7.753 (MSE)|
|40|ʕ•ᴥ•ʔ|0.1956323910427123|0.1664162858748583|RF, 100 trees, test 10%, correlation FE, MAE|109.398|13.82|

All those submission code and files can be found in the repo by chosing the corresponding commit (click on the method name for a submission above).

## References

<https://scikit-learn.org/stable/supervised_learning.html#supervised-learning>  
<https://vishalramesh.substack.com/p/feature-selection-correlation-and-p-value-da8921bfb3cf?s=w>
