# wind-power-prediction

This is a project given in the context of [Introduction To Machine Learning course of the University of Liège](https://www.programmes.uliege.be/cocoon/20222023/cours/ELEN0062-1.html) The [statement](https://github.com/Ad-Vi/wind-power-prediction/blob/main/statement.md) is available in this repo.  
This project is about predicting the wind power production of a wind farm in Australia. A dataset is given over 10 zones and the goal is to predict the power production of the wind farm for the next 24 hours in those 10 zones.  
A contest is held between the students on gradescope and the goal is to have the smaller error possible. The error is calculated using the mean absolute error (MAE) between the predicted and the real values over all the zones - *MAE_GLOB* - and on zone one only -*MAE_Z1*- .

## Run the code

The code is located in the file `submission.py`. It performs the following steps:

1. Load the data
2. Perform or not feature extraction
3. Perform or not data scaling
4. Split the data
5. Train the model
6. Predict the values
7. Calculate the error
8. Write the predictions in submission files

**Usage**:
Run it with python3 with the following parameters.

| Parameter   | Short name  | Description | Values | Default |
| ----------- | ----------- | ----------- | ----------- | --------- |
| `--display` | `-d` | Whether information are print in the terminal during the computing | Boolean | False |
| `--test_size` | `-t` | Proportion of the data used to test our model | float | 0.0 |
|`--feature_selection`|`-fs`|Feature selection to apply. None is applied if None. |`None`, `UnivariateVarianceTreshold`, `Correlation`|`None`|

## Tries and results

Before the submission ʕ•ᴥ•ʔ ʕ•ᴥ•ʔ models were not trained on all the data, but only on the last zone due to an error of implementation.  
Expected error is Mean Absolute error (MAE), exepted when MSE is written and for Neural Networks - where it is the models evaluation method output.  

### Submissions on gradescope

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
|40|ʕ•ᴥ•ʔ|0.1956323910427123|0.1664162858748583|[RF, 100 trees, test 10%, correlation FE, MAE](https://github.com/Ad-Vi/wind-power-prediction/commit/c739b9d8a4e07c899da7ceb306cc52c605c05fb6)|109.398|13.82|
|40|oups|0.2556024822676563|0.2663099177565241|[ANN, 3 hidden layers, test 10%, correlation FE](https://github.com/Ad-Vi/wind-power-prediction/commit/a6f6468ee0871737b71df37bb6ec4155f3d15308)| 12.077|7.29|
|38|oups|0.202639853415171|0.17902481850330204|[ANN, 3 hidden layers, test 10%, correlation FE, 150 epochs, minibatch](https://github.com/Ad-Vi/wind-power-prediction/commit/12fde700fc57bb8a27feee1cfe46fb59c2953a8d)| 2892.28|4.59|
|38|ʕ•ᴥ•ʔ|0.1978985906637341|0.17943384394762932|ANN, 3 hidden layers, test 10%, 50 epochs, batch 1/10| 8.73 |5.1577|
|33        |               | 0.18534205128981526  | 0.1433400154576143   | [random forest with 1000 trees](https://github.com/Ad-Vi/wind-power-prediction/tree/91fd948e9ffa8692c191f30a0cd639216553672c) | |
|37 ||0.186403 | 0.14428083125228475 | [random forest with 4 features (speeds) and 100 trees](https://github.com/Ad-Vi/wind-power-prediction/tree/7d5385ee5a9ff26fbe9f00adadda3b578efa192f) ||
|||0.184010|0.140117|[knn 20 neighbors 4 features](https://github.com/Ad-Vi/wind-power-prediction/tree/34a2eb37f6aa246e717273666b9e858bb490d07a)||
|||0.182138|0.144743|[knn 123 neighbors 4 features](https://github.com/Ad-Vi/wind-power-prediction/tree/bc3ccf6c463e9cfa26fa39508dc70534f099aea7)||
|1||0.176793|0.1395687|[kNN adjusted n_neighbors 4 features](https://github.com/Ad-Vi/wind-power-prediction/tree/48cc69f9821a5731b9a60caaab9c3f3b5b197f1e)||
|2||0.176786|0.1395687|[kNN adjusted n_neighbors 4 features](https://github.com/Ad-Vi/wind-power-prediction/tree/e2cc017feac8d477734ca7c8788e87d318309ba6)||
|2||0.176752|0.139393|[kNN adjusted neighbors feature_vec = (speeds + timestamp)](https://github.com/Ad-Vi/wind-power-prediction/tree/30370e7991afe740bc89dd6e3b13c7a1123f676e)
|41|Adrien & Emil|0.3966516599325793|0.2942257636466016|[ANN, 1 per zone, 3HL, test 10%, 50 epochs, batch 10%](https://github.com/Ad-Vi/wind-power-prediction/commit/699ce9f2f304d95a2514efee9eedf2ac4269defe)|18.673|19.22|

Our chosen implementation for the final submission was [kNN adjusted neighbors feature_vec = (speeds + timestamp)](https://github.com/Ad-Vi/wind-power-prediction/tree/30370e7991afe740bc89dd6e3b13c7a1123f676e) which leads to a score of 0.176752 on the public leaderboard.  
The final ranking can be found in the [leaderboard](https://github.com/Ad-Vi/wind-power-prediction/blob/main/leaderboard.md).

## References

<https://scikit-learn.org/stable/supervised_learning.html#supervised-learning>
<https://vishalramesh.substack.com/p/feature-selection-correlation-and-p-value-da8921bfb3cf?s=w>
<https://chat.openai.com/chat>  
<https://towardsdatascience.com/designing-your-neural-networks-a5e4617027ed>  
<https://keras.io/api/models/sequential/>
