# wind-power-prediction

This is a project given in the context of [Introduction To Machine Learning course of the University of Liège](https://www.programmes.uliege.be/cocoon/20222023/cours/ELEN0062-1.html) The [statement](https://github.com/Ad-Vi/wind-power-prediction/blob/main/statement.md) is available in this repo.  
This project is about predicting the wind power production of a wind farm in Australia. A dataset is given over 10 zones and the goal is to predict the power production of the wind farm for the next 24 hours in those 10 zones.  
A contest is held between the students on gradescope and the goal is to have the smaller error possible. The error is calculated using the mean absolute error (MAE) between the predicted and the real values over all the zones - *MAE_GLOB* - and on zone one only -*MAE_Z1*- .

## Tries on gradescope

| Position |     name      |     MAE_GLOB         |      MAE_Z1          |   method                      | calculation time (s)| Expected error (%)|
|:---------|:--------------|:--------------------:|:--------------------:|:--------------------------:|--------------------:|------------------:|
|    27    |    ʕ•ᴥ•ʔ      | 0.2780296804948635   | 0.2707220813681427   |  mean                        | | |
|   23     | (｡◕‿◕｡)       | 0.20197477113320383  |0.16355695113016427   |  kNN with k=10               | | |
|22        | ʕ•ᴥ•ʔ         | 0.19558727990167069   |0.17447651350699664  |  kNN with k = 100            | | |
|23        | ʕ•ᴥ•ʔ         | 0.19589805746468192   |0.17604443622797977  |  10 bagging kNN with k = 100 | | |
|13        | ʕ•ᴥ•ʔ         | 0.18537202141293346   |0.14236585017869863  |  Random forest 100 trees     | 169.53| |
|29        | should be shitty| 0.20496786870346928 | 0.1665694056025948 | RF, 100 trees, with univariate Feature extraction | 85.4989275932312 | |
| 38 | ʕ•ᴥ•ʔ | 0.21508922377477843 | 0.1813066592111945 | RF, 100 trees, test 10%, correlation FE | 46.81495785713196 |5.247 |
| 35 | ʕ•ᴥ•ʔ | 0.20947656094651754 | 0.17472517235573073 | RF, 100 trees, correlation FE | 49.90913796424866 | |
|38 | ʕ•ᴥ•ʔ |0.2138090495818373 | 0.17895749096035105 | RF, 500 trees, test 10%, correlation FE | 244.5769236087799 | 5.1641 |

## References

<https://vishalramesh.substack.com/p/feature-selection-correlation-and-p-value-da8921bfb3cf?s=w>
