# wind-power-prediction

This is a project given in the context of [Introduction To Machine Learning course of the University of Liège](https://www.programmes.uliege.be/cocoon/20222023/cours/ELEN0062-1.html) The [statement](https://github.com/Ad-Vi/wind-power-prediction/blob/main/statement.md) is available in this repo.  
This project is about predicting the wind power production of a wind farm in Australia. A dataset is given over 10 zones and the goal is to predict the power production of the wind farm for the next 24 hours in those 10 zones.  
A contest is held between the students on gradescope and the goal is to have the smaller error possible. The error is calculated using the mean absolute error (MAE) between the predicted and the real values over all the zones - *MAE_GLOB* - and on zone one only -*MAE_Z1*- .

## Tries on gradescope

| Position |     name      |     MAE_GLOB         |      MAE_Z1          |   method         |
|:---------|:--------------|:--------------------:|:--------------------:|:----------------:|
|    27    |    ʕ•ᴥ•ʔ      | 0.2780296804948635   | 0.2707220813681427   |  mean            |
|   23     | (｡◕‿◕｡)       | 0.20197477113320383  |0.16355695113016427   |  kNN with k=10   |
|22        |	ʕ•ᴥ•ʔ      | 0.19558727990167069   |0.17447651350699664  |  kNN with k = 100 |
