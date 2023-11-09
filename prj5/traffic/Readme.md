I took the model generation code in the digit recognition example from the lecture as a base. In order to find the best configuration, I will run the model creation combining different values for these parameters:

* Number of convolutional+pooling layers (num_conv)
* Sizes of filters (conv_filter_size) and kernel (conv_kernel) for convolutional layers
* Pool size for pooling layers (pool_size)
* Numbers (hidden_num) and sizes (hidden_dense) of pooling layers
* Dropout (hidden_dropout)

To have some starting numbers, I ran the small dataset (3) with the following combinations:

* num_conv = [1, 2]
* conv_filter_size = [8, 16, 32, 64, 128, 256]
* conv_kernel = [3x3, 5x5]
* pool_size = [2x2, 3x3]
* hidden_num = [0, 1, 2]
* hidden_dense = [64, 128, 256]
* hidden_dropout = [0.1, 0.3, 0.5]

From all combinations, I filtered those with accuracy>0.99. I could narrow down some parameters (num_conv, conv_kernel,pool_size) which where constant in all these best accuracy results.

* num_conv = 2
* conv_filter_size = [128, 256] (sizes <128 don't work quite well)
* conv_kernel = 5x5 (only one 3x3 amongst all >0.99 results)
* pool_size = 2x2
* hidden_num = [0, 1] (a 2nd hidden layer doesn't seem to help much)
* hidden_dense = [64, 128, 256]
* hidden_dropout = [0.1, 0.3, 0.5]

I run the test now for all 43. I couldn't get any accuracy > 0.99, but got 2 results with acc >0.98 (removing fixed parameters from previous run):

|conv_filter_size|hidden_num|hidden_dense|hidden_dropout|loss|accuracy|
|---|---|---|---|---|---|
|256|1|128|0.3|0.0738|0.9841|
|256|1|256|0.3|0.1103|0.9800|
|128|1|128|0.3|0.1216|0.9770|
|128|1|128|0.1|0.1448|0.9760|
|256|1|256|0.5|0.1194|0.9756|
|128|1|128|0.5|0.1087|0.9721|

It seems one hidden layer definitely gets the best results. For other parameters, different combinations seem to have similar results. I am running an additional test, fixing hidden_num=1 and changing conv_filter_size, hiden_density and hidden_dropout. Results with accuracy >0.97 follow:

|conv_filter_size|hidden_dense|hidden_dropout|loss|accuracy|
|---|---|---|---|---|
|256|256|0.5|0.1002|0.9791|
|256|128|0.3|0.1204|0.9742|
|128|256|0.3|0.1296|0.9738|
|128|256|0.5|0.1197|0.9737|
|256|128|0.1|0.1546|0.9735|
|256|128|0.5|0.1138|0.9733|

So, I will pick values 2,256,5,2,1,128,0.3 (which has been #1 and #2 in last runs) as final pick, but differences are not much big.

