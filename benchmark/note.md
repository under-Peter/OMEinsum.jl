# July 31, branch: fastergpu/master/torch/numpy
Matrix 100 x 100, Float32
* CPU, matmul -> 39us/-/62us/300us
* CPU, star -> 367ms/366ms/2.93ms/298ms
* GPU, matmul -> 67us/-/83us/-
* GPU, star -> 59ms(145ms for reduction kernel)/93ms/1.13ms/-
