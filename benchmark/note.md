# July 31, branch: fastergpu/master/torch/numpy
Matrix 100 x 100, Float32
* CPU, matmul -> 39us/-/62us/300us
* CPU, star -> 367ms/366ms/4.9ms/298ms
* GPU, matmul -> 67us/-/83us/-
* GPU, star -> 59ms(145ms for reduction kernel)/93ms/1.13ms/-

## GPU star, fastergpu, master, torch
M40 100: 12ms/18ms/320us
TitanV 100: 3.1ms/3.0ms/127us
TitanV 300: 228ms/264ms/2.2ms
TitanV 800: -/-/71ms
