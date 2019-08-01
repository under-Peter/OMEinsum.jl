# July 31, branch: fastergpu/master/torch/numpy
Matrix 100 x 100, Float32
* CPU, matmul -> 222us/-/150us/793us
* CPU, star -> 151ms/366ms/3.03ms/166ms
* GPU, matmul -> 188us/-/241us/-
* GPU, star -> 47ms/93ms/1.37ms/-

## GPU star, fastergpu, master, torch
M40 100: 12ms/18ms/320us
TitanV 100: 3.1ms/3.0ms/127us
TitanV 300: 29ms/264ms/2.2ms
TitanV 800: 1.53s/-/71ms

## fake star 100: torch/numpy
* CPU -> 174ms/202ms/164ms
* GPU -> 48ms/37ms/-
