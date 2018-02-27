# cdc17-python
Python code for our CDC 2017 paper

## Algorithm
The algorithm simulated here is from “**Multi-Agent Discrete Search with Limited Visibility**” by **H. Ding** and **D. Castanon**, Proc. IEEE Conference on Decision and Control (CDC), 2017.

## Code Structure
* alg.ipynb — Python notebook implementing the algorithm with detailed comments
* batch_job_normal
  * num_of_sink.py — Effect of number of sinks on the runtime of the algorithm
  * num_of_source.py — Effect of number of sources on the runtime of the algorithm
  * sparsity.py — Effect of graph sparsity on the runtime of the algorithm
  * supply.py — Effect of source supply on the runtime of the algorithm
* batch_job_UAV — Simulate a UAV search setting
  * sparsity_UAV.py — Effect of graph sparsity on the runtime of the algorithm
  * supply_UAV.py — Effect of source supply on the runtime of the algorithm
  * example_spatial_field.py — An example UAV spatial search field
