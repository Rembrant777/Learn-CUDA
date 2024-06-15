# Introduction to Pack and Split Algorithm 
The 'pack' and 'split' algorithms are fudamental techniques used in parallel computing for rearranging data based on certain conditions or predicates. 

These algorithms are particularly useful in applications like parallel processing on GPUs, where efficient data manipulation is crucial for performance. 

## Pack Algorithm 
The pack algorithm compacts elements of an array that satisfy a given predicate(condition) into a 
contiguous block of memory, while ignoring or discarding elements that do not satisfy the predicate. 

### Steps of Pack Algorithm
* Preciate Calculation: 
Create a predicate array where each element is a binary indicator(0 or 1) indicating whether the correspoinding element in the input array satisifies the condition. 

* Prefix Sum(Scan):
Perform an exclusive prefix sum(scan) on the predicate array to determine the target position of each valid element in the output array. 

* Scatter: 
Using the results of the scan, copy the elements that satisfy the predicate to their corresponding positions in the output array. 

## Split Algorithm 
The split algorithm partitions an array into two parts based on a predicate. One part contians elements that satisfy the predicate, and the other contains elements that do not. 

### Steps of Split Algorithm 
* Preciate Calculation: 
Similar to the pack algorithm, create a predicate array 

* Prefix Sum(Scan):
Compute the prefix sum of the predicate array. 

* Address Calculation:
Calculate the target positions for both parts of the split(satsifying and non-satisfying elements).

* Scatter:
Place elements into their respective position in two separate output arrays or the same array. 