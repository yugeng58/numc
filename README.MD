# numc: A NumPy-like Multi-dimensional Array Library for C++

## Introduction

`numc` is a C++ template class designed to handle multi-dimensional arrays, offering functionality akin to Python's NumPy library ([NumPy Documentation](https://numpy.org/doc/stable/)). It provides dynamic shape management, broadcasting, and a comprehensive set of mathematical operations, making it a robust tool for numerical computations in C++. Implemented as a single header file, `numc` is easy to integrate into existing C++ projects, catering to developers who need efficient array operations with a familiar NumPy-like interface.

## Features

- **Multi-dimensional Arrays**: Supports arrays of any dimension with dynamic shape specification at runtime.
- **Indexing and Slicing**: Access elements or subarrays using intuitive indexing and slicing operations.
- **Broadcasting**: Enables operations between arrays of different shapes, following NumPy-style broadcasting rules.
- **Mathematical Operations**: Includes element-wise operations (addition, subtraction, multiplication, division) and matrix multiplication.
- **Reduction Operations**: Computes aggregates like sum, maximum, and minimum across the entire array or specific axes.
- **Shape Manipulation**: Offers methods to reshape, expand, or squeeze array dimensions.
- **Iterators**: Provides random-access iterators for traversing array elements.
- **Memory Management**: Handles owned and temporary arrays, with automatic cleanup for owned data.
- **Template Support**: Works with various data types (e.g., `int`, `float`, `double`) via C++ templates.
- **Parallelization**: Utilizes OpenMP for parallel execution of operations, enhancing performance on multi-core systems.

## Getting Started

### Including `numc`
To use `numc`, include the header file in your C++ source code:

```cpp
#include "numc.h"
```

### Compilation
Compile your program with a C++11-compliant compiler and enable OpenMP for parallelization. For example, using `g++`:

```sh
g++ -std=c++11 -fopenmp your_program.cpp -o your_program
```

Ensure your compiler supports OpenMP, as it is critical for performance in operations like array initialization and element-wise computations.

## API Reference

Below is a detailed description of the `numc` class and its methods, organized for clarity.

### Constructors

#### Default Constructor
```cpp
numc(int dim, const int* shape, dtype* source = nullptr)
```
- **Parameters**:
  - `dim`: Number of dimensions (non-negative).
  - `shape`: Array specifying the size of each dimension.
  - `source`: Optional pointer to existing data. If provided, the array does not own the data (`temp = true`).
- **Behavior**: Allocates memory for the shape and, if `source` is null, for the data array, initializing it to zero using OpenMP parallelization. Throws exceptions for invalid dimensions or shapes.

#### Copy Constructor
```cpp
numc(const numc& source)
```
- **Parameters**:
  - `source`: The `numc` object to copy.
- **Behavior**: Performs a deep copy of the source array, allocating new memory for both shape and data.

#### Move Constructor
```cpp
numc(numc&& other)
```
- **Parameters**:
  - `other`: The `numc` object to move from.
- **Behavior**: Transfers resources from `other`. If `other.temp == false`, it takes ownership of `other.arr`. If `other.temp == true`, it allocates a new array and copies the data. Sets `other.shape` and `other.arr` to `nullptr`. **Note**: The current implementation sets `temp = true` for the new object, which may cause memory leaks as the data is not deleted in the destructor. This appears to be a potential bug.

### Assignment Operator
```cpp
numc& operator=(const numc& source)
```
- **Parameters**:
  - `source`: The `numc` object to assign from.
- **Behavior**: If the current array is temporary (`temp = true`), it checks shape compatibility and copies data. Otherwise, it deallocates existing memory, allocates new memory, and copies the data. Returns a reference to the current object.

### Destructor
```cpp
~numc()
```
- **Behavior**: Deletes the shape array and, if `temp = false`, the data array, ensuring proper memory cleanup.

### Iterator
```cpp
class iterator { /* ... */ };
```
- **Description**: A random-access iterator for traversing array elements, supporting operations like `++`, `--`, `+`, `-`, and `[]`.

#### begin()
```cpp
iterator begin()
```
- **Returns**: An iterator pointing to the first element of the array.

#### end()
```cpp
iterator end()
```
- **Returns**: An iterator pointing past the last element of the array.

### Indexing and Slicing

#### operator[]
```cpp
numc operator[](int idx)
```
- **Parameters**:
  - `idx`: Index along the first dimension.
- **Returns**: A new `numc` object representing the subarray at `idx`. Throws an exception if `idx` is out of bounds.

#### slice
```cpp
numc slice(int from, int to, int step)
```
- **Parameters**:
  - `from`: Starting index (inclusive).
  - `to`: Ending index (exclusive).
  - `step`: Step size between elements.
- **Returns**: A new `numc` object containing the sliced data. Throws an exception for invalid parameters.

#### get
```cpp
dtype& get(int idx = 0)
```
- **Parameters**:
  - `idx`: Index in the flattened array (default is 0).
- **Returns**: A reference to the element at `idx`. Throws an exception if `idx` is out of bounds.

### Broadcasting

#### broadcast
```cpp
const dtype* broadcast(const numc& source, const int& idx) const
```
- **Parameters**:
  - `source`: The source `numc` object.
  - `idx`: Index in the broadcasted array.
- **Returns**: A pointer to the corresponding element in `source` based on broadcasting rules.

#### broadcast_check
```cpp
bool broadcast_check(const numc& source) const
```
- **Parameters**:
  - `source`: The `numc` object to check compatibility with.
- **Returns**: `true` if broadcasting is possible, `false` otherwise.

#### broadcast_shape
```cpp
int* broadcast_shape(const numc& source) const
```
- **Parameters**:
  - `source`: The `numc` object to broadcast with.
- **Returns**: A new array containing the broadcasted shape (must be deleted by the caller).

### Operations

#### global_op
```cpp
dtype global_op(dtype (*fptr)(dtype, dtype), bool init_to_first_term = false)
dtype global_op(dtype (*fptr)(dtype, dtype, int), bool init_to_first_term = false)
```
- **Parameters**:
  - `fptr`: Function pointer to the operation (e.g., sum, max).
  - `init_to_first_term`: If `true`, initializes the result to the first element.
- **Returns**: The result of applying `fptr` to all elements.

#### p_op
```cpp
numc p_op(dtype (*fptr)(dtype))
numc p_op(dtype (*fptr)(dtype, int))
```
- **Parameters**:
  - `fptr`: Function pointer to the operation (e.g., square, negate).
- **Returns**: A new `numc` object with the operation applied to each element.

#### ptp_broadcast_op
```cpp
numc ptp_broadcast_op(const numc& source, dtype (*fptr)(dtype, dtype))
numc ptp_broadcast_op(const numc& source, dtype (*fptr)(dtype, dtype, int))
```
- **Parameters**:
  - `source`: The `numc` object to operate with.
  - `fptr`: Function pointer to the operation (e.g., add, multiply).
- **Returns**: A new `numc` object with the operation applied, respecting broadcasting rules.

#### axis_op
```cpp
numc axis_op(int axis, dtype (*fptr)(dtype, dtype), bool init_to_first_term = false)
numc axis_op(int axis, dtype (*fptr)(dtype, dtype, int), bool init_to_first_term = false)
```
- **Parameters**:
  - `axis`: The axis to operate along.
  - `fptr`: Function pointer to the operation.
  - `init_to_first_term`: If `true`, initializes the result to the first element.
- **Returns**: A new `numc` object with the operation applied along `axis`.

#### axis_expand
```cpp
numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype))
numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype, int))
```
- **Parameters**:
  - `axis`: The axis to expand (must have size 1).
  - `axis_target`: The target size for the expanded axis.
  - `fptr`: Function pointer to compute new elements.
- **Returns**: A new `numc` object with the expanded dimension.

### Mathematical Operators

#### operator+
```cpp
numc operator+(const numc& source)
```
- **Parameters**:
  - `source`: The `numc` object to add.
- **Returns**: A new `numc` object with element-wise addition, supporting broadcasting.

#### operator-
```cpp
numc operator-(const numc& source)
```
- **Parameters**:
  - `source`: The `numc` object to subtract.
- **Returns**: A new `numc` object with element-wise subtraction, supporting broadcasting.

#### operator*
```cpp
numc operator*(const numc& source)
```
- **Parameters**:
  - `source`: The `numc` object to multiply.
- **Returns**: A new `numc` object with element-wise multiplication, supporting broadcasting.

#### operator/
```cpp
numc operator/(const numc& source)
```
- **Parameters**:
  - `source`: The `numc` object to divide by.
- **Returns**: A new `numc` object with element-wise division, supporting broadcasting. Throws an exception on division by zero.

#### matmul
```cpp
numc matmul(const numc& source)
```
- **Parameters**:
  - `source`: The `numc` object to multiply with.
- **Returns**: A new `numc` object with the matrix product, supporting broadcasting for higher dimensions.

### Shape Manipulation

#### expand_dims
```cpp
void expand_dims(int axis)
```
- **Parameters**:
  - `axis`: The position to insert the new dimension.
- **Behavior**: Adds a dimension of size 1 at `axis`.

#### squeeze
```cpp
void squeeze()
```
- **Behavior**: Removes all dimensions of size 1 from the array.

#### reshape
```cpp
void reshape(int new_dim, const int* new_shape)
```
- **Parameters**:
  - `new_dim`: The new number of dimensions.
  - `new_shape`: Pointer to the new shape array.
- **Behavior**: Reshapes the array, ensuring the total number of elements remains unchanged.

### Reduction Operations

#### sum
```cpp
dtype sum()
```
- **Returns**: The sum of all elements in the array.

#### max
```cpp
dtype max()
```
- **Returns**: The maximum value in the array.

#### min
```cpp
dtype min()
```
- **Returns**: The minimum value in the array.

#### axis_sum
```cpp
numc axis_sum(int axis)
```
- **Parameters**:
  - `axis`: The axis to sum along.
- **Returns**: A new `numc` object with the sums along `axis`.

#### axis_max
```cpp
numc axis_max(int axis)
```
- **Parameters**:
  - `axis`: The axis to operate along.
- **Returns**: A new `numc` object with the maximum values along `axis`.

#### axis_min
```cpp
numc axis_min(int axis)
```
- **Parameters**:
  - `axis`: The axis to operate along.
- **Returns**: A new `numc` object with the minimum values along `axis`.

### Debugging

#### print
```cpp
void print()
```
- **Behavior**: Prints the dimensions, shape, and elements of the array, useful for debugging.

## Examples

The following examples demonstrate common use cases for `numc`.

### 1. Creating and Initializing an Array
```cpp
#include "numc.h"
#include <iostream>

int main() {
    int shape[] = {2, 3};
    numc<double> arr(2, shape);
    arr[0][0] = 1.0;
    arr[0][1] = 2.0;
    arr[0][2] = 3.0;
    arr[1][0] = 4.0;
    arr[1][1] = 5.0;
    arr[1][2] = 6.0;
    arr.print();
    return 0;
}
```
**Output**:
```
dim = 2
shape = (2,3)
steps = (3,1)
idx : [0,0] = 1
idx : [0,1] = 2
idx : [0,2] = 3
idx : [1,0] = 4
idx : [1,1] = 5
idx : [1,2] = 6
```

### 2. Slicing
```cpp
numc<double> slice = arr.slice(0, 1, 1);
slice.print();
```
**Output**:
```
dim = 2
shape = (1,3)
steps = (3,1)
idx : [0,0] = 1
idx : [0,1] = 2
idx : [0,2] = 3
```

### 3. Broadcasting
```cpp
numc<double> arr2(2, shape);
arr2 = arr + 1.0; // Adds 1.0 to each element
arr2.print();
```
**Output**: (Each element incremented by 1.0)
```
dim = 2
shape = (2,3)
steps = (3,1)
idx : [0,0] = 2
idx : [0,1] = 3
idx : [0,2] = 4
idx : [1,0] = 5
idx : [1,1] = 6
idx : [1,2] = 7
```

### 4. Matrix Multiplication
```cpp
int shape1[] = {2, 3};
int shape2[] = {3, 2};
numc<double> a(2, shape1);
numc<double> b(2, shape2);
// Initialize a and b (example values)
a[0][0] = 1.0; a[0][1] = 2.0; a[0][2] = 3.0;
a[1][0] = 4.0; a[1][1] = 5.0; a[1][2] = 6.0;
b[0][0] = 1.0; b[0][1] = 2.0;
b[1][0] = 3.0; b[1][1] = 4.0;
b[2][0] = 5.0; b[2][1] = 6.0;
numc<double> c = a.matmul(b);
c.print();
```
**Output**:
```
dim = 2
shape = (2,2)
steps = (2,1)
idx : [0,0] = 22
idx : [0,1] = 28
idx : [1,0] = 49
idx : [1,1] = 64
```

### 5. Reduction Operations
```cpp
double sum = arr.sum();
double max_val = arr.max();
double min_val = arr.min();
std::cout << "Sum: " << sum << std::endl;
std::cout << "Max: " << max_val << std::endl;
std::cout << "Min: " << min_val << std::endl;
```
**Output**:
```
Sum: 21
Max: 6
Min: 1
```

### 6. Axis Operations
```cpp
numc<double> sum_axis = arr.axis_sum(1); // Sum along axis 1
sum_axis.print();
```
**Output**:
```
dim = 2
shape = (2,1)
steps = (1,1)
idx : [0,0] = 6
idx : [1,0] = 15
```

## Dependencies and Compilation

### Requirements
- **C++11 or Higher**: `numc` uses modern C++ features like templates and exception handling.
- **OpenMP**: Required for parallelization of operations like array initialization and element-wise computations.

### Compilation
Compile your program with a C++11-compliant compiler and enable OpenMP. Example with `g++`:
```sh
g++ -std=c++11 -fopenmp your_program.cpp -o your_program
```
Verify that your compiler supports OpenMP, as some operations rely on it for performance. For other compilers, consult their documentation (e.g., [GCC OpenMP](https://gcc.gnu.org/onlinedocs/libgomp/)).

## Notes

- **Runtime Checks**: `numc` performs extensive runtime checks for shape compatibility, index bounds, and valid operations, throwing exceptions (e.g., `std::invalid_argument`, `std::out_of_range`) when errors occur. Users should handle these exceptions to ensure robust programs.
- **Memory Management**: The library distinguishes between owned arrays (`temp = false`) and temporary arrays (`temp = true`). Owned arrays are automatically deallocated in the destructor, while temporary arrays rely on external memory management.
- **Potential Memory Issue**: The move constructor sets `temp = true` for the new object, even when it takes ownership of data or allocates a new array. This may prevent the data from being deleted in the destructor, potentially causing memory leaks. Users should verify this behavior and consider fixing the implementation.
- **Performance**: OpenMP parallelization enhances performance for large arrays, but its effectiveness depends on the system's multi-core capabilities and compiler configuration. Ensure OpenMP is enabled for optimal results.
- **Comparison with Other Libraries**: Compared to libraries like Blitz++ or Boost.MultiArray, `numc` offers a NumPy-like API and OpenMP support, making it suitable for projects requiring both ease of use and performance. However, it may lack the compile-time optimizations of Blitz++.

## Comparison with Other Libraries

| Library            | Multi-dimensional Support | Broadcasting | Mathematical Operations | Parallelization | Maintenance |
|--------------------|---------------------------|--------------|-------------------------|-----------------|-------------|
| **numc**           | Yes                       | Yes          | Yes (element-wise, matmul) | OpenMP         | Active (assumed) |
| **Blitz++**        | Yes                       | Limited      | Yes                     | None            | Limited     |
| **Boost.MultiArray** | Yes                     | No           | Limited                 | None            | Active      |

- **Blitz++**: Offers high performance via template metaprogramming but has limited maintenance ([Blitz++ Documentation](https://github.com/blitzpp/blitz)).
- **Boost.MultiArray**: Provides multi-dimensional arrays but lacks arithmetic operations and broadcasting ([Boost.MultiArray](https://www.boost.org/doc/libs/1_83_0/libs/multi_array/doc/index.html)).

`numc` stands out for its NumPy-like interface and parallelization, making it a compelling choice for C++ numerical computing.