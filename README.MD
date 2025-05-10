# NumC Class

The `numc` class is a C++ template class designed to handle multi-dimensional arrays, inspired by NumPy arrays in Python. It provides support for operations such as slicing, broadcasting, element-wise operations, matrix multiplication, and aggregation functions.

## Features

- Multi-dimensional array support
- Slicing and indexing
- Broadcasting for operations on arrays of different shapes
- Element-wise arithmetic operations
- Matrix multiplication
- Aggregation operations (e.g., sum, max, min) along axes
- Reshaping and squeezing dimensions

## Usage

To use the `numc` class, include the `numc.h` header file in your C++ program and instantiate `numc` with your desired data type.

### Example

```cpp
#include "numc.h"
#include <iostream>

int main() {
    int shape[] = {2, 3};
    numc<int> a(2, shape);
    // Initialize array elements
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) {
            a[i][j] = i * 3 + j;
        }
    }
    a.print();
    return 0;
}
```

## Methods

- **Constructor**: `numc(int dim, const int* shape, dtype* source = nullptr)`
    - Creates a new `numc` object with specified dimensions and shape. Optionally uses an existing array if `source` is provided.
- **Copy Constructor**: `numc(const numc & source)`
    - Creates a deep copy of an existing `numc` object.
- **Destructor**: `~numc()`
    - Frees allocated memory for shape and array (if not temporary).
- **Assignment Operator**: `numc & operator= (const numc & source)`
    - Assigns values from another `numc` object, handling temporary and non-temporary cases.
- **Indexing Operator**: `numc operator[](int idx)`
    - Returns a sub-array at the specified index along the first dimension.
- **Slice**: `numc slice(int from, int to, int step)`
    - Returns a sliced sub-array along the first dimension.
- **Get Element**: `dtype & get(int idx = 0)`
    - Returns a reference to an element at the specified flattened index.
- **Broadcast**: `const dtype* broadcast(const numc& source, const int& idx)`
    - Maps an index to the corresponding element in a broadcasted array.
- **Broadcast Check**: `bool broadcast_check(const numc & source)`
    - Verifies if broadcasting is possible with another `numc` object.
- **Broadcast Shape**: `int* broadcast_shape(const numc & source)`
    - Computes the resulting shape after broadcasting with another `numc` object.
- **Global Operation**: `dtype global_op(dtype (*fptr)(dtype,dtype), bool init_to_first_term = false)`
    - Applies a global operation (e.g., sum, max) across all elements.
- **Global Operation with Index**: `dtype global_op(dtype (*fptr)(dtype,dtype,int), bool init_to_first_term = false)`
    - Applies a global operation with index awareness.
- **Pointwise Operation**: `numc p_op(dtype (*fptr)(dtype))`
    - Applies a pointwise operation to each element.
- **Pointwise Operation with Index**: `numc p_op(dtype (*fptr)(dtype,int))`
    - Applies a pointwise operation with index awareness.
- **Pointwise Broadcast Operation**: `numc ptp_broadcast_op(const numc & source, dtype (*fptr)(dtype,dtype))`
    - Applies a pointwise operation with broadcasting.
- **Pointwise Broadcast Operation with Index**: `numc ptp_broadcast_op(const numc & source, dtype (*fptr)(dtype,dtype,int))`
    - Applies a pointwise operation with broadcasting and index awareness.
- **Axis Operation**: `numc axis_op(int axis, dtype (*fptr)(dtype,dtype), bool init_to_first_term = false)`
    - Applies an operation along a specified axis.
- **Axis Operation with Index**: `numc axis_op(int axis, dtype (*fptr)(dtype,dtype,int), bool init_to_first_term = false)`
    - Applies an operation along a specified axis with index awareness.
- **Axis Expand**: `numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype))`
    - Expands an axis to a target size using a provided function.
- **Axis Expand with Index**: `numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype,int))`
    - Expands an axis with index-aware function.
- **Arithmetic Operators**: `numc operator+`, `operator-`, `operator*`, `operator/ (const numc & source)`
    - Performs element-wise arithmetic with broadcasting.
- **Matrix Multiplication**: `numc matmul(const numc & source)`
    - Performs matrix multiplication with broadcasting support.
- **Expand Dimensions**: `void expand_dims(int axis)`
    - Adds a new dimension at the specified axis.
- **Squeeze**: `void squeeze()`
    - Removes dimensions of size 1.
- **Reshape**: `void reshape(int new_dim, int* new_shape)`
    - Reshapes the array to a new dimension and shape.
- **Sum**: `dtype sum()`
    - Computes the sum of all elements.
- **Max**: `dtype max()`
    - Finds the maximum element.
- **Min**: `dtype min()`
    - Finds the minimum element.
- **Axis Sum**: `numc axis_sum(int axis)`
    - Computes the sum along a specified axis.
- **Axis Max**: `numc axis_max(int axis)`
    - Finds the maximum along a specified axis.
- **Axis Min**: `numc axis_min(int axis)`
    - Finds the minimum along a specified axis.
- **Print**: `void print()`
    - Prints the array's dimensions, shape, steps, and elements.

## Dependencies

- C++11 or later
- Standard Template Library (STL)

## Notes

- Uses raw pointers for memory management; ensure proper handling to avoid memory leaks.
- Some operations throw `invalid_argument` exceptions for shape mismatches or invalid conditions.