#include <iostream>      // For standard input/output operations
#include <stdexcept>     // For standard exception classes
#include <string>        // For string manipulation in error messages
#include <algorithm>     // For std::max and std::min functions
#include <omp.h>         // For OpenMP parallel programming directives
#pragma omp requires unified_shared_memory // Ensures unified shared memory for OpenMP (required for some GPU architectures)

// Template class implementing a multi-dimensional array (ndarray) with parallel operations
// Type parameter 'type' specifies the data type of array elements (e.g., int, float, double)
template <class type>
class ndarray {
private:
    int dim;           // Number of dimensions in the array (e.g., 2 for a matrix, 3 for a tensor)
    int* shape;        // Array storing the size of each dimension (e.g., {2, 3} for a 2x3 matrix)
    int* mod;          // Array storing modulo values for efficient index calculations
    int* step;         // Array storing step sizes for memory layout (used in strided access)
    int* stride;       // Array storing stride values for view operations (e.g., slicing)
    int* offset;       // Array storing offset values for view operations (e.g., starting point of a slice)
    type* arr;         // Pointer to the underlying data array storing the elements
    int size;          // Total number of elements in the array (product of shape dimensions)
    bool is_view;      // Flag indicating if this array is a view of another array's data (avoids deallocation)

    // Copies contents from source array to destination array with bounds checking
    // Parameters:
    //   dest: Destination array to copy into
    //   source: Source array to copy from (can be nullptr, in which case dest is returned unchanged)
    //   length: Number of elements to copy
    // Returns: Pointer to the destination array
    // Throws: std::invalid_argument if dest is null, std::runtime_error for other errors
    inline int* arr_copy(int* dest, const int* source, int length) {
        try {
            if (!dest) {
                throw std::invalid_argument("Destination array is null in arr_copy");
            }
            if (!source) {
                return dest; // No copy needed if source is null
            }
            for (int i = 0; i < length; i++) {
                dest[i] = source[i]; // Copy each element
            }
            return dest;
        } catch (const std::exception& e) {
            throw std::runtime_error("Array copy failed in arr_copy: " + std::string(e.what()));
        }
    }

    // Converts a linear index to a multi-dimensional index
    // Parameters:
    //   linear_idx: Linear index to convert (e.g., 5 in a flattened array)
    //   out_idx: Output array to store the multi-dimensional index (e.g., {1, 2} for a 2D array)
    //   dimensions: Number of dimensions to consider
    // Throws: std::invalid_argument for invalid parameters, std::runtime_error for shape errors
    inline void linear_to_index(int linear_idx, int* out_idx, int dimensions) const {
        try {
            if (!out_idx) {
                throw std::invalid_argument("Output index array is null in linear_to_index");
            }
            if (dimensions <= 0 || dimensions > dim) {
                throw std::invalid_argument("Invalid dimensions in linear_to_index");
            }
            int remaining = linear_idx;
            for (int j = dimensions - 1; j >= 0; j--) {
                if (shape[j] <= 0) {
                    throw std::runtime_error("Invalid shape dimension in linear_to_index");
                }
                out_idx[j] = remaining % shape[j]; // Compute index for current dimension
                remaining /= shape[j]; // Reduce for next dimension
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Linear to index conversion failed: " + std::string(e.what()));
        }
    }

public:
    // Constructor for creating a new array or a view of existing data
    // Parameters:
    //   dim: Number of dimensions
    //   shape: Array specifying size of each dimension
    //   source: Optional pointer to existing data (for views)
    //   stride: Optional array specifying stride for each dimension
    //   offset: Optional array specifying offset for each dimension
    //   step: Optional array specifying step size for each dimension
    // Throws: std::invalid_argument for invalid inputs, std::runtime_error for allocation failures
    explicit ndarray(int dim, const int* shape, type* source = nullptr,
                     const int* stride = nullptr, const int* offset = nullptr,
                     const int* step = nullptr)
            : dim(dim), is_view(false), size(1) {
        try {
            // Validate input parameters
            if (dim <= 0) {
                throw std::invalid_argument("Number of dimensions must be positive");
            }
            if (!shape) {
                throw std::invalid_argument("Shape array is null");
            }

            // Allocate memory for array attributes
            this->shape = new int[dim];
            this->mod = new int[dim];
            this->step = new int[dim];
            this->stride = new int[dim];
            this->offset = new int[dim];

            // Copy shape and compute total size
            arr_copy(this->shape, shape, dim);
            for (int i = 0; i < dim; i++) {
                if (shape[i] <= 0) {
                    throw std::invalid_argument("Shape dimension must be positive at index " + std::to_string(i));
                }
                size *= shape[i]; // Compute total number of elements
            }

            // Initialize mod array for index calculations
            this->mod[0] = 1;
            for (int i = 1; i < dim; i++) {
                this->mod[i] = this->mod[i - 1] * shape[i - 1];
            }

            // Initialize step array (controls memory layout)
            if (!step) {
                this->step[dim - 1] = 1; // Innermost dimension has step 1
                for (int i = dim - 2; i >= 0; i--) {
                    this->step[i] = this->step[i + 1] * shape[i + 1]; // Compute step for outer dimensions
                }
            } else {
                arr_copy(this->step, step, dim);
            }

            // Initialize stride and offset arrays
            for (int i = 0; i < dim; i++) {
                this->stride[i] = stride ? stride[i] : 1; // Default stride is 1
                this->offset[i] = offset ? offset[i] : 0; // Default offset is 0
            }

            // Initialize data array
            if (source) {
                is_view = true; // Mark as view (won't deallocate arr)
                arr = source;   // Use provided data
            } else {
                arr = new type[size]{}; // Allocate and zero-initialize new array
            }
        } catch (const std::exception& e) {
            // Clean up any allocated memory to prevent leaks
            delete[] shape;
            delete[] mod;
            delete[] step;
            delete[] stride;
            delete[] offset;
            delete[] arr;
            throw std::runtime_error("ndarray construction failed: " + std::string(e.what()));
        }
    }

    // Copy constructor that creates a deep copy of the source array
    // Parameters:
    //   source: The source ndarray to copy
    // Uses OpenMP to parallelize the copying of elements
    // Throws: std::runtime_error for copy or index calculation failures
    ndarray(const ndarray<type>& source) : ndarray(source.dim, source.shape) {
        try {
#pragma omp parallel
            {
                // Each thread allocates its own index array
                int* local_index = new int[dim]{};
#pragma omp for
                for (int i = 0; i < size; i++) {
                    try {
                        // Convert linear index to multi-dimensional index
                        linear_to_index(i, local_index, dim);
                        // Copy value from source to this array
                        arr[i] = source.get_value(local_index);
                    } catch (const std::exception& e) {
                        throw; // Propagate exception to outer catch
                    }
                }
                delete[] local_index; // Clean up thread-local memory
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Copy construction failed: " + std::string(e.what()));
        }
    }

    // Destructor that cleans up allocated memory
    // Only deallocates arr if not a view (is_view == false)
    // Logs exceptions to stderr instead of throwing (as per destructor best practices)
    ~ndarray() {
        try {
            if (!is_view) {
                delete[] arr; // Deallocate data array if owned
            }
            delete[] shape;   // Deallocate shape array
            delete[] stride;  // Deallocate stride array
            delete[] offset;  // Deallocate offset array
            delete[] mod;     // Deallocate mod array
            delete[] step;    // Deallocate step array
        } catch (const std::exception& e) {
            std::cerr << "Warning: Exception in destructor: " << e.what() << std::endl;
        }
    }

    // Assignment operator that copies data from source array
    // Parameters:
    //   source: The source ndarray to assign from
    // Returns: Reference to this array
    // Handles both view and non-view cases
    // Throws: std::runtime_error for allocation or copy failures
    ndarray<type>& operator=(const ndarray<type>& source) {
        try {
            if (&source == this) {
                return *this; // Self-assignment check
            }
            if (is_view) {
                // For views, only copy data without reallocating
                element_to_element_mutator(source, [](const type& a,const type& b) { return b; });
                return *this;
            }

            // Clean up existing resources
            delete[] arr;
            delete[] shape;
            delete[] stride;
            delete[] offset;
            delete[] mod;
            delete[] step;

            // Copy dimensions and size
            dim = source.dim;
            size = source.size;

            // Allocate new resources
            arr = new type[source.size];
            shape = new int[source.dim];
            stride = new int[source.dim];
            offset = new int[source.dim];
            mod = new int[source.dim];
            step = new int[source.dim];

            // Initialize arrays
            arr_copy(shape, source.shape, dim);
            for (int i = 0; i < dim; i++) {
                stride[i] = 1; // Default stride
                offset[i] = 0; // Default offset
            }

            // Initialize mod array
            mod[0] = 1;
            for (int i = 1; i < dim; i++) {
                mod[i] = mod[i - 1] * shape[i - 1];
            }

            // Initialize step array
            step[dim - 1] = 1;
            for (int i = dim - 2; i >= 0; i--) {
                step[i] = step[i + 1] * shape[i + 1];
            }

            // Copy data
            element_to_element_mutator(source, [](const type& a,const type& b) { return b; });
            return *this;
        } catch (const std::exception& e) {
            throw std::runtime_error("Assignment failed: " + std::string(e.what()));
        }
    }

    // Indexing operator for accessing a single index
    // Parameters:
    //   idx: Index along outermost dimension
    // Returns: A new ndarray representing the sub-array at the specified index
    // Throws: std::out_of_range for invalid dimensions, std::runtime_error for other errors
    ndarray<type> operator[](int idx) {
        try {
            if (dim <= 0) {
                throw std::out_of_range("Invalid dimension for indexing");
            }
            // Create a view of the sub-array by adjusting shape, stride, offset, and step
            return ndarray<type>(dim - 1, shape + 1,
                                 arr + (offset[0] + stride[0] * idx) * step[0],
                                 stride + 1, offset + 1, step + 1);
        } catch (const std::exception& e) {
            throw std::runtime_error("Indexing failed: " + std::string(e.what()));
        }
    }

    const ndarray<type> operator[](int idx)const {
        try {
            if (dim <= 0) {
                throw std::out_of_range("Invalid dimension for indexing");
            }
            // Create a view of the sub-array by adjusting shape, stride, offset, and step
            return ndarray<type>(dim - 1, shape + 1,
                                 arr + (offset[0] + stride[0] * idx) * step[0],
                                 stride + 1, offset + 1, step + 1);
        } catch (const std::exception& e) {
            throw std::runtime_error("Indexing failed: " + std::string(e.what()));
        }
    }

    // Indexing operator for slicing
    // Parameters:
    //   slice: Array of three integers {start, stop, step}
    // Returns: A new ndarray representing the sliced array
    // Throws: std::invalid_argument for invalid slice, std::runtime_error for other errors
    ndarray<type> operator[](const int slice[3]) {
        try {
            if (!slice) {
                throw std::invalid_argument("Slice array is null");
            }
            if (slice[2] == 0) {
                throw std::invalid_argument("Slice step cannot be zero");
            }
            // Compute new shape based on slice parameters
            int new_shape[dim];
            arr_copy(new_shape, shape, dim);
            new_shape[0] = (slice[1] - slice[0] + slice[2] - 1) / slice[2];
            // Adjust stride for slicing
            int new_stride[dim];
            arr_copy(new_stride, stride, dim);
            new_stride[0] *= slice[2];
            // Adjust offset for slice start
            int new_offset[dim];
            arr_copy(new_offset, offset, dim);
            new_offset[0] += slice[0] * stride[0];
            // Create view of sliced array
            return ndarray<type>(dim, new_shape, arr, new_stride, new_offset, step);
        } catch (const std::exception& e) {
            throw std::runtime_error("Slice indexing failed: " + std::string(e.what()));
        }
    }

    const ndarray<type> operator[](const int slice[3])const {
        try {
            if (!slice) {
                throw std::invalid_argument("Slice array is null");
            }
            if (slice[2] == 0) {
                throw std::invalid_argument("Slice step cannot be zero");
            }
            // Compute new shape based on slice parameters
            int new_shape[dim];
            arr_copy(new_shape, shape, dim);
            new_shape[0] = (slice[1] - slice[0] + slice[2] - 1) / slice[2];
            // Adjust stride for slicing
            int new_stride[dim];
            arr_copy(new_stride, stride, dim);
            new_stride[0] *= slice[2];
            // Adjust offset for slice start
            int new_offset[dim];
            arr_copy(new_offset, offset, dim);
            new_offset[0] += slice[0] * stride[0];
            // Create view of sliced array
            return ndarray<type>(dim, new_shape, arr, new_stride, new_offset, step);
        } catch (const std::exception& e) {
            throw std::runtime_error("Slice indexing failed: " + std::string(e.what()));
        }
    }

    // Gets value at specified multi-dimensional index
    // Parameters:
    //   idx: Array of indices for each dimension
    // Returns: Reference to the element at the specified index
    // Throws: std::invalid_argument for null index, std::out_of_range for invalid index
    type& get_value(const int* idx) {
        try {
            if (!idx) {
                throw std::invalid_argument("Index array is null in get_value");
            }
            // Compute linear index from multi-dimensional index
            int real_idx = 0;
            for (int i = 0; i < dim; i++) {
                if (shape[i] > 1) {
                    real_idx += (offset[i] + idx[i] * stride[i]) * step[i];
                } else {
                    real_idx += offset[i] * step[i];
                }
            }
            if (real_idx < 0 || real_idx >= size) {
                throw std::out_of_range("Index out of bounds in get_value");
            }
            return arr[real_idx];
        } catch (const std::exception& e) {
            throw std::runtime_error("Get value failed: " + std::string(e.what()));
        }
    }

    const type& get_value(const int* idx)const {
        try {
            if (!idx) {
                throw std::invalid_argument("Index array is null in get_value");
            }
            // Compute linear index from multi-dimensional index
            int real_idx = 0;
            for (int i = 0; i < dim; i++) {
                if (shape[i] > 1) {
                    real_idx += (offset[i] + idx[i] * stride[i]) * step[i];
                } else {
                    real_idx += offset[i] * step[i];
                }
            }
            if (real_idx < 0 || real_idx >= size) {
                throw std::out_of_range("Index out of bounds in get_value");
            }
            return arr[real_idx];
        } catch (const std::exception& e) {
            throw std::runtime_error("Get value failed: " + std::string(e.what()));
        }
    }

    // Gets sub-array at specified index and depth
    // Parameters:
    //   idx: Array of indices for the first 'depth' dimensions
    //   depth: Number of dimensions to index
    // Returns: A new ndarray representing the sub-array
    // Throws: std::invalid_argument for invalid parameters, std::runtime_error for other errors
    ndarray<type> get_array(const int* idx, int depth) {
        try {
            if (!idx) {
                throw std::invalid_argument("Index array is null in get_array");
            }
            if (depth < 0 || depth > dim) {
                throw std::invalid_argument("Invalid depth in get_array");
            }
            // Compute linear index for the specified depth
            int real_idx = 0;
            for (int i = 0; i < depth; i++) {
                if (shape[i] > 1) {
                    real_idx += (offset[i] + idx[i] * stride[i]) * step[i];
                } else {
                    real_idx += offset[i] * step[i];
                }
            }
            // Create view of sub-array
            return ndarray<type>(dim - depth, shape + depth, arr + real_idx,
                                 stride + depth, offset + depth, step + depth);
        } catch (const std::exception& e) {
            throw std::runtime_error("Get array failed: " + std::string(e.what()));
        }
    }

    const ndarray<type> get_array(const int* idx, int depth)const {
        try {
            if (!idx) {
                throw std::invalid_argument("Index array is null in get_array");
            }
            if (depth < 0 || depth > dim) {
                throw std::invalid_argument("Invalid depth in get_array");
            }
            // Compute linear index for the specified depth
            int real_idx = 0;
            for (int i = 0; i < depth; i++) {
                if (shape[i] > 1) {
                    real_idx += (offset[i] + idx[i] * stride[i]) * step[i];
                } else {
                    real_idx += offset[i] * step[i];
                }
            }
            // Create view of sub-array
            return ndarray<type>(dim - depth, shape + depth, arr + real_idx,
                                 stride + depth, offset + depth, step + depth);
        } catch (const std::exception& e) {
            throw std::runtime_error("Get array failed: " + std::string(e.what()));
        }
    }

    // Performs deep slicing across multiple dimensions
    // Parameters:
    //   slice: Array of slice specifications (each is {start, stop, step})
    //   depth: Number of dimensions to slice
    // Returns: A new ndarray representing the sliced array
    // Throws: std::invalid_argument for invalid parameters, std::runtime_error for other errors
    ndarray<type> deep_slice(int** slice, int depth) {
        try {
            if (!slice) {
                throw std::invalid_argument("Slice array is null in deep_slice");
            }
            if (depth <= 0 || depth > dim) {
                throw std::invalid_argument("Invalid depth in deep_slice");
            }
            // Initialize new shape, stride, and offset
            int new_shape[dim];
            arr_copy(new_shape, shape, dim);
            int new_stride[dim];
            arr_copy(new_stride, stride, dim);
            int new_offset[dim];
            arr_copy(new_offset, offset, dim);
            // Apply slice to each dimension
            for (int i = 0; i < depth; i++) {
                if (!slice[i]) {
                    throw std::invalid_argument("Slice specification is null at index " + std::to_string(i));
                }
                if (slice[i][2] == 0) {
                    throw std::invalid_argument("Slice step cannot be zero at index " + std::to_string(i));
                }
                new_shape[i] = (slice[i][1] - slice[i][0] + slice[i][2] - 1) / slice[i][2];
                new_stride[i] *= slice[i][2];
                new_offset[i] += slice[i][0] * stride[i];
            }
            // Create view of sliced array
            return ndarray<type>(dim, new_shape, arr, new_stride, new_offset, step);
        } catch (const std::exception& e) {
            throw std::runtime_error("Deep slice failed: " + std::string(e.what()));
        }
    }

    // Checks if broadcasting is possible with another array
    // Parameters:
    //   source: The other ndarray to check compatibility with
    //   depth: Number of dimensions to check
    // Returns: True if broadcasting is possible, false otherwise
    // Throws: std::runtime_error for errors during check
    bool broadcast_check(const ndarray<type>& source, int depth) {
        try {
            if (source.dim < depth || this->dim < depth) {
                return false; // Dimensions are insufficient
            }
            for (int i = 0; i < depth; i++) {
                // Broadcasting is possible if shapes match or one is 1
                if (source.shape[i] != this->shape[i] &&
                    source.shape[i] != 1 && this->shape[i] != 1) {
                    return false;
                }
            }
            return true;
        } catch (const std::exception& e) {
            throw std::runtime_error("Broadcast check failed: " + std::string(e.what()));
        }
    }

    // Computes the broadcasted shape for operations with another array
    // Parameters:
    //   source: The other ndarray
    //   depth: Number of leading dimensions to broadcast
    //   element_shape: Additional shape for element-wise operations
    //   element_dim: Number of additional dimensions
    // Returns: New shape array (must be deleted by caller)
    // Throws: std::runtime_error for allocation or computation errors
    int* broadcast_shape(const ndarray<type>& source, int depth,
                         const int* element_shape, int element_dim) {
        int* new_shape;
        try {
            // Allocate new shape array
            new_shape = new int[depth + element_dim];
            // Compute broadcasted dimensions
            for (int i = 0; i < depth; i++) {
                new_shape[i] = (this->shape[i] > source.shape[i]) ? this->shape[i] : source.shape[i];
            }
            // Copy additional element shape
            for (int i = 0; i < element_dim; i++) {
                new_shape[depth + i] = element_shape[i];
            }
            return new_shape;
        } catch (const std::exception& e) {
            delete[] new_shape;
            throw std::runtime_error("Broadcast shape calculation failed: " + std::string(e.what()));
        }
    }

    // Iterates an index array forward for traversal
    // Parameters:
    //   idx: Index array to iterate
    //   depth: Number of dimensions to iterate
    // Returns: Updated index array
    // Throws: std::invalid_argument for invalid parameters, std::runtime_error for other errors
    int* index_forward_iterate(int* idx, int depth) {
        try {
            if (!idx) {
                throw std::invalid_argument("Index array is null in index_forward_iterate");
            }
            if (depth <= 0 || depth > dim) {
                throw std::invalid_argument("Invalid depth in index_forward_iterate");
            }
            // Increment innermost dimension
            idx[depth - 1]++;
            // Handle carry-over for multi-dimensional iteration
            for (int i = depth - 1; i >= 0; i--) {
                if (idx[i] >= shape[i]) {
                    if (i > 0) {
                        idx[i - 1] += 1; // Carry to next dimension
                    }
                    idx[i] = 0; // Reset current dimension
                }
            }
            return idx;
        } catch (const std::exception& e) {
            throw std::runtime_error("Index iteration failed: " + std::string(e.what()));
        }
    }

    // Performs element-wise operation with another array
    // Parameters:
    //   source: The other ndarray
    //   fptr: Function pointer to the operation (e.g., addition, multiplication)
    // Returns: New ndarray with the result
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for dimension mismatch, std::runtime_error for other errors
    ndarray<type> element_to_element_operation(const ndarray<type>& source,
                                               type (*fptr)(const type&, const type&)) {
        try {
            if (!broadcast_check(source, dim) || dim != source.dim) {
                throw std::invalid_argument("Invalid dimensions for element operation");
            }
            // Compute broadcasted shape
            int* new_shape = broadcast_shape(source, dim, nullptr, 0);
            // Create result array
            ndarray<type> rt(dim, new_shape);
            delete[] new_shape;

#pragma omp parallel
            {
                // Each thread has its own index array
                int* local_index = new int[dim]{};
#pragma omp for
                for (int i = 0; i < rt.size; i++) {
                    try {
                        // Convert linear index to multi-dimensional
                        rt.linear_to_index(i, local_index, dim);
                        // Apply operation to corresponding elements
                        rt.get_value(local_index) = fptr(this->get_value(local_index),
                                                         source.get_value(local_index));
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] local_index;
            }
            return rt;
        } catch (const std::exception& e) {
            throw std::runtime_error("Element operation failed: " + std::string(e.what()));
        }
    }

    // Mutates this array by applying an element-wise operation with another array
    // Parameters:
    //   source: The other ndarray
    //   fptr: Function pointer to the operation
    // Returns: Reference to this array
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for dimension/shape mismatch, std::runtime_error for other errors
    ndarray<type>& element_to_element_mutator(const ndarray<type>& source,
                                              type (*fptr)(const type&,const type&)) {
        try {
            if (!broadcast_check(source, dim) || dim != source.dim) {
                throw std::invalid_argument("Invalid dimensions for element mutator");
            }
            // Verify shape compatibility
            int* new_shape = broadcast_shape(source, dim, nullptr, 0);
            for (int i = 0; i < dim; i++) {
                if (new_shape[i] != shape[i]) {
                    delete[] new_shape;
                    throw std::invalid_argument("Shape mismatch in element mutator");
                }
            }
            delete[] new_shape;

#pragma omp parallel
            {
                int* local_index = new int[dim]{};
#pragma omp for
                for (int i = 0; i < size; i++) {
                    try {
                        linear_to_index(i, local_index, dim);
                        get_value(local_index) = fptr(this->get_value(local_index),
                                                      source.get_value(local_index));
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] local_index;
            }
            return *this;
        } catch (const std::exception& e) {
            throw std::runtime_error("Element mutator failed: " + std::string(e.what()));
        }
    }

    // Applies an element-wise operation to all elements
    // Parameters:
    //   fptr: Function pointer to the operation
    // Returns: New ndarray with the result
    // Uses OpenMP for parallel execution
    // Throws: std::runtime_error for operation failures
    ndarray<type> element_operation(type (*fptr)(const type&)) {
        try {
            // Create result array with same shape
            ndarray<type> rt(dim, shape);
#pragma omp parallel
            {
                int* local_index = new int[dim]{};
#pragma omp for
                for (int i = 0; i < rt.size; i++) {
                    try {
                        linear_to_index(i, local_index, dim);
                        rt.get_value(local_index) = fptr(this->get_value(local_index));
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] local_index;
            }
            return rt;
        } catch (const std::exception& e) {
            throw std::runtime_error("Element operation failed: " + std::string(e.what()));
        }
    }

    // Mutates all elements by applying an operation
    // Parameters:
    //   fptr: Function pointer to the operation
    // Returns: Reference to this array
    // Uses OpenMP for parallel execution
    // Throws: std::runtime_error for operation failures
    ndarray<type>& element_mutator(type (*fptr)(const type&)) {
        try {
#pragma omp parallel
            {
                int* local_index = new int[dim]{};
#pragma omp for
                for (int i = 0; i < size; i++) {
                    try {
                        linear_to_index(i, local_index, dim);
                        get_value(local_index) = fptr(this->get_value(local_index));
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] local_index;
            }
            return *this;
        } catch (const std::exception& e) {
            throw std::runtime_error("Element mutator failed: " + std::string(e.what()));
        }
    }

    // Performs array-to-array operation on sub-arrays
    // Parameters:
    //   source: The other ndarray
    //   fptr: Function pointer to the operation
    //   depth: Number of leading dimensions to broadcast
    //   element_shape: Shape of the resulting sub-arrays
    //   element_dim: Number of dimensions in sub-arrays
    // Returns: New ndarray with the result
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for dimension mismatch, std::runtime_error for other errors
    ndarray<type> array_to_array_operation(const ndarray<type>& source,
                                           ndarray<type> (*fptr)(const ndarray<type>& , const ndarray<type>& ),
                                           int depth, int* element_shape, int element_dim) {
        try {
            if (!broadcast_check(source, depth)) {
                throw std::invalid_argument("Invalid dimensions for array operation");
            }
            // Compute broadcasted shape
            int* new_shape = broadcast_shape(source, depth, element_shape, element_dim);
            // Create result array
            ndarray<type> rt(depth + element_dim, new_shape);
            delete[] new_shape;

#pragma omp parallel
            {
                int* local_index = new int[depth]{};
#pragma omp for
                for (int i = 0; i < rt.mod[depth]; i++) {
                    try {
                        // Compute multi-dimensional index for current iteration
                        int remaining = i;
                        for (int j = depth - 1; j >= 0; j--) {
                            local_index[j] = remaining % rt.shape[j];
                            remaining /= rt.shape[j];
                        }
                        // Apply operation to sub-arrays
                        rt.get_array(local_index, depth) = fptr(this->get_array(local_index, depth),
                                                                source.get_array(local_index, depth));
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] local_index;
            }
            return rt;
        } catch (const std::exception& e) {
            throw std::runtime_error("Array operation failed: " + std::string(e.what()));
        }
    }

    // Mutates this array by applying an array-to-array operation
    // Parameters:
    //   source: The other ndarray
    //   fptr: Function pointer to the operation
    //   depth: Number of leading dimensions to broadcast
    // Returns: Reference to this array
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for dimension/shape mismatch, std::runtime_error for other errors
    ndarray<type>& array_to_array_mutator(const ndarray<type>& source,
                                          ndarray<type> (*fptr)(const ndarray<type>&,const ndarray<type>&),
                                          int depth) {
        try {
            if (!broadcast_check(source, depth)) {
                throw std::invalid_argument("Invalid dimensions for array mutator");
            }
            // Verify shape compatibility
            int* new_shape = broadcast_shape(source, depth, nullptr, 0);
            for (int i = 0; i < depth; i++) {
                if (new_shape[i] != shape[i]) {
                    delete[] new_shape;
                    throw std::invalid_argument("Shape mismatch in array mutator");
                }
            }
            delete[] new_shape;

#pragma omp parallel
            {
                int* local_index = new int[depth]{};
#pragma omp for
                for (int i = 0; i < mod[depth]; i++) {
                    try {
                        int remaining = i;
                        for (int j = depth - 1; j >= 0; j--) {
                            local_index[j] = remaining % shape[j];
                            remaining /= shape[j];
                        }
                        get_array(local_index, depth) = fptr(this->get_array(local_index, depth),
                                                             source.get_array(local_index, depth));
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] local_index;
            }
            return *this;
        } catch (const std::exception& e) {
            throw std::runtime_error("Array mutator failed: " + std::string(e.what()));
        }
    }

    // Applies an operation to array sections
    // Parameters:
    //   fptr: Function pointer to the operation
    //   depth: Number of leading dimensions to process
    //   element_shape: Shape of the resulting sub-arrays
    //   element_dim: Number of dimensions in sub-arrays
    // Returns: New ndarray with the result
    // Uses OpenMP for parallel execution
    // Throws: std::runtime_error for operation failures
    ndarray<type> array_operation(ndarray<type> (*fptr)(const ndarray<type>&),
                                  int depth, int* element_shape, int element_dim) {
        try {
            // Create new shape combining input dimensions and element shape
            int new_shape[depth + element_dim];
            for (int i = 0; i < depth; i++) {
                new_shape[i] = shape[i];
            }
            for (int i = 0; i < element_dim; i++) {
                new_shape[depth + i] = element_shape[i];
            }
            // Create result array
            ndarray<type> rt(depth + element_dim, new_shape);

#pragma omp parallel
            {
                int* local_index = new int[depth]{};
#pragma omp for
                for (int i = 0; i < rt.mod[depth]; i++) {
                    try {
                        int remaining = i;
                        for (int j = depth - 1; j >= 0; j--) {
                            local_index[j] = remaining % rt.shape[j];
                            remaining /= rt.shape[j];
                        }
                        rt.get_array(local_index, depth) = fptr(this->get_array(local_index, depth));
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] local_index;
            }
            return rt;
        } catch (const std::exception& e) {
            throw std::runtime_error("Array operation failed: " + std::string(e.what()));
        }
    }

    // Mutates array sections by applying an operation
    // Parameters:
    //   fptr: Function pointer to the operation
    //   depth: Number of leading dimensions to process
    // Returns: Reference to this array
    // Uses OpenMP for parallel execution
    // Throws: std::runtime_error for operation failures
    ndarray<type>& array_mutator(ndarray<type> (*fptr)(const ndarray<type>&), int depth) {
        try {
#pragma omp parallel
            {
                int* local_index = new int[depth]{};
#pragma omp for
                for (int i = 0; i < mod[depth]; i++) {
                    try {
                        int remaining = i;
                        for (int j = depth - 1; j >= 0; j--) {
                            local_index[j] = remaining % shape[j];
                            remaining /= shape[j];
                        }
                        get_array(local_index, depth) = fptr(this->get_array(local_index, depth));
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] local_index;
            }
            return *this;
        } catch (const std::exception& e) {
            throw std::runtime_error("Array mutator failed: " + std::string(e.what()));
        }
    }

    // Arithmetic operator: Element-wise addition
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the sum
    ndarray<type> operator+(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return a + b; });
    }

    // Arithmetic operator: In-place element-wise addition
    // Parameters:
    //   source: The other ndarray
    // Returns: Reference to this array
    ndarray<type>& operator+=(const ndarray<type>& source) {
        return element_to_element_mutator(source, [](const type& a, const type& b) { return a + b; });
    }

    // Arithmetic operator: Element-wise subtraction
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the difference
    ndarray<type> operator-(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return a - b; });
    }

    // Arithmetic operator: In-place element-wise subtraction
    // Parameters:
    //   source: The other ndarray
    // Returns: Reference to this array
    ndarray<type>& operator-=(const ndarray<type>& source) {
        return element_to_element_mutator(source, [](const type& a, const type& b) { return a - b; });
    }

    // Arithmetic operator: Element-wise multiplication
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the product
    ndarray<type> operator*(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return a * b; });
    }

    // Arithmetic operator: In-place element-wise multiplication
    // Parameters:
    //   source: The other ndarray
    // Returns: Reference to this array
    ndarray<type>& operator*=(const ndarray<type>& source) {
        return element_to_element_mutator(source, [](const type& a, const type& b) { return a * b; });
    }

    // Arithmetic operator: Element-wise division
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the quotient
    ndarray<type> operator/(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return a / b; });
    }

    // Arithmetic operator: In-place element-wise division
    // Parameters:
    //   source: The other ndarray
    // Returns: Reference to this array
    ndarray<type>& operator/=(const ndarray<type>& source) {
        return element_to_element_mutator(source, [](const type& a, const type& b) { return a / b; });
    }

    // Arithmetic operator: Element-wise modulo
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the remainder
    ndarray<type> operator%(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return a % b; });
    }

    // Arithmetic operator: In-place element-wise modulo
    // Parameters:
    //   source: The other ndarray
    // Returns: Reference to this array
    ndarray<type>& operator%=(const ndarray<type>& source) {
        return element_to_element_mutator(source, [](const type& a, const type& b) { return a % b; });
    }

    // Logical operator: Element-wise OR
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the logical OR
    ndarray<type> operator||(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return a || b; });
    }

    // Logical operator: Element-wise AND
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the logical AND
    ndarray<type> operator&&(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return a && b; });
    }

    ndarray<type> operator==(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return a == b; });
    }

    // Computes element-wise maximum
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the maximum values
    ndarray<type> max(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return std::max(a, b); });
    }

    // Computes element-wise minimum
    // Parameters:
    //   source: The other ndarray
    // Returns: New ndarray with the minimum values
    ndarray<type> min(const ndarray<type>& source) {
        return element_to_element_operation(source, [](const type& a, const type& b) { return std::min(a, b); });
    }

    // Logical operator: Element-wise NOT
    // Returns: New ndarray with the logical NOT
    ndarray<type> operator!() {
        return element_operation([](const type& a) { return !a; });
    }

    // Arithmetic operator: Element-wise increment
    // Returns: Reference to this array
    ndarray<type>& operator++() {
        return element_mutator([](const type& a) { return ++a; });
    }

    // Arithmetic operator: Element-wise decrement
    // Returns: Reference to this array
    ndarray<type>& operator--() {
        return element_mutator([](const type& a) { return --a; });
    }

    // Arithmetic operator: Add scalar to all elements
    // Parameters:
    //   i: Scalar value
    // Returns: New ndarray with the result
    ndarray<type> operator+(type i) {
        return element_operation([=](const type& a) { return a + i; });
    }

    // Arithmetic operator: Subtract scalar from all elements
    // Parameters:
    //   i: Scalar value
    // Returns: New ndarray with the result
    ndarray<type> operator-(type i) {
        return element_operation([=](const type& a) { return a - i; });
    }

    // Arithmetic operator: Multiply all elements by scalar
    // Parameters:
    //   i: Scalar value
    // Returns: New ndarray with the result
    ndarray<type> operator*(type i) {
        return element_operation([=](const type& a) { return a * i; });
    }

    // Arithmetic operator: Divide all elements by scalar
    // Parameters:
    //   i: Scalar value
    // Returns: New ndarray with the result
    ndarray<type> operator/(type i) {
        return element_operation([=](const type& a) { return a / i; });
    }

    // Arithmetic operator: Modulo all elements by scalar
    // Parameters:
    //   i: Scalar value
    // Returns: New ndarray with the result
    ndarray<type> operator%(type i) {
        return element_operation([=](const type& a) { return a % i; });
    }

    // Logical operator: OR all elements with scalar
    // Parameters:
    //   i: Scalar value
    // Returns: New ndarray with the result
    ndarray<type> operator||(type i) {
        return element_operation([=](const type& a) { return a || i; });
    }

    // Logical operator: AND all elements with scalar
    // Parameters:
    //   i: Scalar value
    // Returns: New ndarray with the result
    ndarray<type> operator&&(type i) {
        return element_operation([=](const type& a) { return a && i; });
    }

    // Arithmetic operator: In-place add scalar to all elements
    // Parameters:
    //   i: Scalar value
    // Returns: Reference to this array
    ndarray<type>& operator+=(type i) {
        return element_mutator([=](const type& a) { return a + i; });
    }

    // Arithmetic operator: In-place subtract scalar from all elements
    // Parameters:
    //   i: Scalar value
    // Returns: Reference to this array
    ndarray<type>& operator-=(type i) {
        return element_mutator([=](const type& a) { return a - i; });
    }

    // Arithmetic operator: In-place multiply all elements by scalar
    // Parameters:
    //   i: Scalar value
    // Returns: Reference to this array
    ndarray<type>& operator*=(type i) {
        return element_mutator([=](const type& a) { return a * i; });
    }

    // Arithmetic operator: In-place divide all elements by scalar
    // Parameters:
    //   i: Scalar value
    // Returns: Reference to this array
    ndarray<type>& operator/=(type i) {
        return element_mutator([=](const type& a) { return a / i; });
    }

    // Arithmetic operator: In-place modulo all elements by scalar
    // Parameters:
    //   i: Scalar value
    // Returns: Reference to this array
    ndarray<type>& operator%=(type i) {
        return element_mutator([=](const type& a) { return a % i; });
    }

    // Performs matrix multiplication on 2D arrays
    // Parameters:
    //   a: First input matrix
    //   b: Second input matrix
    // Returns: New ndarray with the matrix product
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for dimension mismatch, std::runtime_error for other errors
    static ndarray<type> elementary_matmul(const ndarray<type>& a, const ndarray<type>& b) {
        try {
            // Validate matrix dimensions
            if (a.dim != 2 || b.dim != 2 || a.shape[1] != b.shape[0]) {
                throw std::invalid_argument("Invalid dimensions for matrix multiplication");
            }
            // Create result array with shape [a.rows, b.cols]
            int element_shape[2] = {a.shape[0], b.shape[1]};
            ndarray<type> rt(2, element_shape);

#pragma omp parallel for collapse(2)
            for (int i = 0; i < a.shape[0]; i++) {
                for (int j = 0; j < b.shape[1]; j++) {
                    try {
                        int rt_index[2] = {i, j};
                        type sum = 0;
                        // Compute dot product
                        for (int k = 0; k < a.shape[1]; k++) {
                            int this_index[2] = {i, k};
                            int source_index[2] = {k, j};
                            sum += a.get_value(this_index) * b.get_value(source_index);
                        }
                        rt.get_value(rt_index) = sum;
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
            }
            return rt;
        } catch (const std::exception& e) {
            throw std::runtime_error("Matrix multiplication failed: " + std::string(e.what()));
        }
    }

    // Performs batched matrix multiplication
    // Parameters:
    //   source: The other ndarray (batch of matrices)
    // Returns: New ndarray with the matrix products
    // Throws: std::invalid_argument for dimension mismatch, std::runtime_error for other errors
    ndarray<type> matmul(const ndarray<type>& source) {
        try {
            if (dim < 2 || source.dim < 2) {
                throw std::invalid_argument("Invalid dimensions for batched matrix multiplication");
            }
            // Define shape for matrix multiplication result
            int element_shape[2] = {this->shape[dim - 2], source.shape[source.dim - 1]};
            // Apply batched operation
            return array_to_array_operation(source, elementary_matmul, dim - 2, element_shape, 2);
        } catch (const std::exception& e) {
            throw std::runtime_error("Batched matrix multiplication failed: " + std::string(e.what()));
        }
    }

    // Computes sum along specified axis
    // Parameters:
    //   axis: The axis to sum along
    // Returns: New ndarray with the sums
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for invalid axis, std::runtime_error for other errors
    ndarray<type> axis_sum(int axis) {
        try {
            if (axis < 0 || axis >= dim) {
                throw std::invalid_argument("Invalid axis for sum");
            }
            // Define shape for result (1 along summed axis)
            int element_shape[dim - axis];
            element_shape[0] = 1;
            for (int i = 1; i < dim - axis; i++) {
                element_shape[i] = shape[axis + i];
            }
            return array_operation([](const ndarray<type>& a) {
                // Create result array with reduced shape
                int element_shape[a.dim];
                element_shape[0] = 1;
                for (int i = 1; i < a.dim; i++) {
                    element_shape[i] = a.shape[i];
                }
                ndarray<type> rt(a.dim, element_shape);

#pragma omp parallel
                {
                    // Each thread has its own local result
                    ndarray<type> local_rt(a.dim, element_shape);
#pragma omp for nowait
                    for (int i = 0; i < a.shape[0]; i++) {
                        try {
                            local_rt[0] = (local_rt[0] + a[i]);
                        } catch (const std::exception& e) {
                            throw;
                        }
                    }
#pragma omp critical
                    {
                        rt[0] = (rt[0] + local_rt[0]); // Combine thread results
                    }
                }
                return rt;
            }, axis, element_shape, dim - axis);
        } catch (const std::exception& e) {
            throw std::runtime_error("Axis sum failed: " + std::string(e.what()));
        }
    }

    // Computes maximum along specified axis
    // Parameters:
    //   axis: The axis to find maximum along
    // Returns: New ndarray with the maximum values
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for invalid axis, std::runtime_error for other errors
    ndarray<type> axis_max(int axis) {
        try {
            if (axis < 0 || axis >= dim) {
                throw std::invalid_argument("Invalid axis for max");
            }
            // Define shape for result
            int element_shape[dim - axis];
            element_shape[0] = 1;
            for (int i = 1; i < dim - axis; i++) {
                element_shape[i] = shape[axis + i];
            }
            return array_operation([](const ndarray<type>& a) {
                int element_shape[a.dim];
                element_shape[0] = 1;
                for (int i = 1; i < a.dim; i++) {
                    element_shape[i] = a.shape[i];
                }
                ndarray<type> rt(a.dim, element_shape);
                rt[0] = a[0]; // Initialize with first element

#pragma omp parallel
                {
                    ndarray<type> local_max(a.dim, element_shape);
                    local_max[0] = a[0];

#pragma omp for nowait
                    for (int i = 1; i < a.shape[0]; i++) {
                        try {
                            local_max[0] = local_max[0].max(a[i]);
                        } catch (const std::exception& e) {
                            throw;
                        }
                    }
#pragma omp critical
                    {
                        rt[0] = rt[0].max(local_max[0]); // Combine thread results
                    }
                }
                return rt;
            }, axis, element_shape, dim - axis);
        } catch (const std::exception& e) {
            throw std::runtime_error("Axis max failed: " + std::string(e.what()));
        }
    }

    // Computes minimum along specified axis
    // Parameters:
    //   axis: The axis to find minimum along
    // Returns: New ndarray with the minimum values
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for invalid axis, std::runtime_error for other errors
    ndarray<type> axis_min(int axis) {
        try {
            if (axis < 0 || axis >= dim) {
                throw std::invalid_argument("Invalid axis for min");
            }
            // Define shape for result
            int element_shape[dim - axis];
            element_shape[0] = 1;
            for (int i = 1; i < dim - axis; i++) {
                element_shape[i] = shape[axis + i];
            }
            return array_operation([](const ndarray<type>& a) {
                int element_shape[a.dim];
                element_shape[0] = 1;
                for (int i = 1; i < a.dim; i++) {
                    element_shape[i] = a.shape[i];
                }
                ndarray<type> rt(a.dim, element_shape);
                rt[0] = a[0]; // Initialize with first element

#pragma omp parallel
                {
                    ndarray<type> local_min(a.dim, element_shape);
                    local_min[0] = a[0];

#pragma omp for nowait
                    for (int i = 1; i < a.shape[0]; i++) {
                        try {
                            local_min[0] = local_min[0].min(a[i]);
                        } catch (const std::exception& e) {
                            throw;
                        }
                    }
#pragma omp critical
                    {
                        rt[0] = rt[0].min(local_min[0]); // Combine thread results
                    }
                }
                return rt;
            }, axis, element_shape, dim - axis);
        } catch (const std::exception& e) {
            throw std::runtime_error("Axis min failed: " + std::string(e.what()));
        }
    }

    // Transposes array according to specified axis order
    // Parameters:
    //   axis_order: Array specifying the new order of axes
    // Returns: New ndarray with transposed axes
    // Uses OpenMP for parallel execution
    // Throws: std::invalid_argument for invalid axis order, std::runtime_error for other errors
    ndarray<type> get_transpose(int* axis_order) {
        try {
            if (!axis_order) {
                throw std::invalid_argument("Axis order array is null in get_transpose");
            }
            // Compute new shape based on axis order
            int new_shape[dim];
            for (int i = 0; i < dim; i++) {
                if (axis_order[i] < 0 || axis_order[i] >= dim) {
                    throw std::invalid_argument("Invalid axis in transpose at index " + std::to_string(i));
                }
                new_shape[i] = shape[axis_order[i]];
            }
            // Create result array
            ndarray<type> rt(dim, new_shape);

#pragma omp parallel
            {
                int* rt_index = new int[dim]{};
                int* index = new int[dim]{};
#pragma omp for
                for (int i = 0; i < rt.size; i++) {
                    try {
                        linear_to_index(i, index, dim);
                        // Map result index to source index based on axis order
                        for (int j = 0; j < dim; j++) {
                            rt_index[j] = index[axis_order[j]];
                        }
                        rt.get_value(rt_index) = this->get_value(index);
                    } catch (const std::exception& e) {
                        throw; // Propagate to outer catch
                    }
                }
                delete[] rt_index;
                delete[] index;
            }
            return rt;
        } catch (const std::exception& e) {
            throw std::runtime_error("Transpose failed: " + std::string(e.what()));
        }
    }

    ndarray<type>& expand_dims(int axis){
        try {
            if (axis < 0 || axis >= dim) {
                throw std::invalid_argument("Invalid axis for expand_dims");
            }
            int* new_shape;
            int* new_mod;
            int* new_step;
            int* new_stride;
            int* new_offset;
            try {
                new_shape = new int[dim + 1];
                new_mod = new int[dim + 1];
                new_step= new int[dim + 1];
                new_stride = new int[dim + 1];
                new_offset = new int[dim + 1];

                for (int i = 0; i < dim + 1; i++) {
                    if (i < axis) {
                        new_shape[i] = shape[i];
                        new_mod[i] = mod[i];
                        new_step[i] = step[i];
                        new_stride[i] = stride[i];
                        new_offset[i] = offset[i];
                    }
                    if (i > axis) {
                        new_shape[i] = shape[i-1];
                        new_mod[i] = mod[i-1];
                        new_step[i] = step[i-1];
                        new_stride[i] = stride[i-1];
                        new_offset[i] = offset[i-1];
                    }
                }

                new_shape[axis] = 1;
                new_mod[axis] = new_mod[axis - 1];
                new_step[axis] = new_step[axis + 1];
                new_stride[axis] = 1;
                new_offset[axis] = 0;
                dim += 1;

                delete [] shape;
                delete [] mod;
                delete [] step;
                delete [] stride;
                delete [] offset;

                shape = new_shape;
                mod = new_mod;
                step = new_step;
                stride = new_stride;
                offset = new_offset;

            }catch (const std::bad_alloc& e){
                delete [] new_shape;
                delete [] new_mod;
                delete [] new_step;
                delete [] new_stride;
                delete [] new_offset;
                throw std::runtime_error("fail to allocate memory for expand_dims");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("expand_dims failed: " + std::string(e.what()));
        }
        return *this;
    }

    ndarray<type>& squeeze(){
        int* new_shape;
        int* new_mod;
        int* new_step;
        int* new_stride;
        int* new_offset;
        int count = 0;
        for(int i = 0;i < dim;i++)
            if(shape[i] == 1)
                count++;
        try{
            new_shape = new int[dim - count];
            new_mod = new int[dim - count];
            new_step= new int[dim - count];
            new_stride = new int[dim - count];
            new_offset = new int[dim - count];
            int iter = 0;

            for(int i = 0;i < dim;i++){
                if(shape[i] != 1){
                    new_shape[iter] = shape[i];
                    new_mod[iter] = mod[i];
                    new_step[iter] = step[i];
                    new_stride[iter] = stride[i];
                    new_offset[iter] = offset[i];
                    iter++;
                }
            }

            dim -= count;

            delete [] shape;
            delete [] mod;
            delete [] step;
            delete [] stride;
            delete [] offset;

            shape = new_shape;
            mod = new_mod;
            step = new_step;
            stride = new_stride;
            offset = new_offset;

        }catch (const std::bad_alloc& e){
            delete [] new_shape;
            delete [] new_mod;
            delete [] new_step;
            delete [] new_stride;
            delete [] new_offset;
            throw std::runtime_error("fail to allocate memory for squeeze");
        }
        return *this;
    }

    ndarray<type> reshape(int new_dim,int* new_shape){
        try {
            if(new_dim <= 0 || !new_shape)
                throw std::invalid_argument("Invalid new_dim or new_shape = nullptr");
            int new_size = 1;
            for(int i = 0;i < new_dim;i++)
                new_size *= new_shape[i];
            if(new_size != size)
                throw std::invalid_argument("size not match");
            ndarray<type> rt = ndarray<type>(new_dim, new_shape);
#pragma omp parallel
            {
                int *rt_index = new int[new_dim];
                int *index = new int[dim];
#pragma omp for
                for (int i = 0; i < size; i++) {
                    rt.linear_to_index(i, rt_index, rt.dim);
                    linear_to_index(i,index,dim);
                    rt.get_value(rt_index) = get_value(index);
                }
                delete[] rt_index;
                delete[] index;
            }
            return rt;
        }catch(const std::exception &e){
            throw std::runtime_error("reshape failed: " + std::string(e.what()));
        }
    }

    // Gets the number of OpenMP threads
    // Returns: Number of threads currently in use
    // Throws: std::runtime_error for OpenMP errors
    static int get_num_threads() {
        try {
            int num_threads = 1;
#pragma omp parallel
            {
#pragma omp single
                num_threads = omp_get_num_threads(); // Get thread count in parallel region
            }
            return num_threads;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to get number of threads: " + std::string(e.what()));
        }
    }

    // Sets the number of OpenMP threads
    // Parameters:
    //   n: Number of threads to set
    // Throws: std::invalid_argument for invalid number, std::runtime_error for other errors
    static void set_num_threads(int n) {
        try {
            if (n <= 0) {
                throw std::invalid_argument("Number of threads must be positive");
            }
            omp_set_num_threads(n);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to set number of threads: " + std::string(e.what()));
        }
    }

    // Prints array shape and contents
    // Outputs to std::cout
    // Throws: std::runtime_error for printing or indexing errors
    void print() {
        try {
            // Print shape
            std::cout << "shape = [";
            for (int i = 0; i < dim; i++) {
                std::cout << shape[i] << ",";
            }
            std::cout << "]" << std::endl;

            // Print each element with its index
            int* index = new int[dim]{};
            for (int i = 0; i < size; i++) {
                std::cout << "[";
                for (int j = 0; j < dim; j++) {
                    std::cout << index[j] << ",";
                }
                std::cout << "] = " << get_value(index) << std::endl;
                index_forward_iterate(index, dim);
            }
            delete[] index;
        } catch (const std::exception& e) {
            throw std::runtime_error("Print failed: " + std::string(e.what()));
        }
    }
};