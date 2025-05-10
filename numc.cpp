#include <iostream>
using namespace std;

template <class dtype>
class numc {
private:
    int dim;          // Number of dimensions
    int* shape;       // Array storing the size of each dimension
    dtype* arr;       // Flattened array containing all elements
    int size;         // Total number of elements (product of shape values)
    bool temp;        // Flag indicating if the array is temporary (not owned)

public:
    // Constructor: Initializes a numc object with given dimensions and shape
    numc(int dim, const int* shape, dtype* source = nullptr): dim(dim), size(1) {
        // Allocate memory for shape if dimensions exist
        if (dim)
            this->shape = new int[dim];
        else
            this->shape = nullptr;
        // Copy shape values and compute total size
        for (int i = 0; i < this->dim; i++) {
            this->shape[i] = shape[i];
            this->size *= shape[i];
        }
        // Use provided source array if available
        if (source) {
            arr = source;
            temp = true;  // Mark as temporary to avoid deletion
        } else {
            // Otherwise, allocate new array and initialize to zero
            arr = new dtype[this->size];
            for (int i = 0; i < size; i++)
                arr[i] = 0;
            temp = false;
        }
    }

    // Copy Constructor: Creates a deep copy of another numc object
    numc(const numc & source): numc(source.dim, source.shape) {
        // Copy all elements from the source array
        for (int i = 0; i < this->size; i++)
            arr[i] = source.arr[i];
    }

    // Destructor: Frees allocated memory
    ~numc() {
        delete[] shape;  // Free shape array
        if (!temp)       // Free data array only if not temporary
            delete[] arr;
    }

    // Assignment Operator: Assigns values from another numc object
    numc & operator= (const numc & source) {
        if (temp) {
            // For temporary objects, ensure shapes match before copying
            if (dim != source.dim)
                throw invalid_argument("shape not match");
            for (int i = 0; i < dim; i++)
                if (shape[i] != source.shape[i])
                    throw invalid_argument("shape not match");
            // Copy elements directly
            for (int i = 0; i < size; i++)
                arr[i] = source.arr[i];
        } else {
            // For non-temporary objects, reallocate and copy everything
            dim = source.dim;
            delete[] shape;
            shape = new int[dim];
            for (int i = 0; i < dim; i++)
                shape[i] = source.shape[i];
            size = source.size;
            delete[] arr;
            arr = new dtype[size];
            for (int i = 0; i < size; i++)
                arr[i] = source.arr[i];
        }
        return *this;
    }

    // Indexing Operator: Accesses a sub-array along the first dimension
    numc operator[](int idx) {
        // Return a new numc object representing the sub-array
        return numc<dtype>(dim - 1, shape + 1, &(arr[idx * (size / shape[0])]));
    }

    // Slice: Extracts a sliced portion of the array along the first dimension
    numc slice(int from, int to, int step) {
        // Compute new shape for the sliced array
        int* new_shape = new int[dim];
        new_shape[0] = (to - from) / step;  // Size of the first dimension after slicing
        for (int i = 1; i < dim; i++)
            new_shape[i] = shape[i];
        // Create a new numc object with the new shape
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        delete[] new_shape;  // Clean up temporary shape array
        // Populate the sliced array using indexing
        for (int i = 0; i < rt.shape[0]; i++)
            rt[i] = (*this)[from + i * step];
        return rt;
    }

    // Get Element: Returns a reference to an element at the specified index
    dtype & get(int idx = 0) {
        return arr[idx];  // Direct access to the flattened array
    }

    // Broadcast: Maps an index to the corresponding element in a broadcasted array
    const dtype* broadcast(const numc& source, const int& idx) {
        int step = 1;        // Step size for this array
        int source_step = 1; // Step size for source array
        int target_idx = 0;  // Computed index in source array
        // Iterate over dimensions from right to left
        for (int i = dim - 1; i >= 0; i--) {
            if (source.shape[i] == shape[i])
                target_idx += source_step * ((idx / step) % shape[i]);
            step *= shape[i];
            source_step *= source.shape[i];
        }
        return &source.arr[target_idx];
    }

    // Broadcast Check: Verifies if broadcasting is possible with another array
    bool broadcast_check(const numc & source) {
        if (dim != source.dim)  // Dimensions must match
            return false;
        // Check compatibility of shapes (must be equal or one of them is 1)
        for (int i = 0; i < dim; i++)
            if ((shape[i] != source.shape[i]) && (source.shape[i] != 1) && (shape[i] != 1))
                return false;
        return true;
    }

    // Broadcast Shape: Computes the resulting shape after broadcasting
    int* broadcast_shape(const numc & source) {
        int* rt = new int[dim];  // New shape array
        // Take the maximum size for each dimension
        for (int i = 0; i < dim; i++)
            rt[i] = (shape[i] >= source.shape[i]) ? shape[i] : source.shape[i];
        return rt;
    }

    // Global Operation: Applies a function across all elements
    dtype global_op(dtype (*fptr)(dtype, dtype), bool init_to_first_term = false) {
        dtype rt = 0;  // Result accumulator
        if (init_to_first_term)
            rt = arr[0];  // Initialize with first element if specified
        // Apply the function to all elements
        for (int i = 0; i < size; i++)
            rt = fptr(rt, arr[i]);
        return rt;
    }

    // Global Operation with Index: Applies an indexed function across all elements
    dtype global_op(dtype (*fptr)(dtype, dtype, int), bool init_to_first_term = false) {
        dtype rt = 0;
        if (init_to_first_term)
            rt = arr[0];
        for (int i = 0; i < size; i++)
            rt = fptr(rt, arr[i], i);  // Pass index to the function
        return rt;
    }

    // Pointwise Operation: Applies a function to each element
    numc p_op(dtype (*fptr)(dtype)) {
        numc<dtype> rt = numc<dtype>(*this);  // Create a copy
        for (int i = 0; i < size; i++)
            rt.arr[i] = fptr(arr[i]);  // Apply function to each element
        return rt;
    }

    // Pointwise Operation with Index: Applies an indexed function to each element
    numc p_op(dtype (*fptr)(dtype, int)) {
        numc<dtype> rt = numc<dtype>(*this);
        for (int i = 0; i < size; i++)
            rt.arr[i] = fptr(arr[i], i);
        return rt;
    }

    // Pointwise Broadcast Operation: Applies a function with broadcasting
    numc ptp_broadcast_op(const numc & source, dtype (*fptr)(dtype, dtype)) {
        if (!broadcast_check(source))
            throw invalid_argument("ptp error : shape not match");
        int* new_shape = broadcast_shape(source);  // Compute broadcasted shape
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        delete[] new_shape;
        // Apply function to corresponding elements after broadcasting
        for (int i = 0; i < rt.size; i++) {
            rt.arr[i] = fptr(*rt.broadcast(*this, i), *rt.broadcast(source, i));
        }
        return rt;
    }

    // Pointwise Broadcast Operation with Index: Applies an indexed function with broadcasting
    numc ptp_broadcast_op(const numc & source, dtype (*fptr)(dtype, dtype, int)) {
        if (!broadcast_check(source))
            throw invalid_argument("ptp error : shape not match");
        int* new_shape = broadcast_shape(source);
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        delete[] new_shape;
        for (int i = 0; i < rt.size; i++) {
            rt.arr[i] = fptr(*rt.broadcast(*this, i), *rt.broadcast(source, i), i);
        }
        return rt;
    }

    // Axis Operation: Applies a function along a specified axis
    numc axis_op(int axis, dtype (*fptr)(dtype, dtype), bool init_to_first_term = false) {
        int step = 1;  // Size of the block after the axis
        int mod = 1;   // Size of the block before the axis
        int* new_shape = new int[dim];
        // Set new shape, reducing the specified axis to 1
        for (int i = 0; i < dim; i++) {
            if (i == axis)
                new_shape[i] = 1;
            else
                new_shape[i] = shape[i];
            if (i > axis)
                step *= shape[i];
            if (i < axis)
                mod *= shape[i];
        }
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        // Apply the operation along the axis
        for (int i = 0; i < mod; i++)
            for (int j = 0; j < step; j++) {
                if (init_to_first_term)
                    rt.arr[i * step + j] = arr[i * step * shape[axis]];
                for (int k = 0; k < shape[axis]; k++)
                    rt.arr[i * step + j] = fptr(rt.arr[i * step + j], arr[i * step * shape[axis] + k * step + j]);
            }
        return rt;
    }

    // Axis Operation with Index: Applies an indexed function along a specified axis
    numc axis_op(int axis, dtype (*fptr)(dtype, dtype, int), bool init_to_first_term = false) {
        int step = 1;
        int mod = 1;
        // Compute step size for dimensions after the axis
        for (int i = dim - 1; i > axis; i--)
            step *= shape[i];
        int* new_shape = new int[dim];
        for (int i = 0; i < dim; i++) {
            if (i == axis)
                new_shape[i] = 1;
            else
                new_shape[i] = shape[i];
            if (i > axis)
                step *= shape[i];
            if (i < axis)
                mod *= shape[i];
        }
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        for (int i = 0; i < mod; i++)
            for (int j = 0; j < step; j++) {
                if (init_to_first_term)
                    rt.arr[i * step + j] = arr[i * step * shape[axis]];
                for (int k = 0; k < shape[axis]; k++)
                    rt.arr[i * step + j] = fptr(rt.arr[i * step + j], arr[i * step * shape[axis] + k * step + j], i * step * shape[axis] + k * step + j);
            }
        dim = dim / shape[axis];  // Update dimensions (though this might be a bug; should be in rt)
        delete[] shape;
        shape = new_shape;
        return rt;
    }

    // Axis Expand: Expands an axis to a target size using a function
    numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype)) {
        if (shape[axis] != 1)
            throw invalid_argument("ptp error : shape not match");
        int step = 1;
        int mod = 1;
        for (int i = dim - 1; i > axis; i--)
            step *= shape[i];
        int* new_shape = new int[dim];
        for (int i = 0; i < dim; i++) {
            if (i == axis)
                new_shape[i] = axis_target;
            else
                new_shape[i] = shape[i];
            if (i > axis)
                step *= shape[i];
            if (i < axis)
                mod *= shape[i];
        }
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        // Expand the axis by applying the function
        for (int i = 0; i < mod; i++)
            for (int j = 0; j < step; j++)
                for (int k = 0; k < axis_target; k++)
                    rt.arr[i * step * axis_target + k * step + j] = fptr(arr[i * step + j]);
        return rt;
    }

    // Axis Expand with Index: Expands an axis with an indexed function
    numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype, int)) {
        if (shape[axis] != 1)
            throw invalid_argument("ptp error : shape not match");
        int step = 1;
        int mod = 1;
        for (int i = dim - 1; i > axis; i--)
            step *= shape[i];
        int* new_shape = new int[dim];
        for (int i = 0; i < dim; i++) {
            if (i == axis)
                new_shape[i] = axis_target;
            else
                new_shape[i] = shape[i];
            if (i > axis)
                step *= shape[i];
            if (i < axis)
                mod *= shape[i];
        }
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        for (int i = 0; i < mod; i++)
            for (int j = 0; j < step; j++)
                for (int k = 0; k < axis_target; k++)
                    rt.arr[i * step * axis_target + k * step + j] = fptr(arr[i * step + j], k);
        return rt;
    }

    // Addition Operator: Element-wise addition with broadcasting
    numc operator+ (const numc & source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b){ return a + b; });
    }

    // Subtraction Operator: Element-wise subtraction with broadcasting
    numc operator- (const numc & source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b){ return a - b; });
    }

    // Multiplication Operator: Element-wise multiplication with broadcasting
    numc operator* (const numc & source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b){ return a * b; });
    }

    // Division Operator: Element-wise division with broadcasting
    numc operator/ (const numc & source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b){ return a / b; });
    }

    // Matrix Multiplication: Performs matrix multiplication with broadcasting
    numc matmul(const numc & source) {
        if (dim != source.dim || dim < 2)
            throw invalid_argument("ptp error : shape not match");
        for (int i = 0; i < dim - 2; i++)
            if ((shape[i] != source.shape[i]) && (source.shape[i] != 1) && (shape[i] != 1))
                throw invalid_argument("ptp error : shape not match");
        if (shape[dim - 1] != source.shape[dim - 2])
            throw invalid_argument("ptp error : shape not match");
        // Compute new shape for the result
        int* new_shape = new int[dim];
        for (int i = 0; i < dim - 2; i++)
            new_shape[i] = (shape[i] >= source.shape[i]) ? shape[i] : source.shape[i];
        new_shape[dim - 2] = shape[dim - 2];
        new_shape[dim - 1] = source.shape[dim - 1];
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        delete[] new_shape;
        int step = rt.shape[dim - 1] * rt.shape[dim - 2];
        // Perform matrix multiplication
        for (int i = 0; i < rt.size; i += step) {
            const dtype* this_mat = rt.broadcast(*this, i);
            const dtype* source_mat = rt.broadcast(source, i);
            for (int _0 = 0; _0 < rt.shape[dim - 2]; _0++) {
                for (int _1 = 0; _1 < rt.shape[dim - 1]; _1++) {
                    rt.arr[i + _0 * rt.shape[dim - 1] + _1] = 0;
                    for (int _2 = 0; _2 < shape[dim - 1]; _2++) {
                        rt.arr[i + _0 * rt.shape[dim - 1] + _1] +=
                                this_mat[_0 * shape[dim - 1] + _2] * source_mat[_2 * source.shape[dim - 1] + _1];
                    }
                }
            }
        }
        return rt;
    }

    // Expand Dimensions: Adds a new dimension at the specified axis
    void expand_dims(int axis) {
        dim += 1;
        int* new_shape = new int[dim];
        int* count = shape;
        // Insert a dimension of size 1 at the specified axis
        for (int i = 0; i < dim; i++) {
            if (i == axis)
                new_shape[i++] = 1;
            if (i < dim)
                new_shape[i] = *count++;
        }
        delete[] shape;
        shape = new_shape;
    }

    // Squeeze: Removes dimensions of size 1
    void squeeze() {
        int count = 0;  // Count of non-1 dimensions
        for (int i = 0; i < dim; i++)
            if (shape[i] != 1)
                count++;
        int* new_shape = new int[count];
        int* iter = new_shape;
        // Copy only non-1 dimensions
        for (int i = 0; i < dim; i++)
            if (shape[i] != 1)
                *(iter++) = shape[i];
        delete[] shape;
        shape = new_shape;
        dim = count;
    }

    // Reshape: Changes the array's shape to new dimensions
    void reshape(int new_dim, int* new_shape) {
        int new_size = 1;
        for (int i = 0; i < new_dim; i++)
            new_size *= new_shape[i];
        if (new_size != size)  // Check if new size matches original
            return;
        dim = new_dim;
        delete[] shape;
        shape = new int[dim];
        for (int i = 0; i < dim; i++)
            shape[i] = new_shape[i];
    }

    // Sum: Computes the sum of all elements
    dtype sum() {
        return global_op([](dtype a, dtype b){ return a + b; });
    }

    // Max: Finds the maximum element
    dtype max() {
        return global_op([](dtype a, dtype b){ return (a > b) ? a : b; });
    }

    // Min: Finds the minimum element
    dtype min() {
        return global_op([](dtype a, dtype b){ return (a < b) ? a : b; });
    }

    // Axis Sum: Computes the sum along a specified axis
    numc axis_sum(int axis) {
        return axis_op(axis, [](dtype a, dtype b){ return a + b; });
    }

    // Axis Max: Finds the maximum along a specified axis
    numc axis_max(int axis) {
        return axis_op(axis, [](dtype a, dtype b){ return (a > b) ? a : b; }, true);
    }

    // Axis Min: Finds the minimum along a specified axis
    numc axis_min(int axis) {
        return axis_op(axis, [](dtype a, dtype b){ return (a < b) ? a : b; }, true);
    }

    // Print: Displays the array's dimensions, shape, steps, and elements
    void print() {
        cout << "dim = " << dim << endl;
        int steps[dim];
        steps[dim - 1] = 1;  // Step size for the last dimension
        cout << "shape = (";
        for (int i = 0; i < dim; i++)
            cout << shape[i] << ",";
        cout << ")" << endl;
        // Compute step sizes for each dimension
        for (int i = dim - 2; i >= 0; i--)
            steps[i] = steps[i + 1] * shape[i + 1];
        cout << "steps = (";
        for (int i = 0; i < dim; i++)
            cout << steps[i] << ",";
        cout << ")" << endl;
        // Print each element with its multi-dimensional index
        for (int i = 0; i < size; i++) {
            cout << "idx : [";
            for (int j = 0; j < dim; j++)
                cout << (i / steps[j]) % shape[j] << ",";
            cout << "] = " << arr[i] << endl;
        }
    }
};