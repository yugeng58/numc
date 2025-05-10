#include <iostream>
#include <stdexcept>
#include "omp.h"
#pragma omp requires unified_shared_memory

//using namespace std;

template <class dtype>
class numc {
private:
    int dim;          // Number of dimensions
    int* shape;       // Shape of each dimension
    dtype* arr;       // Flattened array
    int size;         // Total number of elements
    bool temp;        // Flag indicating if it's a temporary array

public:
    // Constructor: Create a numc object
    explicit numc(int dim, const int* shape, dtype* source = nullptr): dim(dim), size(1) {
        if (dim)
            this->shape = new int[dim];
        else
            this->shape = nullptr;
        for (int i = 0; i < this->dim; i++) {
            this->shape[i] = shape[i];
            this->size *= shape[i];
        }
        if (source) {
            arr = source;
            temp = true;
        } else {
            arr = new dtype[this->size];
#pragma omp parallel for default(none)
            for (int i = 0; i < size; i++)
                arr[i] = 0;
            temp = false;
        }
    }

    // Copy constructor: Deep copy another numc object
    numc(const numc & source): numc(source.dim, source.shape) {
        for (int i = 0; i < this->size; i++)
            arr[i] = source.arr[i];
    }

    // Destructor: Free allocated memory
    ~numc() {
        delete[] shape;
        if (!temp)
            delete[] arr;
    }

    // Assignment operator: Assign from another numc object
    numc & operator= (const numc & source) {
        if (temp) {
            if (dim != source.dim)
                throw std::invalid_argument("shape not match");
            for (int i = 0; i < dim; i++)
                if (shape[i] != source.shape[i])
                    throw std::invalid_argument("shape not match");
#pragma omp parallel for default(none) shared(source)
            for (int i = 0; i < size; i++)
                arr[i] = source.arr[i];
        } else {
            dim = source.dim;
            delete [] shape;
            shape = new int[dim];
            for (int i = 0; i < dim; i++)
                shape[i] = source.shape[i];
            size = source.size;
            delete [] arr;
            arr = new dtype[size];
#pragma omp parallel for default(none) shared(source)
            for (int i = 0; i < size; i++)
                arr[i] = source.arr[i];
        }
        return *this;
    }

    // Index operator: Access subarray
    numc operator[](int idx) {
        return numc<dtype>(dim - 1, shape + 1, &(arr[idx * (size / shape[0])]));
    }

    // Slice: Extract slice along the first dimension
    numc slice(int from, int to, int step) {
        int* new_shape = new int[dim];
        new_shape[0] = (to - from) / step;
        for (int i = 1; i < dim; i++)
            new_shape[i] = shape[i];
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        delete [] new_shape;
        for (int i = 0; i < rt.shape[0]; i++)
            rt[i] = (*this)[from + i * step];
        return rt;
    }

    // Get element: Return reference to element at specified index
    dtype & get(int idx = 0) {
        return arr[idx];
    }

    // Broadcast: Map index to broadcasted array element
    const dtype* broadcast(const numc& source, const int& idx) const {
        int step = 1;
        int source_step = 1;
        int target_idx = 0;
        for (int i = dim - 1; i >= 0; i--) {
            if (source.shape[i] == shape[i])
                target_idx += source_step * ((idx / step) % shape[i]);
            step *= shape[i];
            source_step *= source.shape[i];
        }
        return &source.arr[target_idx];
    }

    // Broadcast check: Verify if broadcasting with another numc object is possible
    bool broadcast_check(const numc & source) const {
        if (dim != source.dim)
            return false;
        for (int i = 0; i < dim; i++)
            if ((shape[i] != source.shape[i]) && (source.shape[i] != 1) && (shape[i] != 1))
                return false;
        return true;
    }

    // Broadcast shape: Compute shape after broadcasting with another numc object
    int* broadcast_shape(const numc & source) const {
        int* rt = new int[dim];
        for (int i = 0; i < dim; i++)
            rt[i] = (shape[i] >= source.shape[i]) ? shape[i] : source.shape[i];
        return rt;
    }

    // Global operation: Apply global operation (e.g., sum, max) to the entire array
    dtype global_op(dtype (*fptr)(dtype, dtype), bool init_to_first_term = false) {
        dtype rt = 0;
        if (init_to_first_term)
            rt = arr[0];
#pragma omp parallel for default(none) shared(rt,fptr)
        for (int i = 0; i < size; i++)
            rt = fptr(rt, arr[i]);
        return rt;
    }

    // Global operation (with index): Apply indexed global operation to the entire array
    dtype global_op(dtype (*fptr)(dtype, dtype, int), bool init_to_first_term = false) {
        dtype rt = 0;
        if (init_to_first_term)
            rt = arr[0];
#pragma omp parallel for default(none) shared(rt,fptr)
        for (int i = 0; i < size; i++)
            rt = fptr(rt, arr[i], i);
        return rt;
    }

    // Pointwise operation: Apply pointwise operation to each element
    numc p_op(dtype (*fptr)(dtype)) {
        numc<dtype> rt = numc<dtype>(*this);
#pragma omp parallel for default(none) shared(rt,fptr) // Parallelize loop with OpenMP
        for (int i = 0; i < size; i++)
            rt.arr[i] = fptr(arr[i]);
        return rt;
    }

    // Pointwise operation (with index): Apply indexed pointwise operation to each element
    numc p_op(dtype (*fptr)(dtype, int)) {
        numc<dtype> rt = numc<dtype>(*this);
#pragma omp parallel for default(none) shared(rt,fptr) // Parallelize loop with OpenMP
        for (int i = 0; i < size; i++)
            rt.arr[i] = fptr(arr[i], i);
        return rt;
    }

    // Broadcast pointwise operation: Apply broadcasted pointwise operation
    numc ptp_broadcast_op(const numc & source, dtype (*fptr)(dtype, dtype)) {
        if (!broadcast_check(source))
            throw std::invalid_argument("ptp error : shape not match");
        int* new_shape = broadcast_shape(source);
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        delete [] new_shape;
#pragma omp parallel for default(none) shared(rt,source,fptr)// Parallelize loop with OpenMP
        for (int i = 0; i < rt.size; i++) {
            rt.arr[i] = fptr(*rt.broadcast(*this, i), *rt.broadcast(source, i));
        }
        return rt;
    }

    // Broadcast pointwise operation (with index): Apply indexed broadcasted pointwise operation
    numc ptp_broadcast_op(const numc & source, dtype (*fptr)(dtype, dtype, int)) {
        if (!broadcast_check(source))
            throw std::invalid_argument("ptp error : shape not match");
        int* new_shape = broadcast_shape(source);
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        delete [] new_shape;
#pragma omp parallel for default(none) shared(rt,source,fptr)// Parallelize loop with OpenMP
        for (int i = 0; i < rt.size; i++) {
            rt.arr[i] = fptr(*rt.broadcast(*this, i), *rt.broadcast(source, i), i);
        }
        return rt;
    }

    // Axis operation: Apply operation along specified axis
    numc axis_op(int axis, dtype (*fptr)(dtype, dtype), bool init_to_first_term = false) {
        int step = 1;
        int mod = 1;
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
#pragma omp parallel for default(none) shared(rt,fptr) collapse(2)  // Parallelize outer two loops with OpenMP
        for (int i = 0; i < mod; i++)
            for (int j = 0; j < step; j++) {
                if (init_to_first_term)
                    rt.arr[i * step + j] = arr[i * step * shape[axis]];
                for (int k = 0; k < shape[axis]; k++)
                    rt.arr[i * step + j] = fptr(rt.arr[i * step + j], arr[i * step * shape[axis] + k * step + j]);
            }
        return rt;
    }

    // Axis operation (with index): Apply indexed operation along specified axis
    numc axis_op(int axis, dtype (*fptr)(dtype, dtype, int), bool init_to_first_term = false) {
        int step = 1;
        int mod = 1;
        for (int i = dim - 1; i > axis; i--)
            step *= shape[i];
        int* new_shape = new int[dim];
        for (int i = 0; i < dim; i++) {
            if (i == axis)
                new_shape[i] = 1;
            else
                new_shape[i] = shape[i];
            if (i < axis)
                mod *= shape[i];
        }
        numc<dtype> rt = numc<dtype>(dim, new_shape);
#pragma omp parallel for default(none) shared(rt,fptr) collapse(2)  // Parallelize outer two loops with OpenMP
        for (int i = 0; i < mod; i++)
            for (int j = 0; j < step; j++) {
                if (init_to_first_term)
                    rt.arr[i * step + j] = arr[i * step * shape[axis]];
                for (int k = 0; k < shape[axis]; k++)
                    rt.arr[i * step + j] = fptr(rt.arr[i * step + j], arr[i * step * shape[axis] + k * step + j], i * step * shape[axis] + k * step + j);
            }
        return rt;
    }

    // Axis expand: Expand array along specified axis
    numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype)) {
        if (shape[axis] != 1)
            throw std::invalid_argument("ptp error : shape not match");
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
            if (i < axis)
                mod *= shape[i];
        }
        numc<dtype> rt = numc<dtype>(dim, new_shape);
#pragma omp parallel for default(none) shared(rt,fptr) collapse(3)  // Parallelize outer two loops with OpenMP
        for (int i = 0; i < mod; i++)
            for (int j = 0; j < step; j++)
                for (int k = 0; k < axis_target; k++)
                    rt.arr[i * step * axis_target + k * step + j] = fptr(arr[i * step + j]);
        return rt;
    }

    // Axis expand (with index): Expand array along specified axis with index
    numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype, int)) {
        if (shape[axis] != 1)
            throw std::invalid_argument("ptp error : shape not match");
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
            if (i < axis)
                mod *= shape[i];
        }
        numc<dtype> rt = numc<dtype>(dim, new_shape);
#pragma omp parallel for default(none) shared(rt,fptr) collapse(3)  // Parallelize outer two loops with OpenMP
        for (int i = 0; i < mod; i++)
            for (int j = 0; j < step; j++)
                for (int k = 0; k < axis_target; k++)
                    rt.arr[i * step * axis_target + k * step + j] = fptr(arr[i * step + j], k);
        return rt;
    }

    // Arithmetic operator: Element-wise addition with broadcasting
    numc operator+ (const numc & source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b){ return a + b; });
    }

    // Arithmetic operator: Element-wise subtraction with broadcasting
    numc operator- (const numc & source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b){ return a - b; });
    }

    // Arithmetic operator: Element-wise multiplication with broadcasting
    numc operator* (const numc & source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b){ return a * b; });
    }

    // Arithmetic operator: Element-wise division with broadcasting
    numc operator/ (const numc & source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b){ return a / b; });
    }

    // Matrix multiplication: Matrix multiplication with broadcasting
    numc matmul(const numc & source) {
        if (dim != source.dim || dim < 2)
            throw std::invalid_argument("ptp error : shape not match");
        for (int i = 0; i < dim - 2; i++)
            if ((shape[i] != source.shape[i]) && (source.shape[i] != 1) && (shape[i] != 1))
                throw std::invalid_argument("ptp error : shape not match");
        if (shape[dim - 1] != source.shape[dim - 2])
            throw std::invalid_argument("ptp error : shape not match");
        int* new_shape = new int[dim];
        for (int i = 0; i < dim - 2; i++)
            new_shape[i] = (shape[i] >= source.shape[i]) ? shape[i] : source.shape[i];
        new_shape[dim - 2] = shape[dim - 2];
        new_shape[dim - 1] = source.shape[dim - 1];
        numc<dtype> rt = numc<dtype>(dim, new_shape);
        delete [] new_shape;
        int step = rt.shape[dim - 1] * rt.shape[dim - 2];
#pragma omp parallel for default(none) shared(rt,source) // Parallelize outer loop with OpenMP
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

    // Expand dimensions: Add a new dimension at specified axis
    void expand_dims(int axis) {
        dim += 1;
        int* new_shape = new int[dim];
        int* count = shape;
        for (int i = 0; i < dim; i++) {
            if (i == axis)
                new_shape[i++] = 1;
            if (i < dim)
                new_shape[i] = *count++;
        }
        delete [] shape;
        shape = new_shape;
    }

    // Squeeze: Remove dimensions of size 1
    void squeeze() {
        int count = 0;
        for (int i = 0; i < dim; i++)
            if (shape[i] != 1)
                count++;
        int* new_shape = new int[count];
        int* iter = new_shape;
        for (int i = 0; i < dim; i++)
            if (shape[i] != 1)
                *(iter++) = shape[i];
        delete [] shape;
        shape = new_shape;
        dim = count;
    }

    // Reshape: Reshape array to new dimensions and shape
    void reshape(int new_dim,const int* new_shape) {
        int new_size = 1;
        for (int i = 0; i < new_dim; i++)
            new_size *= new_shape[i];
        if (new_size != size)
            return;
        dim = new_dim;
        delete [] shape;
        shape = new int[dim];
        for (int i = 0; i < dim; i++)
            shape[i] = new_shape[i];
    }

    // Sum: Compute sum of all elements
    dtype sum() {
        return global_op([](dtype a, dtype b){ return a + b; });
    }

    // Max: Find maximum element
    dtype max() {
        return global_op([](dtype a, dtype b){ return (a > b) ? a : b; });
    }

    // Min: Find minimum element
    dtype min() {
        return global_op([](dtype a, dtype b){ return (a < b) ? a : b; });
    }

    // Axis sum: Compute sum along specified axis
    numc axis_sum(int axis) {
        return axis_op(axis, [](dtype a, dtype b){ return a + b; });
    }

    // Axis max: Find maximum along specified axis
    numc axis_max(int axis) {
        return axis_op(axis, [](dtype a, dtype b){ return (a > b) ? a : b; }, true);
    }

    // Axis min: Find minimum along specified axis
    numc axis_min(int axis) {
        return axis_op(axis, [](dtype a, dtype b){ return (a < b) ? a : b; }, true);
    }

    // Print: Display dimensions, shape, and elements
    void print() {
        std::cout << "dim = " << dim << std::endl;
        int steps[dim];
        steps[dim - 1] = 1;
        std::cout << "shape = (";
        for (int i = 0; i < dim; i++)
            std::cout << shape[i] << ",";
        std::cout << ")" << std::endl;
        for (int i = dim - 2; i >= 0; i--)
            steps[i] = steps[i + 1] * shape[i + 1];
        std::cout << "steps = (";
        for (int i = 0; i < dim; i++)
            std::cout << steps[i] << ",";
        std::cout << ")" << std::endl;
        for (int i = 0; i < size; i++) {
            std::cout << "idx : [";
            for (int j = 0; j < dim; j++)
                std::cout << (i / steps[j]) % shape[j] << ",";
            std::cout << "] = " << arr[i] << std::endl;
        }
    }
};