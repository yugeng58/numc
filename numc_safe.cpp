#include <iostream>
#include <stdexcept>
#include <string>
#include "omp.h"
#pragma omp requires unified_shared_memory

// Template class for multi-dimensional arrays with NumPy-like functionality
template <class dtype>
class numc {
private:
    int dim;          // Number of dimensions
    int* shape;       // Array specifying the size of each dimension
    dtype* arr;       // Flattened array storing the data
    int size;         // Total number of elements (product of shape)
    bool temp;        // Flag indicating if the array is temporary (not owned)

public:
    /**
     * @brief Constructor to create a numc array.
     * @param dim Number of dimensions.
     * @param shape Pointer to an array specifying the size of each dimension.
     * @param source Optional pointer to an existing array to use as data. If provided, the array is not owned and will not be deleted.
     */
    explicit numc(int dim, const int* shape, dtype* source = nullptr) : dim(dim), size(1), temp(false) {
        if (dim < 0) {
            throw std::invalid_argument("Dimension must be non-negative.");
        }
        try {
            this->shape = dim ? new int[dim] : nullptr;
            for (int i = 0; i < dim; i++) {
                if (shape[i] <= 0) {
                    throw std::invalid_argument("Shape dimensions must be positive: shape[" + std::to_string(i) + "] = " + std::to_string(shape[i]));
                }
                this->shape[i] = shape[i];
                size *= shape[i];
            }
            if (source) {
                arr = source;
                temp = true;
            } else {
                arr = new dtype[size];
#pragma omp parallel for
                for (int i = 0; i < size; i++) {
                    arr[i] = 0;
                }
            }
        } catch (const std::bad_alloc& e) {
            delete[] this->shape;
            throw std::runtime_error("Memory allocation failed in constructor: " + std::string(e.what()));
        } catch (...) {
            delete[] this->shape;
            throw;
        }
    }

    /**
     * @brief Copy constructor to perform a deep copy of another numc object.
     * @param source The numc object to copy from.
     */
    numc(const numc& source) : dim(source.dim), size(source.size), temp(false) {
        try {
            shape = new int[dim];
            for (int i = 0; i < dim; i++) {
                shape[i] = source.shape[i];
            }
            arr = new dtype[size];
            for (int i = 0; i < size; i++) {
                arr[i] = source.arr[i];
            }
        } catch (const std::bad_alloc& e) {
            delete[] shape;
            throw std::runtime_error("Memory allocation failed in copy constructor: " + std::string(e.what()));
        }
    }


    numc(numc&& other): dim(other.dim),shape(other.shape),size(other.size),temp(true){
        if(other.temp == false)
            arr = other.arr;
        else{
            arr = new int[size];
#pragma omp parallel for
            for(int i = 0;i < size;i++)
                arr[i] = other.arr[i];
        }
        other.shape = nullptr;
        other.arr = nullptr;
    }

    /**
     * @brief Assignment operator to copy data from another numc object.
     * @param source The numc object to assign from.
     * @return Reference to the current object.
     */
    numc& operator=(const numc& source) {
        if (this == &source) return *this; // Self-assignment check
        if (temp) {
            if (dim != source.dim) {
                throw std::invalid_argument("Dimension mismatch in assignment: " + std::to_string(dim) + " vs " + std::to_string(source.dim));
            }
            for (int i = 0; i < dim; i++) {
                if (shape[i] != source.shape[i]) {
                    throw std::invalid_argument("Shape mismatch in assignment at dimension " + std::to_string(i) + ": " + std::to_string(shape[i]) + " vs " + std::to_string(source.shape[i]));
                }
            }
#pragma omp parallel for
            for (int i = 0; i < size; i++) {
                arr[i] = source.arr[i];
            }
        }
        else{
            int* new_shape = nullptr;
            dtype* new_arr = nullptr;
            try {
                new_shape = new int[source.dim];
                for (int i = 0; i < source.dim; i++) {
                    new_shape[i] = source.shape[i];
                }
                new_arr = new dtype[source.size];
#pragma omp parallel for
                for (int i = 0; i < source.size; i++) {
                    new_arr[i] = source.arr[i];
                }
                delete[] shape;
                delete[] arr;
                dim = source.dim;
                shape = new_shape;
                arr = new_arr;
                size = source.size;
            } catch (const std::bad_alloc& e) {
                delete[] new_shape;
                delete[] new_arr;
                throw std::runtime_error("Memory allocation failed in assignment: " + std::string(e.what()));
            }
        }
        return *this;
    }

    numc& operator=(numc&& other){
        if (this != &other)
            return *this;
        delete [] shape;
        delete [] arr;
        size = other.size;
        dim = other.dim;
        shape = other.shape;
        temp = true;
        if(other.temp == false)
            arr = other.arr;
        else{
            arr = new int[size];
#pragma omp parallel for
            for(int i = 0;i < size;i++)
                arr[i] = other.arr[i];
        }
        other.shape = nullptr;
        other.arr = nullptr;
    }

    /**
     * @brief Destructor to free allocated memory.
     */
    ~numc() {
        delete[] shape;
        if (!temp) {
            delete[] arr;
        }
    }

    /**
     * @brief Iterator class for numc array, supporting random access.
     */
    class iterator {
    private:
        dtype* ptr; // Pointer to the current element
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = dtype;
        using difference_type   = std::ptrdiff_t;
        using pointer           = dtype*;
        using reference         = dtype&;

        explicit iterator(dtype* p = nullptr) : ptr(p) {}

        reference operator*() const { return *ptr; }
        pointer operator->() const { return ptr; }

        iterator& operator++() { ++ptr; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++ptr; return tmp; }
        iterator& operator--() { --ptr; return *this; }
        iterator operator--(int) { iterator tmp = *this; --ptr; return tmp; }

        iterator operator+(difference_type n) const { return iterator(ptr + n); }
        iterator operator-(difference_type n) const { return iterator(ptr - n); }
        difference_type operator-(const iterator& other) const { return ptr - other.ptr; }

        iterator& operator+=(difference_type n) { ptr += n; return *this; }
        iterator& operator-=(difference_type n) { ptr -= n; return *this; }

        reference operator[](difference_type n) const { return ptr[n]; }

        bool operator==(const iterator& other) const { return ptr == other.ptr; }
        bool operator!=(const iterator& other) const { return ptr != other.ptr; }
        bool operator<(const iterator& other) const { return ptr < other.ptr; }
        bool operator>(const iterator& other) const { return ptr > other.ptr; }
        bool operator<=(const iterator& other) const { return ptr <= other.ptr; }
        bool operator>=(const iterator& other) const { return ptr >= other.ptr; }
    };

    /**
     * @brief Returns an iterator to the beginning of the array.
     */
    iterator begin() { return iterator(arr); }

    /**
     * @brief Returns an iterator to the end of the array.
     */
    iterator end() { return iterator(arr + size); }

    /**
     * @brief Index operator to access a subarray.
     * @param idx Index along the first dimension.
     * @return A new numc object representing the subarray.
     */
    numc operator[](int idx) {
        if (dim < 1) {
            throw std::out_of_range("Cannot index array with dimension < 1.");
        }
        if (idx < 0 || idx >= shape[0]) {
            throw std::out_of_range("Index out of bounds: " + std::to_string(idx) + " not in [0, " + std::to_string(shape[0]) + ")");
        }
        return numc<dtype>(dim - 1, shape + 1, &(arr[idx * (size / shape[0])]));
    }

    /**
     * @brief Extracts a slice along the first dimension.
     * @param from Starting index (inclusive).
     * @param to Ending index (exclusive).
     * @param step Step size between elements.
     * @return A new numc object containing the sliced data.
     */
    numc slice(int from, int to, int step) {
        if (dim < 1) {
            throw std::invalid_argument("Cannot slice array with dimension < 1.");
        }
        if (from < 0 || to > shape[0] || from >= to || step <= 0) {
            throw std::invalid_argument("Invalid slice parameters: from=" + std::to_string(from) + ", to=" + std::to_string(to) + ", step=" + std::to_string(step));
        }
        int new_size = (to - from + step - 1) / step; // Ceiling division
        int* new_shape = nullptr;
        try {
            new_shape = new int[dim];
            new_shape[0] = new_size;
            for (int i = 1; i < dim; i++) {
                new_shape[i] = shape[i];
            }
            numc<dtype> rt(dim, new_shape);
            delete[] new_shape;
            for (int i = 0; i < rt.shape[0]; i++) {
                rt[i] = (*this)[from + i * step];
            }
            return rt;
        } catch (...) {
            delete[] new_shape;
            throw;
        }
    }

    /**
     * @brief Returns a reference to an element at the specified index in the flattened array.
     * @param idx Index of the element (default is 0).
     * @return Reference to the element.
     */
    dtype& get(int idx = 0) {
        if (idx < 0 || idx >= size) {
            throw std::out_of_range("Index out of bounds: " + std::to_string(idx) + " not in [0, " + std::to_string(size) + ")");
        }
        return arr[idx];
    }

    /**
     * @brief Maps an index in a broadcasted array to the corresponding element in the source array.
     * @param source The source numc object.
     * @param idx Index in the broadcasted array.
     * @return Pointer to the corresponding element in the source array.
     */
    const dtype* broadcast(const numc& source, const int& idx) const {
        int step = 1;
        int source_step = 1;
        int target_idx = 0;
        for (int i = dim - 1; i >= 0; i--) {
            if (source.shape[i] == shape[i]) {
                target_idx += source_step * ((idx / step) % shape[i]);
            }
            step *= shape[i];
            source_step *= source.shape[i];
        }
        return &source.arr[target_idx];
    }

    /**
     * @brief Checks if broadcasting with another numc object is possible.
     * @param source The numc object to check compatibility with.
     * @return True if broadcasting is possible, false otherwise.
     */
    bool broadcast_check(const numc& source) const {
        if (dim != source.dim) return false;
        for (int i = 0; i < dim; i++) {
            if (shape[i] != source.shape[i] && source.shape[i] != 1 && shape[i] != 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Computes the shape of the resulting array after broadcasting.
     * @param source The numc object to broadcast with.
     * @return Pointer to a new array containing the broadcasted shape (must be deleted by caller).
     */
    int* broadcast_shape(const numc& source) const {
        int* rt = new int[dim];
        for (int i = 0; i < dim; i++) {
            rt[i] = (shape[i] >= source.shape[i]) ? shape[i] : source.shape[i];
        }
        return rt;
    }

    /**
     * @brief Applies a global operation to the entire array (serial execution).
     * @param fptr Function pointer to the operation (e.g., sum, max).
     * @param init_to_first_term If true, initializes the result to the first element.
     * @return Result of the operation.
     */
    dtype global_op(dtype (*fptr)(dtype, dtype), bool init_to_first_term = false) {
        if (size == 0) {
            throw std::runtime_error("Cannot perform global operation on empty array.");
        }
        dtype rt = init_to_first_term ? arr[0] : 0;
        for (int i = init_to_first_term ? 1 : 0; i < size; i++) {
            rt = fptr(rt, arr[i]);
        }
        return rt;
    }

    /**
     * @brief Applies an indexed global operation to the entire array (serial execution).
     * @param fptr Function pointer to the indexed operation.
     * @param init_to_first_term If true, initializes the result to the first element.
     * @return Result of the operation.
     */
    dtype global_op(dtype (*fptr)(dtype, dtype, int), bool init_to_first_term = false) {
        if (size == 0) {
            throw std::runtime_error("Cannot perform global operation on empty array.");
        }
        dtype rt = init_to_first_term ? arr[0] : 0;
        for (int i = init_to_first_term ? 1 : 0; i < size; i++) {
            rt = fptr(rt, arr[i], i);
        }
        return rt;
    }

    /**
     * @brief Applies a pointwise operation to each element.
     * @param fptr Function pointer to the operation (e.g., square, negate).
     * @return A new numc object with the operation applied.
     */
    numc p_op(dtype (*fptr)(dtype)) {
        numc<dtype> rt(*this);
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            rt.arr[i] = fptr(arr[i]);
        }
        return rt;
    }

    /**
     * @brief Applies an indexed pointwise operation to each element.
     * @param fptr Function pointer to the indexed operation.
     * @return A new numc object with the operation applied.
     */
    numc p_op(dtype (*fptr)(dtype, int)) {
        numc<dtype> rt(*this);
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            rt.arr[i] = fptr(arr[i], i);
        }
        return rt;
    }

    /**
     * @brief Applies a broadcasted pointwise operation with another numc object.
     * @param source The numc object to operate with.
     * @param fptr Function pointer to the operation (e.g., add, multiply).
     * @return A new numc object with the operation applied.
     */
    numc ptp_broadcast_op(const numc& source, dtype (*fptr)(dtype, dtype)) {
        if (!broadcast_check(source)) {
            std::string msg = "Broadcasting not possible. Shapes: (";
            for (int i = 0; i < dim; i++) msg += (i > 0 ? ", " : "") + std::to_string(shape[i]);
            msg += ") vs (";
            for (int i = 0; i < source.dim; i++) msg += (i > 0 ? ", " : "") + std::to_string(source.shape[i]);
            msg += ")";
            throw std::invalid_argument(msg);
        }
        int* new_shape = nullptr;
        try {
            new_shape = broadcast_shape(source);
            numc<dtype> rt(dim, new_shape);
            delete[] new_shape;
#pragma omp parallel for
            for (int i = 0; i < rt.size; i++) {
                rt.arr[i] = fptr(*rt.broadcast(*this, i), *rt.broadcast(source, i));
            }
            return rt;
        } catch (...) {
            delete[] new_shape;
            throw;
        }
    }

    /**
     * @brief Applies an indexed broadcasted pointwise operation with another numc object.
     * @param source The numc object to operate with.
     * @param fptr Function pointer to the indexed operation.
     * @return A new numc object with the operation applied.
     */
    numc ptp_broadcast_op(const numc& source, dtype (*fptr)(dtype, dtype, int)) {
        if (!broadcast_check(source)) {
            std::string msg = "Broadcasting not possible. Shapes: (";
            for (int i = 0; i < dim; i++) msg += (i > 0 ? ", " : "") + std::to_string(shape[i]);
            msg += ") vs (";
            for (int i = 0; i < source.dim; i++) msg += (i > 0 ? ", " : "") + std::to_string(source.shape[i]);
            msg += ")";
            throw std::invalid_argument(msg);
        }
        int* new_shape = nullptr;
        try {
            new_shape = broadcast_shape(source);
            numc<dtype> rt(dim, new_shape);
            delete[] new_shape;
#pragma omp parallel for
            for (int i = 0; i < rt.size; i++) {
                rt.arr[i] = fptr(*rt.broadcast(*this, i), *rt.broadcast(source, i), i);
            }
            return rt;
        } catch (...) {
            delete[] new_shape;
            throw;
        }
    }

    /**
     * @brief Applies an operation along a specified axis (serial execution).
     * @param axis The axis to operate along.
     * @param fptr Function pointer to the operation.
     * @param init_to_first_term If true, initializes the result to the first element.
     * @return A new numc object with the operation applied.
     */
    numc axis_op(int axis, dtype (*fptr)(dtype, dtype), bool init_to_first_term = false) {
        if (axis < 0 || axis >= dim) {
            throw std::invalid_argument("Axis out of range: " + std::to_string(axis) + " not in [0, " + std::to_string(dim) + ")");
        }
        int step = 1, mod = 1;
        int* new_shape = nullptr;
        try {
            new_shape = new int[dim];
            for (int i = 0; i < dim; i++) {
                new_shape[i] = (i == axis) ? 1 : shape[i];
                if (i > axis) step *= shape[i];
                if (i < axis) mod *= shape[i];
            }
            numc<dtype> rt(dim, new_shape);
            delete[] new_shape;
#pragma omp parallel for collapse(2)
            for (int i = 0; i < mod; i++) {
                for (int j = 0; j < step; j++) {
                    rt.arr[i * step + j] = init_to_first_term ? arr[i * step * shape[axis]] : 0;
                    for (int k = init_to_first_term ? 1 : 0; k < shape[axis]; k++) {
                        rt.arr[i * step + j] = fptr(rt.arr[i * step + j], arr[i * step * shape[axis] + k * step + j]);
                    }
                }
            }
            return rt;
        } catch (...) {
            delete[] new_shape;
            throw;
        }
    }

    /**
     * @brief Applies an indexed operation along a specified axis (serial execution).
     * @param axis The axis to operate along.
     * @param fptr Function pointer to the indexed operation.
     * @param init_to_first_term If true, initializes the result to the first element.
     * @return A new numc object with the operation applied.
     */
    numc axis_op(int axis, dtype (*fptr)(dtype, dtype, int), bool init_to_first_term = false) {
        if (axis < 0 || axis >= dim) {
            throw std::invalid_argument("Axis out of range: " + std::to_string(axis) + " not in [0, " + std::to_string(dim) + ")");
        }
        int step = 1, mod = 1;
        int* new_shape = nullptr;
        try {
            new_shape = new int[dim];
            for (int i = 0; i < dim; i++) {
                new_shape[i] = (i == axis) ? 1 : shape[i];
                if (i > axis) step *= shape[i];
                if (i < axis) mod *= shape[i];
            }
            numc<dtype> rt(dim, new_shape);
            delete[] new_shape;
#pragma omp parallel for collapse(2)
            for (int i = 0; i < mod; i++) {
                for (int j = 0; j < step; j++) {
                    rt.arr[i * step + j] = init_to_first_term ? arr[i * step * shape[axis]] : 0;
                    for (int k = init_to_first_term ? 1 : 0; k < shape[axis]; k++) {
                        rt.arr[i * step + j] = fptr(rt.arr[i * step + j], arr[i * step * shape[axis] + k * step + j], i * step * shape[axis] + k * step + j);
                    }
                }
            }
            return rt;
        } catch (...) {
            delete[] new_shape;
            throw;
        }
    }

    /**
     * @brief Expands the array along a specified axis.
     * @param axis The axis to expand (must have size 1).
     * @param axis_target The target size for the expanded axis.
     * @param fptr Function pointer to compute new elements.
     * @return A new numc object with the expanded dimension.
     */
    numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype)) {
        if (axis < 0 || axis >= dim) {
            throw std::invalid_argument("Axis out of range: " + std::to_string(axis) + " not in [0, " + std::to_string(dim) + ")");
        }
        if (shape[axis] != 1) {
            throw std::invalid_argument("Can only expand axis with size 1, got " + std::to_string(shape[axis]) + " at axis " + std::to_string(axis));
        }
        if (axis_target <= 0) {
            throw std::invalid_argument("Target axis size must be positive: " + std::to_string(axis_target));
        }
        int step = 1, mod = 1;
        int* new_shape = nullptr;
        try {
            new_shape = new int[dim];
            for (int i = 0; i < dim; i++) {
                new_shape[i] = (i == axis) ? axis_target : shape[i];
                if (i > axis) step *= shape[i];
                if (i < axis) mod *= shape[i];
            }
            numc<dtype> rt(dim, new_shape);
            delete[] new_shape;
#pragma omp parallel for collapse(3)
            for (int i = 0; i < mod; i++) {
                for (int j = 0; j < step; j++) {
                    for (int k = 0; k < axis_target; k++) {
                        rt.arr[i * step * axis_target + k * step + j] = fptr(arr[i * step + j]);
                    }
                }
            }
            return rt;
        } catch (...) {
            delete[] new_shape;
            throw;
        }
    }

    /**
     * @brief Expands the array along a specified axis with an indexed function.
     * @param axis The axis to expand (must have size 1).
     * @param axis_target The target size for the expanded axis.
     * @param fptr Function pointer to compute new elements with index.
     * @return A new numc object with the expanded dimension.
     */
    numc axis_expand(int axis, int axis_target, dtype (*fptr)(dtype, int)) {
        if (axis < 0 || axis >= dim) {
            throw std::invalid_argument("Axis out of range: " + std::to_string(axis) + " not in [0, " + std::to_string(dim) + ")");
        }
        if (shape[axis] != 1) {
            throw std::invalid_argument("Can only expand axis with size 1, got " + std::to_string(shape[axis]) + " at axis " + std::to_string(axis));
        }
        if (axis_target <= 0) {
            throw std::invalid_argument("Target axis size must be positive: " + std::to_string(axis_target));
        }
        int step = 1, mod = 1;
        int* new_shape = nullptr;
        try {
            new_shape = new int[dim];
            for (int i = 0; i < dim; i++) {
                new_shape[i] = (i == axis) ? axis_target : shape[i];
                if (i > axis) step *= shape[i];
                if (i < axis) mod *= shape[i];
            }
            numc<dtype> rt(dim, new_shape);
            delete[] new_shape;
#pragma omp parallel for collapse(3)
            for (int i = 0; i < mod; i++) {
                for (int j = 0; j < step; j++) {
                    for (int k = 0; k < axis_target; k++) {
                        rt.arr[i * step * axis_target + k * step + j] = fptr(arr[i * step + j], k);
                    }
                }
            }
            return rt;
        } catch (...) {
            delete[] new_shape;
            throw;
        }
    }

    /**
     * @brief Element-wise addition with broadcasting.
     * @param source The numc object to add.
     * @return A new numc object with the result.
     */
    numc operator+(const numc& source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b) { return a + b; });
    }

    /**
     * @brief Element-wise subtraction with broadcasting.
     * @param source The numc object to subtract.
     * @return A new numc object with the result.
     */
    numc operator-(const numc& source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b) { return a - b; });
    }

    /**
     * @brief Element-wise multiplication with broadcasting.
     * @param source The numc object to multiply.
     * @return A new numc object with the result.
     */
    numc operator*(const numc& source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b) { return a * b; });
    }

    /**
     * @brief Element-wise division with broadcasting.
     * @param source The numc object to divide by.
     * @return A new numc object with the result.
     */
    numc operator/(const numc& source) {
        return ptp_broadcast_op(source, [](dtype a, dtype b) {
            if (b == 0) throw std::runtime_error("Division by zero encountered.");
            return a / b;
        });
    }

    /**
     * @brief Performs matrix multiplication with broadcasting for higher dimensions.
     * @param source The numc object to multiply with.
     * @return A new numc object with the result.
     */
    numc matmul(const numc& source) {
        if (dim != source.dim || dim < 2) {
            throw std::invalid_argument("Matrix multiplication requires same dimensions >= 2: " + std::to_string(dim) + " vs " + std::to_string(source.dim));
        }
        for (int i = 0; i < dim - 2; i++) {
            if (shape[i] != source.shape[i] && source.shape[i] != 1 && shape[i] != 1) {
                throw std::invalid_argument("Broadcasting mismatch at dimension " + std::to_string(i) + ": " + std::to_string(shape[i]) + " vs " + std::to_string(source.shape[i]));
            }
        }
        if (shape[dim - 1] != source.shape[dim - 2]) {
            throw std::invalid_argument("Matrix dimensions do not align: " + std::to_string(shape[dim - 1]) + " vs " + std::to_string(source.shape[dim - 2]));
        }
        int* new_shape = nullptr;
        try {
            new_shape = new int[dim];
            for (int i = 0; i < dim - 2; i++) {
                new_shape[i] = (shape[i] >= source.shape[i]) ? shape[i] : source.shape[i];
            }
            new_shape[dim - 2] = shape[dim - 2];
            new_shape[dim - 1] = source.shape[dim - 1];
            numc<dtype> rt(dim, new_shape);
            delete[] new_shape;
            int step = rt.shape[dim - 1] * rt.shape[dim - 2];
#pragma omp parallel for
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
        } catch (...) {
            delete[] new_shape;
            throw;
        }
    }

    /**
     * @brief Adds a new dimension of size 1 at the specified axis.
     * @param axis The position to insert the new dimension.
     */
    void expand_dims(int axis) {
        if (axis < 0 || axis > dim) {
            throw std::invalid_argument("Axis out of range for expand_dims: " + std::to_string(axis) + " not in [0, " + std::to_string(dim) + "]");
        }
        int* new_shape = nullptr;
        try {
            new_shape = new int[dim + 1];
            int count = 0;
            for (int i = 0; i < dim + 1; i++) {
                if (i == axis) {
                    new_shape[i] = 1;
                } else {
                    new_shape[i] = shape[count++];
                }
            }
            delete[] shape;
            shape = new_shape;
            dim += 1;
        } catch (const std::bad_alloc& e) {
            delete[] new_shape;
            throw std::runtime_error("Memory allocation failed in expand_dims: " + std::string(e.what()));
        }
    }

    /**
     * @brief Removes all dimensions of size 1.
     */
    void squeeze() {
        int count = 0;
        for (int i = 0; i < dim; i++) {
            if (shape[i] != 1) count++;
        }
        int* new_shape = nullptr;
        try {
            new_shape = new int[count];
            int iter = 0;
            for (int i = 0; i < dim; i++) {
                if (shape[i] != 1) {
                    new_shape[iter++] = shape[i];
                }
            }
            delete[] shape;
            shape = new_shape;
            dim = count;
        } catch (const std::bad_alloc& e) {
            delete[] new_shape;
            throw std::runtime_error("Memory allocation failed in squeeze: " + std::string(e.what()));
        }
    }

    /**
     * @brief Reshapes the array to new dimensions and shape.
     * @param new_dim The new number of dimensions.
     * @param new_shape Pointer to the new shape array.
     */
    void reshape(int new_dim, const int* new_shape) {
        if (new_dim < 0) {
            throw std::invalid_argument("New dimension must be non-negative: " + std::to_string(new_dim));
        }
        int new_size = 1;
        for (int i = 0; i < new_dim; i++) {
            if (new_shape[i] <= 0) {
                throw std::invalid_argument("New shape dimensions must be positive: new_shape[" + std::to_string(i) + "] = " + std::to_string(new_shape[i]));
            }
            new_size *= new_shape[i];
        }
        if (new_size != size) {
            throw std::invalid_argument("Total size mismatch in reshape: " + std::to_string(new_size) + " vs " + std::to_string(size));
        }
        int* temp_shape = nullptr;
        try {
            temp_shape = new int[new_dim];
            for (int i = 0; i < new_dim; i++) {
                temp_shape[i] = new_shape[i];
            }
            delete[] shape;
            shape = temp_shape;
            dim = new_dim;
        } catch (const std::bad_alloc& e) {
            delete[] temp_shape;
            throw std::runtime_error("Memory allocation failed in reshape: " + std::string(e.what()));
        }
    }

    /**
     * @brief Computes the sum of all elements.
     * @return The sum.
     */
    dtype sum() {
        return global_op([](dtype a, dtype b) { return a + b; });
    }

    /**
     * @brief Finds the maximum element.
     * @return The maximum value.
     */
    dtype max() {
        return global_op([](dtype a, dtype b) { return (a > b) ? a : b; }, true);
    }

    /**
     * @brief Finds the minimum element.
     * @return The minimum value.
     */
    dtype min() {
        return global_op([](dtype a, dtype b) { return (a < b) ? a : b; }, true);
    }

    /**
     * @brief Computes the sum along a specified axis.
     * @param axis The axis to sum along.
     * @return A new numc object with the sums.
     */
    numc axis_sum(int axis) {
        return axis_op(axis, [](dtype a, dtype b) { return a + b; });
    }

    /**
     * @brief Finds the maximum along a specified axis.
     * @param axis The axis to operate along.
     * @return A new numc object with the maximum values.
     */
    numc axis_max(int axis) {
        return axis_op(axis, [](dtype a, dtype b) { return (a > b) ? a : b; }, true);
    }

    /**
     * @brief Finds the minimum along a specified axis.
     * @param axis The axis to operate along.
     * @return A new numc object with the minimum values.
     */
    numc axis_min(int axis) {
        return axis_op(axis, [](dtype a, dtype b) { return (a < b) ? a : b; }, true);
    }

    /**
     * @brief Prints the dimensions, shape, and elements of the array for debugging.
     */
    void print() {
        std::cout << "dim = " << dim << std::endl;
        std::cout << "shape = (";
        for (int i = 0; i < dim; i++) {
            std::cout << shape[i] << (i < dim - 1 ? "," : "");
        }
        std::cout << ")" << std::endl;
        if (dim == 0) return;
        int steps[dim];
        steps[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; i--) {
            steps[i] = steps[i + 1] * shape[i + 1];
        }
        std::cout << "steps = (";
        for (int i = 0; i < dim; i++) {
            std::cout << steps[i] << (i < dim - 1 ? "," : "");
        }
        std::cout << ")" << std::endl;
        for (int i = 0; i < size; i++) {
            std::cout << "idx : [";
            for (int j = 0; j < dim; j++) {
                std::cout << (i / steps[j]) % shape[j] << (j < dim - 1 ? "," : "");
            }
            std::cout << "] = " << arr[i] << std::endl;
        }
    }
};