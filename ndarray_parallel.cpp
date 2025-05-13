#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm> // For std::max, std::min
#include "omp.h"
#pragma omp requires unified_shared_memory

template <class type>
class ndarray{
public:
    int dim;
    int* shape;
    int * mod;
    int * step;
    int * stride;
    int * offset;
    type* arr;
    int size;
    bool is_view = false;

    inline int* arr_copy(int* dest, const int * source, int length){
        if(!source)
            return dest;
        else
            for(int i = 0;i < length;i++)
                dest[i] = source[i];
        return dest;
    }

    // Helper function to convert linear index to multi-dimensional index
    inline void linear_to_index(int linear_idx, int* out_idx, int dimensions) const {
        int remaining = linear_idx;
        for(int j = dimensions-1; j >= 0; j--) {
            out_idx[j] = remaining % shape[j];
            remaining /= shape[j];
        }
    }

public:
    explicit ndarray<type>(int dim,const int* shape,type* source = nullptr, const int* stride = nullptr,
                           const int* offset = nullptr,const int* step = nullptr):
            dim(dim),shape(new int [dim]),mod(new int [dim]),step(new int [dim]),stride(new int [dim]),offset(new int [dim]),size(1){

        arr_copy(this->shape,shape,dim);

        for(int i = 0;i < dim;i++)
            size *= shape[i];


        this->mod[0] = 1;
        for (int i = 1; i < dim; i++)
            this->mod[i] = this->mod[i - 1] * shape[i - 1];

        if(!step) {
            this->step[dim - 1] = 1;
            for (int i = dim - 2; i >= 0; i--)
                this->step[i] = this->step[i + 1] * shape[i + 1];
        }
        else
            arr_copy(this->step,step,dim);

        for(int i = 0;i < dim;i++)
            this->stride[i] = (stride) ? stride[i] : 1;

        for(int i = 0;i < dim;i++)
            this->offset[i] = (offset) ? offset[i] : 0;

        if(source) {
            is_view = true;
            arr = source;
        }
        else
            arr = new type[size]{};
    }

    ndarray<type>(ndarray<type>& source): ndarray<type>(source.dim,source.shape){
#pragma omp parallel
        {
            int* local_index = new int[dim]{};
#pragma omp for
            for(int i = 0;i < size;i++) {
                // Use helper function to calculate index
                linear_to_index(i, local_index, dim);
                arr[i] = source.get_value(local_index);
            }
            delete [] local_index;
        }
    }

    ~ndarray<type>(){
        if(!is_view)
            delete [] arr;
        delete [] shape;
        delete [] stride;
        delete [] offset;
        delete [] mod;
        delete [] step;
    }

    ndarray<type>& operator= (ndarray<type>& source){
        if(&source == this)
            return *this;
        if(is_view) {
            element_to_element_mutator(source, [](type a, type b) { return b; });
            return *this;
        }
        else{
            dim = source.dim;
            size = source.size;
            delete [] arr;
            delete [] shape;
            delete [] stride;
            delete [] offset;
            delete [] mod;
            delete [] step;

            arr = new type[source.size];
            shape = new int[source.dim];
            stride = new int[source.dim];
            offset = new int[source.dim];
            mod = new int[source.dim];
            step = new int[source.dim];

            arr_copy(shape, source.shape, dim);

            for(int i = 0;i < dim;i++)
                stride[i] = 1;

            for(int i = 0;i < dim;i++)
                offset[i] = 0;

            this->mod[0] = 1;
            for (int i = 1; i < dim; i++)
                this->mod[i] = this->mod[i - 1] * shape[i - 1];

            this->step[dim - 1] = 1;
            for (int i = dim - 2; i >= 0; i--)
                this->step[i] = this->step[i + 1] * shape[i + 1];

            element_to_element_mutator(source, [](type a, type b) { return b; });

            return *this;
        }
    }

    ndarray<type>& operator= (ndarray<type> source){
        if(&source == this)
            return *this;
        if(is_view) {
            element_to_element_mutator(source, [](type a, type b) { return b; });
            return *this;
        }
        else{
            dim = source.dim;
            size = source.size;
            delete [] arr;
            delete [] shape;
            delete [] stride;
            delete [] offset;
            delete [] mod;
            delete [] step;

            arr = new type[source.size];
            shape = new int[source.dim];
            stride = new int[source.dim];
            offset = new int[source.dim];
            mod = new int[source.dim];
            step = new int[source.dim];

            arr_copy(shape, source.shape, dim);

            for(int i = 0;i < dim;i++)
                stride[i] = 1;

            for(int i = 0;i < dim;i++)
                offset[i] = 0;

            this->mod[0] = 1;
            for (int i = 1; i < dim; i++)
                this->mod[i] = this->mod[i - 1] * shape[i - 1];

            this->step[dim - 1] = 1;
            for (int i = dim - 2; i >= 0; i--)
                this->step[i] = this->step[i + 1] * shape[i + 1];

            element_to_element_mutator(source, [](type a, type b) { return b; });

            return *this;
        }
    }

    ndarray<type> operator[](const int idx){
        return ndarray<type>(dim - 1,shape + 1,arr + (offset[0] + stride[0] * idx) * step[0],
                             stride + 1,offset + 1,step + 1);
    }

    ndarray<type> operator[](const int slice[3]){
        int new_shape[dim];
        arr_copy(new_shape,shape,dim);
        new_shape[0] = (slice[1] - slice[0] + slice[2] - 1)/slice[2];
        int new_stride[dim];
        arr_copy(new_stride,stride,dim);
        new_stride[0] *= slice[2];
        int new_offset[dim];
        arr_copy(new_offset,offset,dim);
        new_offset[0] += slice[0]*stride[0];
        return ndarray<type>(dim,new_shape,arr,new_stride,new_offset,step);
    }

    type& get_value(const int* idx){
        int real_idx = 0;
        for(int i = 0;i < dim;i++) {
            if(shape[i] > 1)
                real_idx += (offset[i] + idx[i] * stride[i]) * step[i];
            else
                real_idx += offset[i] * step[i];
        }
        return arr[real_idx];
    }

    ndarray<type> get_array(const int* idx,const int depth){
        int real_idx = 0;
        for(int i = 0;i < depth;i++) {
            if(shape[i] > 1)
                real_idx += (offset[i] + idx[i] * stride[i]) * step[i];
            else
                real_idx += offset[i] * step[i];
        }
        return ndarray<type>(dim - depth,shape + depth,arr + real_idx,stride + depth,offset + depth,step + depth);
    }

    ndarray<type> deep_slice(int **slice, const int depth){
        int new_shape[dim];
        arr_copy(new_shape,shape,dim);
        int new_stride[dim];
        arr_copy(new_stride,stride,dim);
        int new_offset[dim];
        arr_copy(new_offset,offset,dim);
        for(int i = 0;i < depth;i++) {
            new_shape[i] = (slice[i][1] - slice[i][0] + slice[i][2] - 1) / slice[i][2];
            new_stride[i] *= slice[i][2];
            new_offset[i] += slice[i][0] * stride[i];
        }
        return ndarray<type>(dim,new_shape,arr,new_stride,new_offset,step);
    }

    bool broadcast_check(ndarray<type>& source,const int depth){
        if(source.dim < depth or this->dim < depth)
            return false;
        else
            for(int i = 0;i < depth;i++)
                if(source.shape[i] != this->shape[i] and source.shape[i] != 1 and this->shape[i] != 1)
                    return false;
        return true;
    }

    int* broadcast_shape(ndarray<type>& source,const int depth,const int* element_shape,const int element_dim){
        int* new_shape = new int[depth + element_dim];
        for(int i = 0;i < depth;i++)
            new_shape[i] = (this->shape[i] > source.shape[i]) ? this->shape[i] : source.shape[i];
        for(int i = 0;i < element_dim;i++)
            new_shape[depth + i] = element_shape[i];
        return new_shape;
    }

    int* index_forward_iterate(int* idx,int depth){
        idx[depth - 1]++;
        for(int i = depth - 1;i >= 0;i--)
            if(idx[i] >= shape[i]) {
                if(i > 0)
                    idx[i-1] += 1;
                idx[i] = 0;
            }
        return idx;
    }

    ndarray<type> element_to_element_operation(ndarray<type>& source,type (*fptr)(type,type)){
        if (!broadcast_check(source,dim) or dim != source.dim)
            throw std::invalid_argument("element_to_element_operation");
        int* new_shape = broadcast_shape(source,dim, nullptr,0);
        ndarray<type> rt = ndarray<type>(dim,new_shape);
        delete [] new_shape;

#pragma omp parallel
        {
            int* local_index = new int[dim]{};
#pragma omp for
            for(int i = 0;i < rt.size;i++) {
                // Use helper function to calculate index
                rt.linear_to_index(i, local_index, dim);
                rt.get_value(local_index) = fptr(this->get_value(local_index),source.get_value(local_index));
            }
            delete [] local_index;
        }

        return rt;
    }

    ndarray<type> & element_to_element_mutator(ndarray<type>& source,type (*fptr)(type,type)){
        if (!broadcast_check(source,dim) or dim != source.dim)
            throw std::invalid_argument("element_to_element_operation");
        int* new_shape = broadcast_shape(source,dim, nullptr,0);
        for(int i = 0;i < dim;i++)
            if(new_shape[i] != shape[i])
                throw std::invalid_argument("element_to_element_operation");
        delete [] new_shape;
#pragma omp parallel
        {
            int* local_index = new int[dim]{};
#pragma omp for
            for(int i = 0;i < size;i++){
                // Use helper function to calculate index
                linear_to_index(i, local_index, dim);
                get_value(local_index) = fptr(this->get_value(local_index),source.get_value(local_index));
            }
            delete [] local_index;
        }
        return *this;
    }

    ndarray<type> element_operation(type (*fptr)(type)){
        ndarray<type> rt = ndarray<type>(dim,shape);
#pragma omp parallel
        {
            int* local_index = new int[dim]{};
#pragma omp for
            for(int i = 0;i < rt.size;i++){
                // Use helper function to calculate index
                linear_to_index(i, local_index, dim);
                rt.get_value(local_index) = fptr(this->get_value(local_index));
            }
            delete [] local_index;
        }

        return rt;
    }

    ndarray<type> & element_mutator(type (*fptr)(type)){
#pragma omp parallel
        {
            int* local_index = new int[dim]{};
#pragma omp for
            for(int i = 0;i < this->size;i++){
                // Use helper function to calculate index
                linear_to_index(i, local_index, dim);
                this->get_value(local_index) = fptr(this->get_value(local_index));
            }
            delete [] local_index;
        }
        return *this;
    }

    ndarray<type> array_to_array_operation(ndarray<type>& source,ndarray<type> (*fptr)(ndarray<type>,ndarray<type>),int depth,int* element_shape,int element_dim){
        if (!broadcast_check(source,depth))
            throw std::invalid_argument("array_to_array_operation");
        int* new_shape = broadcast_shape(source,depth, element_shape,element_dim);
        ndarray<type> rt = ndarray<type>(depth + element_dim,new_shape);
        delete [] new_shape;

#pragma omp parallel
        {
            int* local_index = new int[depth]{};
#pragma omp for
            for(int i = 0;i < rt.mod[depth];i++){
                // Use helper function for depth-limited index calculation
                int remaining = i;
                for(int j = depth-1; j >= 0; j--) {
                    local_index[j] = remaining % rt.shape[j];
                    remaining /= rt.shape[j];
                }

                rt.get_array(local_index,depth) = fptr(this->get_array(local_index,depth),source.get_array(local_index,depth));
            }
            delete [] local_index;
        }

        return rt;
    }

    ndarray<type> & array_to_array_mutator(ndarray<type>& source,ndarray<type> (*fptr)(ndarray<type>,ndarray<type>),int depth){
        if (!broadcast_check(source,depth))
            throw std::invalid_argument("array_to_array_operation");
        int* new_shape = broadcast_shape(source,depth);
        for(int i = 0;i < depth;i++)
            if(new_shape[i] != shape[i])
                throw std::invalid_argument("array_to_array_operation");
        delete [] new_shape;
#pragma omp parallel
        {
            int* local_index = new int[depth]{};
#pragma omp for
            for(int i = 0;i < mod[depth];i++){
                // Use helper function for depth-limited index calculation
                int remaining = i;
                for(int j = depth-1; j >= 0; j--) {
                    local_index[j] = remaining % shape[j];
                    remaining /= shape[j];
                }

                get_array(local_index,depth) = fptr(this->get_array(local_index,depth),source.get_array(local_index,depth));
            }
            delete [] local_index;
        }
        return *this;
    }

    ndarray<type> array_operation(ndarray<type> (*fptr)(ndarray<type>),int depth,int* element_shape,int element_dim){
        int new_shape[depth + element_dim];
        for(int i = 0;i < depth;i++)
            new_shape[i] = shape[i];
        for(int i = 0;i < element_dim;i++)
            new_shape[depth + i] = element_shape[i];
        ndarray<type> rt = ndarray<type>(depth + element_dim,new_shape);

#pragma omp parallel
        {
            int* local_index = new int[depth]{};
#pragma omp for
            for(int i = 0;i < rt.mod[depth];i++){
                // Use helper function for depth-limited index calculation
                int remaining = i;
                for(int j = depth-1; j >= 0; j--) {
                    local_index[j] = remaining % rt.shape[j];
                    remaining /= rt.shape[j];
                }

                rt.get_array(local_index,depth) = fptr(this->get_array(local_index,depth));
            }
            delete [] local_index;
        }

        return rt;
    }

    ndarray<type> & array_mutator(ndarray<type> (*fptr)(ndarray<type>),int depth){
#pragma omp parallel
        {
            int* local_index = new int[depth]{};
#pragma omp for
            for(int i = 0;i < mod[depth];i++){
                // Use helper function for depth-limited index calculation
                int remaining = i;
                for(int j = depth-1; j >= 0; j--) {
                    local_index[j] = remaining % shape[j];
                    remaining /= shape[j];
                }

                this->get_array(local_index,depth) = fptr(this->get_array(local_index,depth));
            }
            delete [] local_index;
        }
        return *this;
    }

    ndarray<type> operator+ (ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return a + b;});
    }

    ndarray<type>& operator+= (ndarray<type> source){
        return element_to_element_mutator(source,[](type a,type b){return a + b;});
    }

    ndarray<type> operator- (ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return a - b;});
    }

    ndarray<type>& operator-= (ndarray<type> source){
        return element_to_element_mutator(source,[](type a,type b){return a - b;});
    }

    ndarray<type> operator* (ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return a * b;});
    }

    ndarray<type>& operator*= (ndarray<type> source){
        return element_to_element_mutator(source,[](type a,type b){return a * b;});
    }

    ndarray<type> operator/ (ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return a / b;});
    }

    ndarray<type>& operator/= (ndarray<type> source){
        return element_to_element_mutator(source,[](type a,type b){return a / b;});
    }

    ndarray<type> operator% (ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return a % b;});
    }

    ndarray<type>& operator%= (ndarray<type> source){
        return element_to_element_mutator(source,[](type a,type b){return a % b;});
    }

    ndarray<type> operator|| (ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return a || b;});
    }

    ndarray<type> operator&& (ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return a && b;});
    }

    ndarray<type> max(ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return std::max(a,b);});
    }

    ndarray<type> min(ndarray<type> source){
        return element_to_element_operation(source,[](type a,type b){return std::min(a,b);});
    }

    ndarray<type> operator!(){
        return element_mutator([](type a){return !a;});
    }

    ndarray<type>& operator++(){
        return element_mutator([](type a){return ++a;});
    }

    ndarray<type>& operator--(){
        return element_mutator([](type a){return --a;});
    }

    ndarray<type> operator+ (type i){
        return element_operation([=](type a){return a + i;});
    }

    ndarray<type> operator- (type i){
        return element_operation([=](type a){return a - i;});
    }

    ndarray<type> operator* (type i){
        return element_operation([=](type a){return a * i;});
    }

    ndarray<type> operator/ (type i){
        return element_operation([=](type a){return a / i;});
    }

    ndarray<type> operator% (type i){
        return element_operation([=](type a){return a % i;});
    }

    ndarray<type> operator|| (type i){
        return element_operation([=](type a){return a || i;});
    }

    ndarray<type> operator&& (type i){
        return element_operation([=](type a){return a && i;});
    }

    ndarray<type>& operator+= (type i){
        return element_mutator([=](type a){return a + i;});
    }

    ndarray<type>& operator-= (type i){
        return element_mutator([=](type a){return a - i;});
    }

    ndarray<type>& operator*= (type i){
        return element_mutator([=](type a){return a * i;});
    }

    ndarray<type>& operator/= (type i){
        return element_mutator([=](type a){return a / i;});
    }

    ndarray<type>& operator%= (type i){
        return element_mutator([=](type a){return a % i;});
    }

    static ndarray<type> elementary_matmul(ndarray<type> a,ndarray<type> b){
        int element_shape[3]{a.shape[0],b.shape[1]};
        ndarray<type> rt = ndarray<type>(2,element_shape);

#pragma omp parallel for collapse(2)
        for(int i = 0;i < a.shape[0];i++){
            for(int j = 0;j < b.shape[1];j++){
                int rt_index[2]{i, j};
                type sum = 0;

                for(int k = 0;k < a.shape[1];k++){
                    int this_index[2]{i, k};
                    int source_index[2]{k, j};
                    sum += a.get_value(this_index) * b.get_value(source_index);
                }

                rt.get_value(rt_index) = sum;
            }
        }

        return rt;
    }

    ndarray<type> matmul (ndarray<type> source){
        int element_shape[3]{this->shape[dim-2],source.shape[dim-1]};
        return array_to_array_operation(source,elementary_matmul,dim-2,element_shape,2);
    }

    ndarray<type> axis_sum (int axis){
        int element_shape[dim - axis];
        element_shape[0] = 1;
        for(int i = 1;i < dim - axis;i++)
            element_shape[i] = shape[axis + i];
        return array_operation([](ndarray<type> a){
            int element_shape[a.dim];
            element_shape[0] = 1;
            for(int i = 1;i < a.dim;i++)
                element_shape[i] = a.shape[i];
            ndarray<type> rt = ndarray<type>(a.dim,element_shape);

#pragma omp parallel
            {
                ndarray<type> local_rt = ndarray<type>(a.dim,element_shape);
#pragma omp for nowait
                for(int i = 0;i < a.shape[0];i++) {
                    local_rt[0] = (local_rt[0] + a[i]);
                }

#pragma omp critical
                {
                    rt[0] = (rt[0] + local_rt[0]);
                }
            }

            return rt;
        },axis,element_shape,dim-axis);
    }

    ndarray<type> axis_max (int axis){
        int element_shape[dim - axis];
        element_shape[0] = 1;
        for(int i = 1;i < dim - axis;i++)
            element_shape[i] = shape[axis + i];
        return array_operation([](ndarray<type> a){
            int element_shape[a.dim];
            element_shape[0] = 1;
            for(int i = 1;i < a.dim;i++)
                element_shape[i] = a.shape[i];
            ndarray<type> rt = ndarray<type>(a.dim,element_shape);
            rt[0] = a[0];

#pragma omp parallel
            {
                ndarray<type> local_max = ndarray<type>(a.dim,element_shape);
                local_max[0] = a[0];

#pragma omp for nowait
                for(int i = 1;i < a.shape[0];i++) {
                    local_max[0] = local_max[0].max(a[i]);
                }

#pragma omp critical
                {
                    rt[0] = rt[0].max(local_max[0]);
                }
            }

            return rt;
        },axis,element_shape,dim-axis);
    }

    ndarray<type> axis_min (int axis){
        int element_shape[dim - axis];
        element_shape[0] = 1;
        for(int i = 1;i < dim - axis;i++)
            element_shape[i] = shape[axis + i];
        return array_operation([](ndarray<type> a){
            int element_shape[a.dim];
            element_shape[0] = 1;
            for(int i = 1;i < a.dim;i++)
                element_shape[i] = a.shape[i];
            ndarray<type> rt = ndarray<type>(a.dim,element_shape);
            rt[0] = a[0];

#pragma omp parallel
            {
                ndarray<type> local_min = ndarray<type>(a.dim,element_shape);
                local_min[0] = a[0];

#pragma omp for nowait
                for(int i = 1;i < a.shape[0];i++) {
                    local_min[0] = local_min[0].min(a[i]);
                }

#pragma omp critical
                {
                    rt[0] = rt[0].min(local_min[0]);
                }
            }

            return rt;
        },axis,element_shape,dim-axis);
    }

    ndarray<type> get_transpose(int* axis_order){
        int new_shape[dim];
        for(int i = 0;i < dim;i++)
            new_shape[i] = shape[axis_order[i]];
        ndarray<type> rt = ndarray<type>(dim,new_shape);
#pragma omp parallel
        {
            int* rt_index = new int[dim]{};
            int* index = new int[dim]{};
#pragma omp for
            for(int i = 0;i < rt.size;i++) {
                rt.linear_to_index(i, rt_index, dim);
                for(int j = 0;j < dim;j++)
                    index[j] = rt_index[axis_order[j]];
                rt.get_value(rt_index) = this->get_value(index);
            }
            delete [] rt_index;
            delete [] index;
        }
        return rt;
    }

    // Helper method to get the number of threads
    static int get_num_threads() {
        int num_threads = 1;
#pragma omp parallel
        {
#pragma omp single
            num_threads = omp_get_num_threads();
        }
        return num_threads;
    }

    // Method to set the number of threads
    static void set_num_threads(int n) {
        omp_set_num_threads(n);
    }

    void print(){
        std::cout << "shape = [";
        for(int i = 0;i < dim;i++)
            std::cout << shape[i] << ",";
        std::cout << "]" << std::endl;

        int* index = new int[dim]{};
        for(int i = 0;i < size;i++){
            std::cout << "[";
            for(int j = 0;j < dim;j++)
                std::cout << index[j] << ",";
            std::cout << "] = " << get_value(index) << std::endl;
            index_forward_iterate(index,dim);
        }
        delete [] index;
    }

    // Compare operator for debugging
    bool operator==(const ndarray<type>& other) const {
        if (dim != other.dim)
            return false;

        for (int i = 0; i < dim; i++)
            if (shape[i] != other.shape[i])
                return false;

        int* index = new int[dim]{};
        bool result = true;

        for (int i = 0; i < size; i++) {
            if (get_value(index) != other.get_value(index)) {
                result = false;
                break;
            }
            index_forward_iterate(index, dim);
        }

        delete[] index;
        return result;
    }
};