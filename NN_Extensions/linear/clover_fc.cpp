#include <CloverMatrix4.h>

#include <CloverVector32.h>

extern "C"
{

float dot_4(const int n, const float* array1, const float* array2)
{

    CloverVector32 a_vector_32bit(n);
    CloverVector32 b_vector_32bit(n);


    float* a = a_vector_32bit.getData();
    float* b = b_vector_32bit.getData();
    for (int i = 0; i < n; i += 1) {
        a[i] = array1[i];       // a_vector_32bit.set(i, arr1[i]);
        b[i] = array2[i];       // b_vector_32bit.set(i, arr2[i]);
    }

    CloverVector4 a_vector_4bit(n);
    CloverVector4 b_vector_4bit(n);

    a_vector_4bit.quantize(a_vector_32bit);
    b_vector_4bit.quantize(b_vector_32bit);

    float dot = a_vector_4bit.dot(b_vector_4bit);
//    std::cout << "The dot product is: " << dot << std::endl;

    return dot;
}



}

