#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


/*
xx : (m * n)
yy : (n * k)
z : (m * k)
*/
void mat_mul(const float *xx, const float *yy,
								  float *z,  size_t m, size_t n, size_t k)
{
    for(int i = 0; i < m; i++){
        for(int j = 0; j < k; j++){
            z[i*k + j] = 0;
            for(int t = 0 ; t < n; t++){
                z[i*k + j] += xx[i*n+t]*yy[t*k+j];
            }
        }
    }
}
                                


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int iterations = (m + batch - 1) / batch;
    for(int it = 0; it < iterations; it++){
        const float* xx = &X[it*batch*n]; // (batch x n)
        const unsigned char* yy = &y[it*batch]; // (batch x 1)
        float * z = new float[batch*k]; // (batch x n) (n x k) = (batch x k)
        mat_mul(xx, theta, z, batch, n, k);
        for(int i = 0; i < batch*k; i++){
            z[i] = exp(z[i]);
        }
        for(int i = 0; i < batch; i++){
            float sum = 0.0;
            for(int j = 0; j < k; j++){
                sum += z[i*k+j];
            }
            for(int j = 0; j < k; j++){
                z[i*k + j] /= sum;
            }
        }
        for(int i = 0; i < batch; i++){
            z[i*k + yy[i]] -= 1;
        }
        float * xx_T = new float[n*batch];
        for(int i = 0; i < batch; i++){
            for(int j = 0; j < n; j++){
                xx_T[j*batch + i] = xx[i*n+j];
            }
        }
        // (n x batch) * (batch x k) = (n x k) 
        float * grad = new float[n*k];
        mat_mul(xx_T, z, grad, n, batch, k);
        for(int i = 0; i < n * k; i++){
            theta[i] -= lr*grad[i]/batch;
        }
        delete[] z;
        delete[] xx_T;
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
