#include "tensor.h"
#include "layers.h"
#include "activations.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <sstream>
#include <vector>
#include <string>
#include <cstring>

namespace py = pybind11;

namespace {

Tensor make_tensor(const std::vector<size_t>& shape, Device device) {
    return Tensor(shape, device);
}

Tensor tensor_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> array,
                         Device device) {
    py::buffer_info info = array.request();
    std::vector<size_t> shape(info.ndim);
    for (py::ssize_t i = 0; i < info.ndim; ++i) {
        shape[i] = static_cast<size_t>(info.shape[i]);
    }
    Tensor t(shape, device);
    size_t total = static_cast<size_t>(info.size);
    std::vector<float> data(total);
    std::memcpy(data.data(), info.ptr, sizeof(float) * total);
    t.fill_with(data);
    return t;
}

py::array_t<float> tensor_to_numpy(const Tensor& t) {
    auto shape = t.shape();
    std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
    if (py_shape.empty()) {
        py_shape.push_back(0);
    }
    py::array_t<float> array(py_shape);
    auto flat = t.to_vector();
    std::memcpy(array.mutable_data(), flat.data(), sizeof(float) * flat.size());
    return array;
}

Tensor relu_forward_py(const Tensor& x) {
    Tensor y(x.shape(), x.device());
    ReLU_forward(x, y);
    return y;
}

Tensor relu_backward_py(const Tensor& x, const Tensor& dy) {
    Tensor dx(x.shape(), x.device());
    ReLU_backward(x, dy, dx);
    return dx;
}

Tensor sigmoid_forward_py(const Tensor& x) {
    Tensor y(x.shape(), x.device());
    Sigmoid_forward(x, y);
    return y;
}

Tensor sigmoid_backward_py(const Tensor& y, const Tensor& dy) {
    Tensor dx(y.shape(), y.device());
    Sigmoid_backward(y, dy, dx);
    return dx;
}

Tensor linear_forward_py(const Tensor& X, const Tensor& W, const Tensor& b) {
    std::vector<size_t> y_shape = {X.shape()[0], W.shape()[0]};
    Tensor Y(y_shape, X.device());
    Linear_forward(X, W, b, Y);
    return Y;
}

py::tuple linear_backward_py(const Tensor& X, const Tensor& W, const Tensor& dY) {
    std::vector<size_t> dX_shape = {X.shape()[0], X.shape()[1]};
    std::vector<size_t> dW_shape = {W.shape()[0], W.shape()[1]};
    std::vector<size_t> db_shape = {W.shape()[0]};
    Tensor dX(dX_shape, X.device());
    Tensor dW(dW_shape, W.device());
    Tensor db(db_shape, W.device());
    Linear_backward(X, W, dY, dX, dW, db);
    return py::make_tuple(dX, dW, db);
}

Tensor conv2d_forward_py(const Tensor& X, const Tensor& W, const Tensor& b) {
    auto x_shape = X.shape();
    auto w_shape = W.shape();
    std::vector<size_t> y_shape = {x_shape[0], w_shape[0], x_shape[2], x_shape[3]};
    Tensor Y(y_shape, X.device());
    Conv2d_forward(X, W, b, Y);
    return Y;
}

py::tuple conv2d_backward_py(const Tensor& X, const Tensor& W, const Tensor& dY) {
    auto x_shape = X.shape();
    auto w_shape = W.shape();

    Tensor dX(x_shape, X.device());
    Tensor dW(w_shape, W.device());
    Tensor db({w_shape[0]}, W.device());

    Conv2d_backward(X, W, dY, dX, dW, db);
    return py::make_tuple(dX, dW, db);
}

Tensor maxpool_forward_py(const Tensor& X) {
    auto shape = X.shape();
    std::vector<size_t> y_shape = {shape[0], shape[1], shape[2] / 2, shape[3] / 2};
    Tensor Y(y_shape, X.device());
    MaxPool2d_forward(X, Y);
    return Y;
}

Tensor maxpool_backward_py(const Tensor& X, const Tensor& Y, const Tensor& dY) {
    Tensor dX(X.shape(), X.device());
    MaxPool2d_backward(X, Y, dY, dX);
    return dX;
}

Tensor softmax_forward_py(const Tensor& X) {
    Tensor Y(X.shape(), Device::CPU);
    Softmax_forward(X.cpu(), Y);
    if (X.device() == Device::GPU) {
        return Y.gpu();
    }
    return Y;
}

float cross_entropy_forward_py(const Tensor& probs, const Tensor& labels) {
    return CrossEntropyLoss_forward(probs.cpu(), labels.cpu());
}

Tensor cross_entropy_backward_py(const Tensor& probs, const Tensor& labels) {
    Tensor dLogits(probs.shape(), probs.device());
    CrossEntropyLoss_backward(probs, labels, dLogits);
    return dLogits;
}

std::string tensor_repr(const Tensor& t) {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    auto shape = t.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) oss << ", ";
    }
    oss << "], device=" << (t.device() == Device::CPU ? "CPU" : "GPU")
        << ", size=" << t.size() << ")";
    return oss.str();
}

}  // namespace

PYBIND11_MODULE(_C, m) {
    m.doc() = "Pybind11 bridge for custom Tensor and NN modules";

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](const std::vector<size_t>& shape, Device device) {
            return make_tensor(shape, device);
        }), py::arg("shape"), py::arg("device") = Device::CPU)
        .def("shape", &Tensor::shape)
        .def("size", &Tensor::size)
        .def("device", &Tensor::device)
        .def("zero_", &Tensor::zero_)
        .def("fill_with", &Tensor::fill_with)
        .def("to_vector", &Tensor::to_vector)
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("__repr__", &tensor_repr);

    m.def("tensor_from_numpy", &tensor_from_numpy, py::arg("array"),
          py::arg("device") = Device::CPU, py::return_value_policy::move);
    m.def("tensor_to_numpy", &tensor_to_numpy, py::arg("tensor"));

    m.def("relu_forward", &relu_forward_py, py::arg("x"), py::return_value_policy::move);
    m.def("relu_backward", &relu_backward_py, py::arg("x"), py::arg("grad_output"),
          py::return_value_policy::move);

    m.def("sigmoid_forward", &sigmoid_forward_py, py::arg("x"), py::return_value_policy::move);
    m.def("sigmoid_backward", &sigmoid_backward_py, py::arg("y"), py::arg("grad_output"),
          py::return_value_policy::move);

    m.def("linear_forward", &linear_forward_py, py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::return_value_policy::move);
    m.def("linear_backward", &linear_backward_py, py::arg("x"), py::arg("weight"), py::arg("grad_output"));

    m.def("conv2d_forward", &conv2d_forward_py, py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::return_value_policy::move);
    m.def("conv2d_backward", &conv2d_backward_py, py::arg("x"), py::arg("weight"), py::arg("grad_output"));

    m.def("maxpool2d_forward", &maxpool_forward_py, py::arg("x"), py::return_value_policy::move);
    m.def("maxpool2d_backward", &maxpool_backward_py, py::arg("x"), py::arg("y"), py::arg("grad_output"),
          py::return_value_policy::move);

    m.def("softmax_forward", &softmax_forward_py, py::arg("x"), py::return_value_policy::move);
    m.def("cross_entropy_forward", &cross_entropy_forward_py, py::arg("probs"), py::arg("labels"));
    m.def("cross_entropy_backward", &cross_entropy_backward_py, py::arg("probs"), py::arg("labels"),
          py::return_value_policy::move);
}

