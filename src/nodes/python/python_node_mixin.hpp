#pragma once
#ifdef PYTHON_MODULE
#include "../../graph_interfaces.hpp"
#include "../../util.hpp"
#include <pybind11/gil.h>
#include <exception>

namespace py = pybind11;

// Shared plumbing for every python_node_* node: holds the Python-side handle
// and wraps the GIL-guarded calls into it. Concrete nodes inherit this in
// addition to their input/output Node base, and only implement what differs
// between topologies (stop()'s parent call, onEofConsumed()'s output handling,
// create()'s edge wiring).
//
// Inherit this AFTER the Node base so ~PythonNodeMixin (which nulls the handle
// under the GIL) runs before the Node bases are torn down.
class PythonNodeMixin: public IPythonNode {
protected:
    py::object python_node_;
    bool stopped_ = false;

    py::object& requirePythonNode() {
        if (python_node_.ptr() == nullptr || python_node_.is_none()) {
            throw Error("Python node is not set");
        }
        return python_node_;
    }

    void callProcess() {
        py::gil_scoped_acquire gil;
        requirePythonNode().attr("process")();
    }

    // Call doStop() on the Python node exactly once. Safe to invoke from both
    // stop() and onEofConsumed(); acquires the GIL itself (nesting is allowed).
    void callDoStopOnce() {
        py::gil_scoped_acquire gil;
        if (stopped_) {
            return;
        }
        stopped_ = true;
        requirePythonNode().attr("doStop")();
    }

    // Run callDoStopOnce(), capturing any exception so the caller can finish
    // the node's own stop sequence before rethrowing.
    std::exception_ptr captureStop() {
        try {
            callDoStopOnce();
        } catch (...) {
            return std::current_exception();
        }
        return nullptr;
    }

    // Call an optional Python lifecycle hook (e.g. doStart, flush_open_segment)
    // if the node defines it. A missing handle or method is a no-op.
    void callOptional(const char* attr_name) {
        py::gil_scoped_acquire gil;
        if (python_node_.ptr() == nullptr || python_node_.is_none()) {
            return;
        }
        if (py::hasattr(python_node_, attr_name)) {
            python_node_.attr(attr_name)();
        }
    }

public:
    void set_python_node(py::object python_node) override {
        py::gil_scoped_acquire gil;
        python_node_ = std::move(python_node);
    }

    ~PythonNodeMixin() override {
        py::gil_scoped_acquire gil;
        // Leave a null handle so member destruction after GIL release is safe.
        python_node_ = py::object();
    }
};
#endif
