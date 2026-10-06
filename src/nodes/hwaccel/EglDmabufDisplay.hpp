#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "../../util.hpp"

#include <cstring>
#include <optional>

// The EGL display the DMA-BUF importers (drm_prime_to_egl_image, drm_prime_to_cuda)
// create images on. Importing a DMA-BUF (eglCreateImage with EGL_NO_CONTEXT), and
// registering the image with CUDA, need only an initialized display: no GL context
// or pbuffer is created, as each would cost device memory per node.
struct EglDmabufDisplay {
    EGLDisplay dpy = EGL_NO_DISPLAY;
    bool have_modifiers = false;   // EGL_EXT_image_dma_buf_import_modifiers
    PFNEGLCREATEIMAGEKHRPROC create = nullptr;
    PFNEGLDESTROYIMAGEKHRPROC destroy = nullptr;

    /// Initializes the default display for DMA-BUF import. On failure logs why, prefixed
    /// with `node`, and returns nothing; the importer tries again on its next frame.
    static std::optional<EglDmabufDisplay> open(const char *node) {
        EglDmabufDisplay egl;
        egl.dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (egl.dpy == EGL_NO_DISPLAY) {
            logstream << node << ": eglGetDisplay failed";
            return std::nullopt;
        }
        EGLint major = 0, minor = 0;
        if (!eglInitialize(egl.dpy, &major, &minor)) {
            logstream << node << ": eglInitialize failed";
            return std::nullopt;
        }
        logstream << node << ": EGL vendor: " << safeString(eglQueryString(egl.dpy, EGL_VENDOR));
        if (!extensionSupported(egl.dpy, "EGL_EXT_image_dma_buf_import")) {
            logstream << node << ": EGL_EXT_image_dma_buf_import missing, exts="
                      << safeString(eglQueryString(egl.dpy, EGL_EXTENSIONS));
            return std::nullopt;
        }
        egl.have_modifiers = extensionSupported(egl.dpy, "EGL_EXT_image_dma_buf_import_modifiers");
        // Resolved at runtime to avoid link-time dependencies; some implementations
        // export only the core names.
        egl.create = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImageKHR"));
        if (!egl.create)
            egl.create = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImage"));
        egl.destroy = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(eglGetProcAddress("eglDestroyImageKHR"));
        if (!egl.destroy)
            egl.destroy = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(eglGetProcAddress("eglDestroyImage"));
        if (!egl.create || !egl.destroy) {
            logstream << node << ": failed to load eglCreateImage/eglDestroyImage";
            return std::nullopt;
        }
        return egl;
    }

private:
    static const char *safeString(const char *value) { return value ? value : ""; }

    // A whole token of the space-separated list: a prefix of a longer name does not count.
    static bool extensionSupported(EGLDisplay display, const char *extension) {
        const char *extensions = eglQueryString(display, EGL_EXTENSIONS);
        if (!extensions || !extension || !extension[0])
            return false;
        const size_t len = std::strlen(extension);
        const char *current = extensions;
        while ((current = std::strstr(current, extension)) != nullptr) {
            const bool starts = current == extensions || current[-1] == ' ';
            const bool ends = current[len] == '\0' || current[len] == ' ';
            if (starts && ends)
                return true;
            current += len;
        }
        return false;
    }
};
