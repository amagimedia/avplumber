#pragma once
#include <vector>
#include <memory>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <libavutil/hwcontext.h>
#include "../instance_shared.hpp"
#include "../ObjectPool.hpp"
#
// CUDA driver API (dynlink) - included where used
#include "../../deps/cuda_loader/cuda_drvapi_dynlink_cuda.h"
#include "../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"
#
struct CudaEglImageEntry {
	GLuint gl_tex_rgba = 0;
	EGLImageKHR egl_image = EGL_NO_IMAGE_KHR;
	CUgraphicsResource cu_tex_res = nullptr;
};
#
class CudaEglImagePool {
private:
	ObjectPool<CudaEglImageEntry> pool_;
	int width_ = 0;
	int height_ = 0;
public:
	bool initializedFor(int w, int h) const {
		return pool_.size() > 0 && width_ == w && height_ == h;
	}
	// Reinitialize the pool for given size and count. EGL context must be current.
	bool reinit(int w, int h,
	            int pool_size,
	            EGLDisplay egl_display,
	            EGLContext egl_context,
	            PFNEGLCREATEIMAGEKHRPROC EGLCreateImageKHR_,
	            CUcontext cu_ctx) {
		width_ = 0;
		height_ = 0;
		auto create = [=](CudaEglImageEntry &e)->bool {
			glGenTextures(1, &e.gl_tex_rgba);
			if (!e.gl_tex_rgba) return false;
			glBindTexture(GL_TEXTURE_2D, e.gl_tex_rgba);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			glBindTexture(GL_TEXTURE_2D, 0);
			EGLint img_attrs[] = { EGL_GL_TEXTURE_LEVEL_KHR, 0, EGL_NONE };
			e.egl_image = EGLCreateImageKHR_(egl_display, egl_context, EGL_GL_TEXTURE_2D_KHR, (EGLClientBuffer)(uintptr_t)e.gl_tex_rgba, img_attrs);
			if (e.egl_image == EGL_NO_IMAGE_KHR) return false;
			// CUDA register
			if (cu_ctx) {
				if (cuCtxPushCurrent) cuCtxPushCurrent(cu_ctx);
				int err = cuGraphicsGLRegisterImage(&e.cu_tex_res, e.gl_tex_rgba, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD | CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
				if (cuCtxPopCurrent) { CUcontext dummy; cuCtxPopCurrent(&dummy); }
				if (err != CUDA_SUCCESS) return false;
			}
			return true;
		};
		auto destroy = [=](CudaEglImageEntry &e) {
			if (e.cu_tex_res) {
				cuGraphicsUnregisterResource(e.cu_tex_res);
				e.cu_tex_res = nullptr;
			}
			if (e.egl_image != EGL_NO_IMAGE_KHR && egl_display != EGL_NO_DISPLAY) {
				eglDestroyImageKHR(egl_display, e.egl_image);
				e.egl_image = EGL_NO_IMAGE_KHR;
			}
			if (e.gl_tex_rgba) {
				glDeleteTextures(1, &e.gl_tex_rgba);
				e.gl_tex_rgba = 0;
			}
		};
		if (!pool_.reset(pool_size, create, destroy)) return false;
		width_ = w;
		height_ = h;
		return true;
	}
	std::optional<size_t> acquire() { return pool_.acquire(); }
	void release(size_t idx) { pool_.release(idx); }
	CudaEglImageEntry& entry(size_t idx) { return pool_.entry(idx); }
	int width() const { return width_; }
	int height() const { return height_; }
};


