#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <libavutil/hwcontext.h>
#include "../instance_shared.hpp"
#include "../ObjectPool.hpp"

// CUDA driver API (dynlink) - included where used
#include "../../deps/cuda_loader/cuda_drvapi_dynlink_cuda.h"
#include "../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"

struct CudaEglImageEntry {
	GLuint gl_tex_rgba = 0;
	EGLImageKHR egl_image = EGL_NO_IMAGE_KHR;
	CUgraphicsResource cu_tex_res = nullptr;
};

class CudaEglImagePool {
private:
	ObjectPool<CudaEglImageEntry> pool_;
	int width_ = 0;
	int height_ = 0;
	std::mutex egl_mtx_;
	EGLDisplay egl_display_ = EGL_NO_DISPLAY;
	EGLContext egl_context_ = EGL_NO_CONTEXT;
	EGLSurface egl_surface_ = EGL_NO_SURFACE;
	EGLConfig egl_config_ = nullptr;
	PFNEGLCREATEIMAGEKHRPROC egl_create_image_khr_ = nullptr;
	PFNEGLDESTROYIMAGEKHRPROC egl_destroy_image_khr_ = nullptr;
	CUcontext registered_cu_ctx_ = nullptr;

	bool ensureEglContextLocked() {
		if (egl_context_ != EGL_NO_CONTEXT) return true;
		egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
		if (egl_display_ == EGL_NO_DISPLAY) return false;
		EGLint major, minor;
		if (!eglInitialize(egl_display_, &major, &minor)) {
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		if (!eglBindAPI(EGL_OPENGL_API)) {
			//eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		EGLint config_attrs[] = {
			EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
			EGL_SURFACE_TYPE,   EGL_PBUFFER_BIT,
			EGL_RED_SIZE,       8,
			EGL_GREEN_SIZE,     8,
			EGL_BLUE_SIZE,      8,
			EGL_ALPHA_SIZE,     8,
			EGL_NONE
		};
		EGLint num_configs;
		if (!eglChooseConfig(egl_display_, config_attrs, &egl_config_, 1, &num_configs) || num_configs == 0) {
			//eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		EGLint ctx_attr[] = {
			#ifdef EGL_CONTEXT_MAJOR_VERSION
			EGL_CONTEXT_MAJOR_VERSION, 3,
			EGL_CONTEXT_MINOR_VERSION, 3,
			#endif
			EGL_NONE
		};
		egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, ctx_attr);
		if (egl_context_ == EGL_NO_CONTEXT) {
			//eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		EGLint pbuf_attr[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
		egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, pbuf_attr);
		if (egl_surface_ == EGL_NO_SURFACE) {
			eglDestroyContext(egl_display_, egl_context_);
			egl_context_ = EGL_NO_CONTEXT;
			//eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
			eglDestroySurface(egl_display_, egl_surface_);
			egl_surface_ = EGL_NO_SURFACE;
			eglDestroyContext(egl_display_, egl_context_);
			egl_context_ = EGL_NO_CONTEXT;
			//eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		egl_create_image_khr_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
		egl_destroy_image_khr_ = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
		if (!egl_create_image_khr_ || !egl_destroy_image_khr_) {
			eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
			eglDestroySurface(egl_display_, egl_surface_);
			egl_surface_ = EGL_NO_SURFACE;
			eglDestroyContext(egl_display_, egl_context_);
			egl_context_ = EGL_NO_CONTEXT;
			//eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		return true;
	}
	void destroyEglContextLocked() {
		if (egl_context_ != EGL_NO_CONTEXT) {
			eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
			eglDestroyContext(egl_display_, egl_context_);
			egl_context_ = EGL_NO_CONTEXT;
		}
		if (egl_surface_ != EGL_NO_SURFACE) {
			eglDestroySurface(egl_display_, egl_surface_);
			egl_surface_ = EGL_NO_SURFACE;
		}
		if (egl_display_ != EGL_NO_DISPLAY) {
			//eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
		}
		egl_config_ = nullptr;
		egl_create_image_khr_ = nullptr;
		egl_destroy_image_khr_ = nullptr;
	}
	void destroyEntriesLocked() {
		if (pool_.size() == 0) {
			width_ = 0;
			height_ = 0;
			return;
		}
		if (egl_display_ != EGL_NO_DISPLAY && egl_context_ != EGL_NO_CONTEXT && egl_surface_ != EGL_NO_SURFACE) {
			eglBindAPI(EGL_OPENGL_API);
			eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_);
		}
		auto create_noop = [](CudaEglImageEntry &) { return true; };
		CUcontext destroy_ctx = registered_cu_ctx_;
		auto destroy = [&](CudaEglImageEntry &e) {
			if (e.cu_tex_res) {
				if (destroy_ctx && cuCtxPushCurrent) {
					cuCtxPushCurrent(destroy_ctx);
				}
				cuGraphicsUnregisterResource(e.cu_tex_res);
				if (destroy_ctx && cuCtxPopCurrent) {
					CUcontext dummy;
					cuCtxPopCurrent(&dummy);
				}
				e.cu_tex_res = nullptr;
			}
			if (e.egl_image != EGL_NO_IMAGE_KHR && egl_display_ != EGL_NO_DISPLAY) {
				if (egl_destroy_image_khr_) {
					egl_destroy_image_khr_(egl_display_, e.egl_image);
				}
				e.egl_image = EGL_NO_IMAGE_KHR;
			}
			if (e.gl_tex_rgba) {
				glDeleteTextures(1, &e.gl_tex_rgba);
				e.gl_tex_rgba = 0;
			}
		};
		pool_.reset(0, create_noop, destroy);
		width_ = 0;
		height_ = 0;
	}
	bool reinitLocked(int w, int h, int pool_size, CUcontext cu_ctx) {
		auto destroy_ctx = registered_cu_ctx_;
		auto destroy = [&](CudaEglImageEntry &e) {
			if (e.cu_tex_res) {
				if (destroy_ctx && cuCtxPushCurrent) {
					cuCtxPushCurrent(destroy_ctx);
				}
				cuGraphicsUnregisterResource(e.cu_tex_res);
				if (destroy_ctx && cuCtxPopCurrent) {
					CUcontext dummy;
					cuCtxPopCurrent(&dummy);
				}
				e.cu_tex_res = nullptr;
			}
			if (e.egl_image != EGL_NO_IMAGE_KHR && egl_display_ != EGL_NO_DISPLAY) {
				if (egl_destroy_image_khr_) {
					egl_destroy_image_khr_(egl_display_, e.egl_image);
				}
				e.egl_image = EGL_NO_IMAGE_KHR;
			}
			if (e.gl_tex_rgba) {
				glDeleteTextures(1, &e.gl_tex_rgba);
				e.gl_tex_rgba = 0;
			}
		};
		auto create = [&](CudaEglImageEntry &e)->bool {
			glGenTextures(1, &e.gl_tex_rgba);
			if (!e.gl_tex_rgba) return false;
			glBindTexture(GL_TEXTURE_2D, e.gl_tex_rgba);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			// Back EGLImage with an sRGB-capable texture so OBS can decode to linear when sampling
			glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			glBindTexture(GL_TEXTURE_2D, 0);
			EGLint img_attrs[] = { EGL_GL_TEXTURE_LEVEL_KHR, 0, EGL_NONE };
			e.egl_image = egl_create_image_khr_(egl_display_, egl_context_, EGL_GL_TEXTURE_2D_KHR, (EGLClientBuffer)(uintptr_t)e.gl_tex_rgba, img_attrs);
			if (e.egl_image == EGL_NO_IMAGE_KHR) return false;
			if (cu_ctx) {
				if (cuCtxPushCurrent) cuCtxPushCurrent(cu_ctx);
				int err = cuGraphicsGLRegisterImage(&e.cu_tex_res, e.gl_tex_rgba, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD | CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
				if (cuCtxPopCurrent) { CUcontext dummy; cuCtxPopCurrent(&dummy); }
				if (err != CUDA_SUCCESS) return false;
			}
			return true;
		};
		eglBindAPI(EGL_OPENGL_API);
		if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
			return false;
		}
		if (!pool_.reset(pool_size, create, destroy)) {
			return false;
		}
		width_ = w;
		height_ = h;
		registered_cu_ctx_ = cu_ctx;
		return true;
	}
public:
	~CudaEglImagePool() {
		std::lock_guard<std::mutex> lock(egl_mtx_);
		destroyEntriesLocked();
		destroyEglContextLocked();
	}
	bool initializedFor(int w, int h) const {
		return pool_.size() > 0 && width_ == w && height_ == h;
	}
	bool ensureInitialized(int w, int h, int pool_size, CUcontext cu_ctx) {
		std::lock_guard<std::mutex> lock(egl_mtx_);
		if (!ensureEglContextLocked()) return false;
		if (initializedFor(w, h) && (int)pool_.size() == pool_size) {
			return true;
		}
		return reinitLocked(w, h, pool_size, cu_ctx);
	}
	std::optional<size_t> acquire() { return pool_.acquire(); }
	void release(size_t idx) { pool_.release(idx); }
	CudaEglImageEntry& entry(size_t idx) { return pool_.entry(idx); }
	int width() const { return width_; }
	int height() const { return height_; }
};


