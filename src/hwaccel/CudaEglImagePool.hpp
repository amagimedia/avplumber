#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
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

#ifndef GL_SRGB8_ALPHA8
#define GL_SRGB8_ALPHA8 0x8C43
#endif

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
		if (egl_context_ != EGL_NO_CONTEXT) {
			// Context binding is thread-local. After node restart the worker thread can
			// change, so rebind the existing EGL context on each ensure call.
			if (!eglBindAPI(EGL_OPENGL_API)) return false;
			if (egl_display_ == EGL_NO_DISPLAY || egl_surface_ == EGL_NO_SURFACE) return false;
			if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) return false;
			return true;
		}
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
	bool growLocked(int add, CUcontext cu_ctx) {
		if (add <= 0) return true;
		// Don't mix different CUDA contexts in the same pool: cuGraphicsGLRegisterImage is context-sensitive.
		if (registered_cu_ctx_ && cu_ctx && registered_cu_ctx_ != cu_ctx) {
			return false;
		}
		CUcontext effective_ctx = registered_cu_ctx_ ? registered_cu_ctx_ : cu_ctx;
		const int w = width_;
		const int h = height_;
		if (w <= 0 || h <= 0) return false;

		auto destroy_ctx = effective_ctx;
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
			glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			glBindTexture(GL_TEXTURE_2D, 0);
			EGLint img_attrs[] = { EGL_GL_TEXTURE_LEVEL_KHR, 0, /*EGL_GL_COLORSPACE_KHR, EGL_GL_COLORSPACE_SRGB_KHR,*/ EGL_NONE };
			e.egl_image = egl_create_image_khr_(egl_display_, egl_context_, EGL_GL_TEXTURE_2D_KHR, (EGLClientBuffer)(uintptr_t)e.gl_tex_rgba, img_attrs);
			if (e.egl_image == EGL_NO_IMAGE_KHR) return false;
			if (effective_ctx) {
				if (cuCtxPushCurrent) cuCtxPushCurrent(effective_ctx);
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
		if (!pool_.grow((size_t)add, create, destroy)) {
			return false;
		}
		registered_cu_ctx_ = effective_ctx;
		return true;
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
			// Allocate as sRGB so OBS can do a single sRGB->linear decode on sampling.
			glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			glBindTexture(GL_TEXTURE_2D, 0);
			EGLint img_attrs[] = { EGL_GL_TEXTURE_LEVEL_KHR, 0, /*EGL_GL_COLORSPACE_KHR, EGL_GL_COLORSPACE_SRGB_KHR,*/ EGL_NONE };
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
	size_t size() const { return pool_.size(); }
	bool initializedFor(int w, int h) const {
		return pool_.size() > 0 && width_ == w && height_ == h;
	}
	bool ensureInitialized(int w, int h, int pool_size, CUcontext cu_ctx) {
		std::lock_guard<std::mutex> lock(egl_mtx_);
		if (!ensureEglContextLocked()) return false;
		if (pool_size < 0) pool_size = 0;

		// If dimensions changed (or pool uninitialized), we must reinit.
		if (!initializedFor(w, h)) {
			return reinitLocked(w, h, pool_size, cu_ctx);
		}
		// CUDA-GL interop registrations are context-specific; when pipeline restart yields
		// a different CUDA context, recreate registrations for the new context.
		if (registered_cu_ctx_ && cu_ctx && registered_cu_ctx_ != cu_ctx) {
			const int keep_size = std::max((int)pool_.size(), pool_size);
			return reinitLocked(w, h, keep_size, cu_ctx);
		}

		// IMPORTANT: never shrink/reinit just because someone requested a different pool_size.
		// This pool is typically shared across multiple producers via pool_id, and outstanding
		// EglImageFrames may still be held by downstream consumers (e.g. OBS). Resetting would
		// destroy EGLImages/GL textures/CUDA registrations while they're still in use → flicker.
		const int cur_size = (int)pool_.size();
		if (cur_size >= pool_size) {
			return true;
		}

		// Grow up to requested size (append-only).
		return growLocked(pool_size - cur_size, cu_ctx);
	}
	// Tries to acquire a free entry. If none are available and the pool is below max_pool_size,
	// it grows the pool by up to grow_step entries (without destroying existing entries) and retries.
	std::optional<size_t> acquireOrGrow(int w, int h, int max_pool_size, int grow_step, CUcontext cu_ctx) {
		// Fast path: try acquire without EGL lock.
		if (auto idx = pool_.acquire()) return idx;

		if (max_pool_size <= 0) return std::nullopt;
		if (grow_step <= 0) grow_step = 1;

		std::lock_guard<std::mutex> lock(egl_mtx_);
		if (!ensureEglContextLocked()) return std::nullopt;
		if (!initializedFor(w, h)) return std::nullopt;

		const int cur_size = (int)pool_.size();
		if (cur_size >= max_pool_size) return std::nullopt;

		const int add = std::min(grow_step, max_pool_size - cur_size);
		if (!growLocked(add, cu_ctx)) return std::nullopt;

		return pool_.acquire();
	}
	std::optional<size_t> acquire() { return pool_.acquire(); }
	void release(size_t idx) { pool_.release(idx); }
	CudaEglImageEntry& entry(size_t idx) { return pool_.entry(idx); }
	int width() const { return width_; }
	int height() const { return height_; }
};


