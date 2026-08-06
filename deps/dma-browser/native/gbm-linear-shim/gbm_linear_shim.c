#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct gbm_bo;
struct gbm_device;
struct gbm_surface;

#ifndef GBM_BO_USE_SCANOUT
#define GBM_BO_USE_SCANOUT (1u << 0)
#endif

#ifndef GBM_BO_USE_CURSOR
#define GBM_BO_USE_CURSOR (1u << 1)
#endif

#ifndef GBM_BO_USE_CURSOR_64X64
#define GBM_BO_USE_CURSOR_64X64 GBM_BO_USE_CURSOR
#endif

#ifndef GBM_BO_USE_RENDERING
#define GBM_BO_USE_RENDERING (1u << 2)
#endif

#ifndef GBM_BO_USE_WRITE
#define GBM_BO_USE_WRITE (1u << 3)
#endif

#ifndef GBM_BO_USE_LINEAR
#define GBM_BO_USE_LINEAR (1u << 4)
#endif

#ifndef GBM_BO_USE_TEXTURING
#define GBM_BO_USE_TEXTURING (1u << 5)
#endif

#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR 0ULL
#endif

typedef struct gbm_bo* (*gbm_bo_create_fn)(struct gbm_device* gbm,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t format,
                                           uint32_t flags);
typedef struct gbm_bo* (*gbm_bo_create_with_modifiers_fn)(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    const unsigned int count);
typedef struct gbm_bo* (*gbm_bo_create_with_modifiers2_fn)(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    const unsigned int count,
    uint32_t flags);
typedef struct gbm_surface* (*gbm_surface_create_fn)(struct gbm_device* gbm,
                                                     uint32_t width,
                                                     uint32_t height,
                                                     uint32_t format,
                                                     uint32_t flags);
typedef struct gbm_surface* (*gbm_surface_create_with_modifiers_fn)(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    const unsigned int count);
typedef struct gbm_surface* (*gbm_surface_create_with_modifiers2_fn)(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    const unsigned int count,
    uint32_t flags);
typedef void* (*gbm_bo_map_fn)(struct gbm_bo* bo,
                               uint32_t x,
                               uint32_t y,
                               uint32_t width,
                               uint32_t height,
                               uint32_t flags,
                               uint32_t* stride,
                               void** map_data);

static bool env_enabled(const char* name, bool default_value) {
  const char* value = getenv(name);
  if (!value || !*value) {
    return default_value;
  }
  if (!strcmp(value, "0") || !strcasecmp(value, "false") ||
      !strcasecmp(value, "no") || !strcasecmp(value, "off")) {
    return false;
  }
  return true;
}

static bool shim_enabled(void) {
  return env_enabled("GBM_LINEAR_SHIM", true);
}

static bool log_enabled(void) {
  return env_enabled("GBM_LINEAR_SHIM_LOG", false);
}

static bool strip_linear_modifier_enabled(void) {
  return env_enabled("GBM_LINEAR_SHIM_STRIP_LINEAR_MODIFIER", false);
}

// Force GBM_BO_USE_SCANOUT onto buffer allocations. This replicates, at the
// GBM layer, what the custom Chromium patch
// (RenderableMappableSharedImageForceScanout) does at the SharedImage layer:
// make the offscreen/renderable buffer scanout-capable so NVIDIA exports it as
// a usable renderable dmabuf. Needed for offscreen useSharedTexture capture on
// a stock (unpatched) Chromium.
static bool add_scanout_enabled(void) {
  return env_enabled("GBM_LINEAR_SHIM_ADD_SCANOUT", false);
}

// Force a CPU-linear *and* renderable buffer: add GBM_BO_USE_LINEAR (so NVIDIA
// lays the pixels out linearly / CPU-mappable) together with
// GBM_BO_USE_RENDERING (so stock Chromium's renderable-mappable SharedImage can
// still use it as a render target and SkSurface init succeeds), and pin any
// modifier list to [DRM_FORMAT_MOD_LINEAR]. This is what a downstream consumer
// using a plain DRM hwdownload (no CUDA/EGL detiling) needs: a scanout-capable
// but *tiled* buffer (see add_scanout_enabled) captures fine but maps to
// sheared garbage. Takes precedence over GBM_LINEAR_SHIM_ADD_SCANOUT.
static bool force_linear_enabled(void) {
  return env_enabled("GBM_LINEAR_SHIM_FORCE_LINEAR", false);
}

static void* load_next(const char* symbol) {
  void* fn = dlsym(RTLD_NEXT, symbol);
  if (!fn) {
    fprintf(stderr, "[gbm-linear-shim] missing RTLD_NEXT symbol: %s: %s\n",
            symbol, dlerror());
  }
  return fn;
}

static uint32_t strip_linear_flag(const char* function_name,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t format,
                                  uint32_t flags) {
  if (!shim_enabled()) {
    return flags;
  }

  uint32_t stripped;
  if (force_linear_enabled()) {
    // Keep LINEAR (CPU-mappable, no tiling) AND add RENDERING (render target for
    // SkSurface). No SCANOUT: on NVIDIA a scanout buffer is biased toward a
    // tiled/block-linear layout, which is exactly what breaks a plain DRM
    // hwdownload downstream. The forced [LINEAR] modifier list below makes this
    // authoritative even when the allocator would otherwise pick a tiled one.
    stripped = flags | GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING;
  } else {
    stripped = flags & ~GBM_BO_USE_LINEAR;
    if (add_scanout_enabled()) {
      // Force SCANOUT *and* RENDERING. Stock Chromium requests a CPU-mappable
      // usage for renderable-mappable SharedImages, which on NVIDIA GBM yields a
      // scanout-only buffer WITHOUT GBM_BO_USE_RENDERING -> the buffer cannot be
      // a render target and SkSurface init fails. The custom patch 0005 forces
      // gfx::BufferUsage::SCANOUT (which carries RENDERING); we replicate that
      // here by adding the RENDERING bit the stock path drops.
      stripped |= (GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    }
  }
  if (log_enabled()) {
    if (flags != stripped) {
      fprintf(stderr,
              "[gbm-linear-shim] %s %ux%u format=0x%08x flags 0x%08x -> "
              "0x%08x (rewrote usage flags)\n",
              function_name, width, height, format, flags, stripped);
    } else {
      fprintf(stderr,
              "[gbm-linear-shim] %s %ux%u format=0x%08x flags 0x%08x "
              "(unchanged)\n",
              function_name, width, height, format, flags);
    }
  }
  return stripped;
}

// Decide the modifier list to hand to the real allocator. Writes the chosen
// list + count through *out_modifiers / *out_count (which default to the
// caller's own list, i.e. a no-op). `scratch` must hold >= scratch_count
// entries. Handles two mutually exclusive rewrites:
//   - force-linear: replace the whole list with [DRM_FORMAT_MOD_LINEAR] so the
//     driver cannot pick a tiled layout (pairs with the LINEAR usage flag);
//   - strip-linear-modifier: drop DRM_FORMAT_MOD_LINEAR entries (legacy path).
static void choose_modifiers(const char* function_name,
                             const uint64_t* modifiers,
                             unsigned int count,
                             uint64_t* scratch,
                             unsigned int scratch_count,
                             const uint64_t** out_modifiers,
                             unsigned int* out_count) {
  *out_modifiers = modifiers;
  *out_count = count;
  if (!modifiers || count == 0 || !shim_enabled()) {
    return;
  }

  if (force_linear_enabled()) {
    if (scratch_count >= 1) {
      scratch[0] = DRM_FORMAT_MOD_LINEAR;
      *out_modifiers = scratch;
      *out_count = 1;
      if (log_enabled()) {
        fprintf(stderr,
                "[gbm-linear-shim] %s forced modifier list -> [LINEAR] "
                "(was %u)\n",
                function_name, count);
      }
    }
    return;
  }

  if (!strip_linear_modifier_enabled()) {
    return;
  }

  unsigned int out = 0;
  unsigned int linear = 0;
  for (unsigned int i = 0; i < count; i++) {
    if (modifiers[i] == DRM_FORMAT_MOD_LINEAR) {
      linear++;
      continue;
    }
    if (out < scratch_count) {
      scratch[out++] = modifiers[i];
    }
  }

  if (linear == 0 || out == 0) {
    if (out == 0 && linear != 0 && log_enabled()) {
      fprintf(stderr,
              "[gbm-linear-shim] %s modifier list only had LINEAR; leaving it "
              "unchanged\n",
              function_name);
    }
    return;
  }

  if (log_enabled()) {
    fprintf(stderr,
            "[gbm-linear-shim] %s removed %u DRM_FORMAT_MOD_LINEAR modifier(s), "
            "%u remain\n",
            function_name, linear, out);
  }
  *out_modifiers = scratch;
  *out_count = out;
}

struct gbm_bo* gbm_bo_create(struct gbm_device* gbm,
                             uint32_t width,
                             uint32_t height,
                             uint32_t format,
                             uint32_t flags) {
  static gbm_bo_create_fn real_fn = NULL;
  if (!real_fn) {
    real_fn = (gbm_bo_create_fn)load_next("gbm_bo_create");
  }
  if (!real_fn) {
    errno = ENOSYS;
    return NULL;
  }
  return real_fn(gbm, width, height, format,
                 strip_linear_flag("gbm_bo_create", width, height, format,
                                   flags));
}

struct gbm_bo* gbm_bo_create_with_modifiers(struct gbm_device* gbm,
                                            uint32_t width,
                                            uint32_t height,
                                            uint32_t format,
                                            const uint64_t* modifiers,
                                            const unsigned int count) {
  static gbm_bo_create_with_modifiers_fn real_fn = NULL;
  if (!real_fn) {
    real_fn = (gbm_bo_create_with_modifiers_fn)load_next(
        "gbm_bo_create_with_modifiers");
  }
  if (!real_fn) {
    errno = ENOSYS;
    return NULL;
  }

  uint64_t scratch[64];
  const uint64_t* out_modifiers;
  unsigned int out_count;
  choose_modifiers("gbm_bo_create_with_modifiers", modifiers, count, scratch, 64,
                   &out_modifiers, &out_count);
  if (log_enabled()) {
    fprintf(stderr,
            "[gbm-linear-shim] gbm_bo_create_with_modifiers %ux%u "
            "format=0x%08x modifier_count=%u%s\n",
            width, height, format, count,
            out_modifiers == modifiers ? "" : " (rewritten)");
  }
  return real_fn(gbm, width, height, format, out_modifiers, out_count);
}

struct gbm_bo* gbm_bo_create_with_modifiers2(struct gbm_device* gbm,
                                             uint32_t width,
                                             uint32_t height,
                                             uint32_t format,
                                             const uint64_t* modifiers,
                                             const unsigned int count,
                                             uint32_t flags) {
  static gbm_bo_create_with_modifiers2_fn real_fn = NULL;
  if (!real_fn) {
    real_fn = (gbm_bo_create_with_modifiers2_fn)load_next(
        "gbm_bo_create_with_modifiers2");
  }
  if (!real_fn) {
    errno = ENOSYS;
    return NULL;
  }

  uint64_t scratch[64];
  const uint64_t* out_modifiers;
  unsigned int out_count;
  choose_modifiers("gbm_bo_create_with_modifiers2", modifiers, count, scratch,
                   64, &out_modifiers, &out_count);
  return real_fn(gbm, width, height, format, out_modifiers, out_count,
                 strip_linear_flag("gbm_bo_create_with_modifiers2", width,
                                   height, format, flags));
}

struct gbm_surface* gbm_surface_create(struct gbm_device* gbm,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t format,
                                       uint32_t flags) {
  static gbm_surface_create_fn real_fn = NULL;
  if (!real_fn) {
    real_fn = (gbm_surface_create_fn)load_next("gbm_surface_create");
  }
  if (!real_fn) {
    errno = ENOSYS;
    return NULL;
  }
  return real_fn(gbm, width, height, format,
                 strip_linear_flag("gbm_surface_create", width, height, format,
                                   flags));
}

struct gbm_surface* gbm_surface_create_with_modifiers(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    const unsigned int count) {
  static gbm_surface_create_with_modifiers_fn real_fn = NULL;
  if (!real_fn) {
    real_fn = (gbm_surface_create_with_modifiers_fn)load_next(
        "gbm_surface_create_with_modifiers");
  }
  if (!real_fn) {
    errno = ENOSYS;
    return NULL;
  }

  uint64_t scratch[64];
  const uint64_t* out_modifiers;
  unsigned int out_count;
  choose_modifiers("gbm_surface_create_with_modifiers", modifiers, count,
                   scratch, 64, &out_modifiers, &out_count);
  if (log_enabled()) {
    fprintf(stderr,
            "[gbm-linear-shim] gbm_surface_create_with_modifiers %ux%u "
            "format=0x%08x modifier_count=%u%s\n",
            width, height, format, count,
            out_modifiers == modifiers ? "" : " (rewritten)");
  }
  return real_fn(gbm, width, height, format, out_modifiers, out_count);
}

struct gbm_surface* gbm_surface_create_with_modifiers2(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    const unsigned int count,
    uint32_t flags) {
  static gbm_surface_create_with_modifiers2_fn real_fn = NULL;
  if (!real_fn) {
    real_fn = (gbm_surface_create_with_modifiers2_fn)load_next(
        "gbm_surface_create_with_modifiers2");
  }
  if (!real_fn) {
    errno = ENOSYS;
    return NULL;
  }

  uint64_t scratch[64];
  const uint64_t* out_modifiers;
  unsigned int out_count;
  choose_modifiers("gbm_surface_create_with_modifiers2", modifiers, count,
                   scratch, 64, &out_modifiers, &out_count);
  return real_fn(gbm, width, height, format, out_modifiers, out_count,
                 strip_linear_flag("gbm_surface_create_with_modifiers2", width,
                                   height, format, flags));
}

void* gbm_bo_map(struct gbm_bo* bo,
                 uint32_t x,
                 uint32_t y,
                 uint32_t width,
                 uint32_t height,
                 uint32_t flags,
                 uint32_t* stride,
                 void** map_data) {
  static gbm_bo_map_fn real_fn = NULL;
  if (!real_fn) {
    real_fn = (gbm_bo_map_fn)load_next("gbm_bo_map");
  }
  if (!real_fn) {
    errno = ENOSYS;
    return NULL;
  }
  void* result = real_fn(bo, x, y, width, height, flags, stride, map_data);
  if (log_enabled()) {
    fprintf(stderr,
            "[gbm-linear-shim] gbm_bo_map x=%u y=%u %ux%u flags=0x%08x -> %p "
            "stride=%u errno=%d\n",
            x, y, width, height, flags, result, stride ? *stride : 0, errno);
  }
  return result;
}
