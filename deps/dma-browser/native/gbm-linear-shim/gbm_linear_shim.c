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

#define GBM_BO_USE_SCANOUT (1u << 0)
#define GBM_BO_USE_RENDERING (1u << 2)
#define GBM_BO_USE_LINEAR (1u << 4)

typedef struct gbm_bo* (*gbm_bo_create_fn)(struct gbm_device* gbm,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t format,
                                           uint32_t flags);
typedef struct gbm_bo* (*gbm_bo_create_with_modifiers2_fn)(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    unsigned int count,
    uint32_t flags);
typedef struct gbm_surface* (*gbm_surface_create_fn)(struct gbm_device* gbm,
                                                     uint32_t width,
                                                     uint32_t height,
                                                     uint32_t format,
                                                     uint32_t flags);
typedef struct gbm_surface* (*gbm_surface_create_with_modifiers2_fn)(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    unsigned int count,
    uint32_t flags);

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

static bool log_enabled(void) {
  return env_enabled("GBM_LINEAR_SHIM_LOG", false);
}

static void* load_next(const char* symbol) {
  void* function = dlsym(RTLD_NEXT, symbol);
  if (!function) {
    fprintf(stderr, "[gbm-linear-shim] missing RTLD_NEXT symbol: %s: %s\n",
            symbol, dlerror());
  }
  return function;
}

static uint32_t rewrite_usage(const char* function_name,
                              uint32_t width,
                              uint32_t height,
                              uint32_t format,
                              uint32_t flags) {
  if (!env_enabled("GBM_LINEAR_SHIM", true)) {
    return flags;
  }

  uint32_t rewritten = flags & ~GBM_BO_USE_LINEAR;
  if (env_enabled("GBM_LINEAR_SHIM_ADD_SCANOUT", true)) {
    rewritten |= GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING;
  }

  if (log_enabled()) {
    fprintf(stderr,
            "[gbm-linear-shim] %s %ux%u format=0x%08x flags 0x%08x -> "
            "0x%08x%s\n",
            function_name, width, height, format, flags, rewritten,
            flags == rewritten ? " (unchanged)" : " (rewrote usage flags)");
  }
  return rewritten;
}

#define RESOLVE_REAL(type, variable, symbol)             \
  static type variable = NULL;                           \
  if (!variable) {                                       \
    variable = (type)load_next(symbol);                   \
  }                                                       \
  if (!variable) {                                       \
    errno = ENOSYS;                                      \
    return NULL;                                         \
  }

struct gbm_bo* gbm_bo_create(struct gbm_device* gbm,
                             uint32_t width,
                             uint32_t height,
                             uint32_t format,
                             uint32_t flags) {
  RESOLVE_REAL(gbm_bo_create_fn, real_fn, "gbm_bo_create");
  return real_fn(gbm, width, height, format,
                 rewrite_usage("gbm_bo_create", width, height, format, flags));
}

struct gbm_bo* gbm_bo_create_with_modifiers2(struct gbm_device* gbm,
                                             uint32_t width,
                                             uint32_t height,
                                             uint32_t format,
                                             const uint64_t* modifiers,
                                             unsigned int count,
                                             uint32_t flags) {
  RESOLVE_REAL(gbm_bo_create_with_modifiers2_fn, real_fn,
               "gbm_bo_create_with_modifiers2");
  return real_fn(gbm, width, height, format, modifiers, count,
                 rewrite_usage("gbm_bo_create_with_modifiers2", width, height,
                               format, flags));
}

struct gbm_surface* gbm_surface_create(struct gbm_device* gbm,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t format,
                                       uint32_t flags) {
  RESOLVE_REAL(gbm_surface_create_fn, real_fn, "gbm_surface_create");
  return real_fn(
      gbm, width, height, format,
      rewrite_usage("gbm_surface_create", width, height, format, flags));
}

struct gbm_surface* gbm_surface_create_with_modifiers2(
    struct gbm_device* gbm,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    const uint64_t* modifiers,
    unsigned int count,
    uint32_t flags) {
  RESOLVE_REAL(gbm_surface_create_with_modifiers2_fn, real_fn,
               "gbm_surface_create_with_modifiers2");
  return real_fn(gbm, width, height, format, modifiers, count,
                 rewrite_usage("gbm_surface_create_with_modifiers2", width,
                               height, format, flags));
}
