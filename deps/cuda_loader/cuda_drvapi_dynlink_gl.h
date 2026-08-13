/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __cuda_drvapi_dynlink_cuda_gl_h__
#define __cuda_drvapi_dynlink_cuda_gl_h__

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <GL/gl.h>
#include <EGL/egl.h>
#include "EGL/eglext.h"

typedef enum CUeglFrameType_enum {
    CU_EGL_FRAME_TYPE_ARRAY = 0,
    CU_EGL_FRAME_TYPE_PITCH = 1,
} CUeglFrameType;

typedef enum CUeglColorFormat_enum {
    CU_EGL_COLOR_FORMAT_RGB = 0x04,
    CU_EGL_COLOR_FORMAT_BGR = 0x05,
    CU_EGL_COLOR_FORMAT_ARGB = 0x06,
    CU_EGL_COLOR_FORMAT_RGBA = 0x07,
    CU_EGL_COLOR_FORMAT_ABGR = 0x0E,
    CU_EGL_COLOR_FORMAT_BGRA = 0x0F,
} CUeglColorFormat;

#ifndef MAX_PLANES
#define MAX_PLANES 3
#endif

typedef struct CUeglFrame_st {
    union {
        CUarray pArray[MAX_PLANES];
        void *pPitch[MAX_PLANES];
    } frame;
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    unsigned int pitch;
    unsigned int planeCount;
    unsigned int numChannels;
    CUeglFrameType frameType;
    CUeglColorFormat eglColorFormat;
    CUarray_format cuFormat;
} CUeglFrame;

/************************************
 **
 **    OpenGL Graphics/Interop
 **
 ***********************************/

// OpenGL/CUDA interop (CUDA 2.0+)
typedef CUresult CUDAAPI tcuGLCtxCreate(CUcontext *pCtx, unsigned int Flags, CUdevice device);
typedef CUresult CUDAAPI tcuGraphicsGLRegisterBuffer(CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags);
typedef CUresult CUDAAPI tcuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags);

typedef CUresult CUDAAPI tcuGraphicsEGLRegisterImage(
		    CUgraphicsResource *pCudaResource, EGLImageKHR image, unsigned int flags);
typedef CUresult CUDAAPI tcuGraphicsResourceGetMappedEglFrame(
		    CUeglFrame *eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel);


extern tcuGLCtxCreate *cuGLCtxCreate;
extern tcuGraphicsGLRegisterBuffer *cuGraphicsGLRegisterBuffer;
extern tcuGraphicsGLRegisterImage *cuGraphicsGLRegisterImage;
extern tcuGraphicsEGLRegisterImage *cuGraphicsEGLRegisterImage;
extern tcuGraphicsResourceGetMappedEglFrame *cuGraphicsResourceGetMappedEglFrame;

#endif // __cuda_drvapi_dynlink_cuda_gl_h__
