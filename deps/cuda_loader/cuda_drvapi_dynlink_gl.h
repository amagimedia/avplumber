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

struct CUeglFrame_st;
typedef struct CUeglFrame_st CUeglFrame;

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

#endif // __cuda_drvapi_dynlink_cuda_gl_h__
