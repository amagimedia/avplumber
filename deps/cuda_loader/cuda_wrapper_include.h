// includes
#include "cuda_drvapi_dynlink.h"
#include "drvapi_error_string.h"

// missing extern function signatures
extern tcuIpcGetMemHandle *cuIpcGetMemHandle;
extern tcuIpcOpenMemHandle *cuIpcOpenMemHandle;
extern tcuIpcOpenMemHandle *cuIpcOpenMemHandle;
extern tcuIpcCloseMemHandle *cuIpcCloseMemHandle;
#ifdef CUDA_INIT_OPENGL
extern tcuGraphicsGLRegisterImage *cuGraphicsGLRegisterImage;
#endif

// helpers
#define checkCudaErrors(err, func_call)  __checkCudaErrors(err, __FILE__, __LINE__, func_call)
#define CHECK_CU(x) checkCudaErrors((x), #x)

inline static int __checkCudaErrors(CUresult err, const char *file, const int line, const char *func_call)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i (%s).\n",
                err, getCudaDrvErrorString(err), file, line, func_call);
        return -1;
    }
    return 0;
}
