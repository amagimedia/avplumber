#include "hwaccel_mgmt.hpp"
#include "hwaccel.hpp"

void initHWAccel(InstanceData &instance, Parameters &params) {
    using ISOs = InstanceSharedObjects<HWAccelDevice>;
    ISOs::emplace(instance, params["name"], ISOs::PolicyIfExists::Ignore, params);
}
