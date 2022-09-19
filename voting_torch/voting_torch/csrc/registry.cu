#include <torch/script.h>
#include <torch/extension.h>

#include "vote_kernels.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("cc_2d", &connected_componnets_labeling_2d, "connected_componnets_labeling_2d");
    // m.def("cc_3d", &connected_componnets_labeling_3d, "connected_componnets_labeling_d");
    m.def("vote_translation", &vote_translation, "vote_translation");
    m.def("vote_rotation", &vote_rotation, "vote_rotation");
    m.def("backvote", &backvote, "backvote");
}