#pragma once

#include "network/network_builder.hpp"

#include <string>
#include <vector>

namespace neurong::runtime_api::core {
class SimRuntimeCore;
}

namespace neurong_network {

int resolve_arti_presyn_index_by_loc(neurong::runtime_api::core::SimRuntimeCore& core,
                                     const std::string& mech,
                                     int gid,
                                     const std::string& section_name,
                                     double loc,
                                     int slot = 0);

std::vector<int> resolve_arti_presyn_indices_by_locs(
    neurong::runtime_api::core::SimRuntimeCore& core,
    const std::string& mech,
    const std::vector<int>& gids,
    const std::vector<std::string>& section_names,
    const std::vector<double>& locs);

int load_network_real(neurong::runtime_api::core::SimRuntimeCore& core,
                      const std::vector<ConnectionSpec>& connections);

int load_network_artificial(neurong::runtime_api::core::SimRuntimeCore& core,
                            const std::vector<ConnectionSpec>& connections);

int load_network_artificial_packed(
    neurong::runtime_api::core::SimRuntimeCore& core,
    const std::vector<int>& pre_artificial_indices,
    const std::vector<int>& post_gids,
    const std::vector<int>& post_section_indices,
    const std::vector<double>& post_locs,
    const std::vector<double>& weights,
    const std::vector<double>& delays,
    const std::string& post_mech);

}  // namespace neurong_network
