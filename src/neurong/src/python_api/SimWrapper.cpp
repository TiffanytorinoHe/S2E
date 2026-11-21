#include "SimWrapper.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <cuda_runtime.h>
#include <stdexcept>
#include "celltemplate_pybind_parsing.hpp"
#include "network/network_pybind_parsing.hpp"
#include "network/gap_route_ops.hpp"
#include "network/network_ops.hpp"
#include "spike/vecevent.h"

SimWrapper::SimWrapper() {
}

SimWrapper::~SimWrapper() {
    clear_dense_blocks();
}

int SimWrapper::set_data_path(const string& path) {
    return core_.set_data_path(path);
}

int SimWrapper::set_device(const string& dev) {
    return core_.set_device(dev);
}

int SimWrapper::set_source(const string& source) {
    return core_.set_source(source);
}

string SimWrapper::get_source() const {
    return core_.get_source();
}

int SimWrapper::load_mech_library(const string& path) {
    return core_.load_mech_library(path);
}

std::vector<std::string> SimWrapper::get_loaded_mech_libraries() const {
    return core_.get_loaded_mech_libraries();
}

int SimWrapper::load_model() {
    gap_routes_.clear(nullptr);
    sim = nullptr;
    int rc = core_.load_model();
    if (rc < 0) {
        return rc;
    }
    sim = core_.sim();
    if (sim == nullptr) {
        printf("load_model: core_.sim() is null after successful load\n");
        return -1;
    }
    return rc;
}

int SimWrapper::maybe_finalize_gap_routes() {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return gap_routes_.finalize(*sim);
}

std::vector<std::tuple<std::string, int, int>> SimWrapper::load_celltemplate_morphology(
    const nb::list& morph_templates,
    const nb::object& random_seed) {
    gap_routes_.clear(nullptr);
    const auto specs = neurong_pybind::parse_celltemplate_morphology_specs(morph_templates);
    std::optional<std::uint64_t> random_seed_u64{};
    if (!random_seed.is_none()) {
        random_seed_u64 = nb::cast<std::uint64_t>(random_seed);
    }
    auto gid_ranges = core_.load_celltemplate_morphology(specs, random_seed_u64);
    sim = core_.sim();
    return gid_ranges;
}

std::vector<std::tuple<int, int, int>> SimWrapper::apply_celltemplate_biophysics(
    const nb::list& biophys_templates,
    double celsius,
    const std::vector<int>& record_gids) {
    const auto specs = neurong_pybind::parse_celltemplate_biophys_specs(biophys_templates);
    auto handles = core_.apply_celltemplate_biophysics(specs, celsius, record_gids);
    sim = core_.sim();
    return handles;
}

std::vector<std::tuple<std::string, int, int>> SimWrapper::get_celltemplate_gid_ranges() const {
    return core_.get_celltemplate_gid_ranges();
}

std::tuple<int, int> SimWrapper::get_celltemplate_gid_range(const std::string& template_name) const {
    const auto ranges = core_.get_celltemplate_gid_ranges();
    for (const auto& row : ranges) {
        if (std::get<0>(row) == template_name) {
            return std::make_tuple(std::get<1>(row), std::get<2>(row));
        }
    }
    throw std::runtime_error("template_name not found in loaded morphology: " + template_name);
}

int SimWrapper::load_network_real(const nb::list& connections) {
    const auto specs = neurong_pybind::parse_connections_real_only(connections);
    return neurong_network::load_network_real(core_, specs);
}

int SimWrapper::load_network_artificial(const nb::list& connections) {
    const auto specs = neurong_pybind::parse_connections_artificial_only(connections);
    return neurong_network::load_network_artificial(core_, specs);
}

int SimWrapper::load_network_artificial_packed(
    const std::vector<int>& pre_artificial_indices,
    const std::vector<int>& post_gids,
    const std::vector<int>& post_section_indices,
    const std::vector<double>& post_locs,
    const std::vector<double>& weights,
    const std::vector<double>& delays,
    const std::string& post_mech) {
    return neurong_network::load_network_artificial_packed(
        core_, pre_artificial_indices, post_gids, post_section_indices,
        post_locs, weights, delays, post_mech);
}

int SimWrapper::resolve_node_index(int gid, const string& section_name, int seg_index) {
    return core_.resolve_node_index(gid, section_name, seg_index);
}

int SimWrapper::resolve_mech_index_by_loc(const string& mech, int gid, const string& section_name, double loc, int slot) {
    return core_.resolve_mech_index_by_loc(mech, gid, section_name, loc, slot);
}

int SimWrapper::resolve_segment_index_by_loc(int gid, const string& section_name, double loc) {
    return core_.resolve_segment_index_by_loc(gid, section_name, loc);
}

int SimWrapper::resolve_node_index_by_loc(int gid, const string& section_name, double loc) {
    return core_.resolve_node_index_by_loc(gid, section_name, loc);
}

int SimWrapper::resolve_node_index_internal_from_original(int node_idx_original) {
    return core_.resolve_node_index_internal_from_original(node_idx_original);
}

int SimWrapper::resolve_node_index_permuted_from_handle(int handle) const {
    return core_.resolve_node_index_permuted_from_handle(handle);
}

int SimWrapper::resolve_mech_index_permuted_from_handle(int handle) const {
    return core_.resolve_mech_index_permuted_from_handle(handle);
}

double SimWrapper::resolve_segment_center_loc_by_loc(int gid, const string& section_name, double loc) {
    return core_.resolve_segment_center_loc_by_loc(gid, section_name, loc);
}

double SimWrapper::resolve_effective_loc_by_loc(const string& mech,
                                                int gid,
                                                const string& section_name,
                                                double loc) {
    return core_.resolve_effective_loc_by_loc(mech, gid, section_name, loc);
}

int SimWrapper::finitialize(double v_init) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (maybe_finalize_gap_routes() < 0) {
        return -1;
    }
    
    // 在初始化前同步脏变量到GPU
    core_.flush_dirty_variables();
    
    // printf("v_init: %f\n", v_init);
    sim->finitialize(v_init);
    // printf("finitialize finished\n");
    return 0;
}

int SimWrapper::run(double tstop) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (!sim->is_finitialized()) {
        printf("run: finitialize() must be called before run()\n");
        return -1;
    }
    if (maybe_finalize_gap_routes() < 0) {
        return -1;
    }
    
    // 在运行前同步脏变量到GPU
    core_.flush_dirty_variables();
    
    sim->tstop = tstop;
    // printf("tstop: %f\n", tstop);

    auto start_time = clock();
    sim->run();
    auto end_time = clock();
    // printf("simulation finished, time:%fs\n", ((float)end_time - start_time) / CLOCKS_PER_SEC);

    if (core_.is_spike_output_enabled()) {
        sim->output_spikes();
    }
    return 0;
}

int SimWrapper::continue_run(double runtime) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (!sim->is_finitialized()) {
        printf("continue_run: finitialize() must be called before continue_run()\n");
        return -1;
    }
    if (maybe_finalize_gap_routes() < 0) {
        return -1;
    }

    // 在运行前同步脏变量到GPU
    core_.flush_dirty_variables();

    sim->continue_run(runtime);

    if (core_.is_spike_output_enabled()) {
        sim->output_spikes();
    }
    return 0;
}

int SimWrapper::fadvance() {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (!sim->is_finitialized()) {
        printf("fadvance: finitialize() must be called before fadvance()\n");
        return -1;
    }
    if (maybe_finalize_gap_routes() < 0) {
        return -1;
    }

    // step前同步脏变量到GPU（例如用户通过handle写入了参数）
    core_.flush_dirty_variables();
    sim->fadvance();
    return 0;
}

double SimWrapper::get_t() const {
    if (sim == nullptr) {
        return 0.0;
    }
    return sim->t;
}

int SimWrapper::flush_recorders() {
    return core_.flush_recorders();
}

int SimWrapper::set_spike_output_enabled(bool enable) {
    return core_.set_spike_output_enabled(enable);
}

bool SimWrapper::is_spike_output_enabled() const {
    return core_.is_spike_output_enabled();
}

int SimWrapper::set_permute_type(int type) {
    return core_.set_permute_type(type);
}

int SimWrapper::add_monitor(const string& mech, const string& var, int node_or_mech_idx) {
    return core_.add_monitor(mech, var, node_or_mech_idx);
}

int SimWrapper::add_monitor_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index) {
    return core_.add_monitor_with_array(mech, var, node_or_mech_idx, array_index);
}

int SimWrapper::set_dt(double dt) {
    return core_.set_dt(dt);
}

double SimWrapper::get_dt() {
    return core_.get_dt();
}

int SimWrapper::set_output_dir(const string& dir) {
    return core_.set_output_dir(dir);
}

vector<double> SimWrapper::get_monitor_data(int handle) {
    return core_.get_monitor_data(handle);
}

int SimWrapper::get_monitor_sample_count(int handle) {
    return core_.get_monitor_sample_count(handle);
}

map<int, vector<double>> SimWrapper::get_multiple_monitor_data(const vector<int>& handles) {
    return core_.get_multiple_monitor_data(handles);
}

int SimWrapper::get_multiple_monitor_data_f32_into(const std::vector<int>& handles,
                                                   nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> out) {
    if (out.ndim() != 2) {
        printf("get_multiple_monitor_data_f32_into: out must be 2D\n");
        return -1;
    }

    const int n_handle = static_cast<int>(handles.size());
    if (static_cast<int>(out.shape(0)) != n_handle) {
        printf("get_multiple_monitor_data_f32_into: first dim mismatch (%zd vs %d)\n",
               static_cast<ssize_t>(out.shape(0)), n_handle);
        return -1;
    }

    const int n_step = static_cast<int>(out.shape(1));
    if (n_step < 0) {
        printf("get_multiple_monitor_data_f32_into: invalid step size %d\n", n_step);
        return -1;
    }
    if (n_handle == 0 || n_step == 0) {
        return 0;
    }

    float* out_cpu = out.data();
    return core_.get_multiple_monitor_data_f32(handles, out_cpu, n_handle, n_step);
}

int SimWrapper::get_multiple_monitor_data_f32_from_into(const std::vector<int>& handles,
                                                        int start_step,
                                                        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> out) {
    if (out.ndim() != 2) {
        printf("get_multiple_monitor_data_f32_from_into: out must be 2D\n");
        return -1;
    }

    const int n_handle = static_cast<int>(handles.size());
    if (static_cast<int>(out.shape(0)) != n_handle) {
        printf("get_multiple_monitor_data_f32_from_into: first dim mismatch (%zd vs %d)\n",
               static_cast<ssize_t>(out.shape(0)), n_handle);
        return -1;
    }

    const int n_step = static_cast<int>(out.shape(1));
    if (n_step < 0) {
        printf("get_multiple_monitor_data_f32_from_into: invalid step size %d\n", n_step);
        return -1;
    }
    if (n_handle == 0 || n_step == 0) {
        return 0;
    }

    float* out_cpu = out.data();
    return core_.get_multiple_monitor_data_f32_from(handles, start_step, out_cpu, n_handle, n_step);
}

int SimWrapper::add_window_observer(const string& mech, const string& var, int node_or_mech_idx) {
    return core_.add_window_observer(mech, var, node_or_mech_idx);
}

int SimWrapper::add_window_observer_with_array(const string& mech,
                                               const string& var,
                                               int node_or_mech_idx,
                                               int array_index) {
    return core_.add_window_observer_with_array(mech, var, node_or_mech_idx, array_index);
}

int SimWrapper::get_window_observer_handle(const string& mech, const string& var, int node_or_mech_idx) {
    return core_.get_window_observer_handle(mech, var, node_or_mech_idx);
}

int SimWrapper::get_window_observer_handle_with_array(const string& mech,
                                                      const string& var,
                                                      int node_or_mech_idx,
                                                      int array_index) {
    return core_.get_window_observer_handle_with_array(mech, var, node_or_mech_idx, array_index);
}

int SimWrapper::begin_window_observer_recording(const std::vector<int>& handles, int step_count) {
    return core_.begin_window_observer_recording(handles, step_count);
}

int SimWrapper::end_window_observer_recording() {
    return core_.end_window_observer_recording();
}

int SimWrapper::get_window_observer_data_f32_into(const std::vector<int>& handles,
                                                  nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> out) {
    if (out.ndim() != 2) {
        printf("get_window_observer_data_f32_into: out must be 2D\n");
        return -1;
    }
    const int n_handle = static_cast<int>(handles.size());
    if (static_cast<int>(out.shape(0)) != n_handle) {
        printf("get_window_observer_data_f32_into: first dim mismatch (%zd vs %d)\n",
               static_cast<ssize_t>(out.shape(0)), n_handle);
        return -1;
    }
    const int n_step = static_cast<int>(out.shape(1));
    if (n_step < 0) {
        printf("get_window_observer_data_f32_into: invalid step size %d\n", n_step);
        return -1;
    }
    return core_.get_window_observer_data_f32(handles, out.data(), n_handle, n_step);
}


double SimWrapper::get_variable_value(const string& mech, const string& var, int node_or_mech_idx){
    return core_.get_variable_value(mech, var, node_or_mech_idx);
}

double SimWrapper::get_variable_value_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index){
    return core_.get_variable_value_with_array(mech, var, node_or_mech_idx, array_index);
}

int SimWrapper::get_group_len(int group_index) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (group_index < 0 || group_index >= static_cast<int>(sim->neuron_group_list.size()) ||
        sim->neuron_group_list[group_index] == nullptr) {
        printf("get_group_len: invalid group_index=%d\n", group_index);
        return -1;
    }
    return sim->neuron_group_list[group_index]->len;
}

int SimWrapper::prepare_vjp_from_handles(
    int group_index,
    const std::vector<int>& local_grad_handles,
    int step_count,
    const std::vector<string>& current_vjp_mechs) {
    if (sim == nullptr) {
        throw std::runtime_error("Simulate not initialized");
    }
    if (maybe_finalize_gap_routes() < 0) {
        throw std::runtime_error("prepare_vjp_from_handles: failed to finalize pending gap routes");
    }

    std::vector<int> local_grad_rows;
    local_grad_rows.reserve(local_grad_handles.size());
    for (int handle : local_grad_handles) {
        const int row = core_.resolve_node_index_permuted_from_handle(handle);
        if (row < 0) {
            throw std::runtime_error(
                "prepare_vjp_from_handles: failed to resolve voltage handle " +
                std::to_string(handle) +
                " to an internal VJP node row");
        }
        local_grad_rows.push_back(row);
    }

    sim->prepare_vjp(
        group_index,
        local_grad_rows,
        step_count,
        current_vjp_mechs);
    return 0;
}

int SimWrapper::set_spike_vjp_surrogate(const string& kind, double width_mv) {
    if (sim == nullptr) {
        throw std::runtime_error("Simulate not initialized");
    }
    sim->set_spike_vjp_surrogate(kind, width_mv);
    return 0;
}

int SimWrapper::run_vjp(
    const std::vector<std::vector<double>>& local_grad_row_history) {
    if (sim == nullptr) {
        throw std::runtime_error("Simulate not initialized");
    }
    sim->run_vjp(local_grad_row_history);
    return 0;
}

int SimWrapper::run_vjp_from_forward_grads(nb::ndarray<double, nb::c_contig> local_grad_forward) {
    if (sim == nullptr) {
        throw std::runtime_error("Simulate not initialized");
    }
    const int ndim = static_cast<int>(local_grad_forward.ndim());
    if (ndim < 2) {
        throw std::runtime_error("run_vjp_from_forward_grads: expected at least 2 dimensions");
    }

    const int step_count = static_cast<int>(local_grad_forward.shape(ndim - 1));
    if (step_count < 0) {
        throw std::runtime_error("run_vjp_from_forward_grads: invalid step count");
    }

    std::size_t local_count = 1;
    for (int axis = 0; axis < ndim - 1; ++axis) {
        local_count *= static_cast<std::size_t>(local_grad_forward.shape(axis));
    }
    if (local_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("run_vjp_from_forward_grads: too many local grad rows");
    }

    sim->run_vjp_forward_grads_contiguous(
        local_grad_forward.data(),
        static_cast<int>(local_count),
        step_count);
    return 0;
}

int SimWrapper::zero_learnable_grads_cpu() {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    for (auto* group : sim->neuron_group_list) {
        if (group == nullptr) {
            continue;
        }
        for (auto* mech : group->mechanism_list) {
            if (mech == nullptr || !mech->has_learnable_params()) {
                continue;
            }
            for (const auto& name : mech->listLearnableGradNames()) {
                const int count = mech->node_count();
                double* grad_ptr_cpu = mech->getLearnableGradPtr(name, CPU);
                if (grad_ptr_cpu != nullptr) {
                    std::fill_n(grad_ptr_cpu, count, 0.0);
                }
                if (sim->mode == GPU) {
                    double* grad_ptr_gpu = mech->getLearnableGradPtr(name, GPU);
                    if (grad_ptr_gpu != nullptr) {
                        if (grad_ptr_cpu != nullptr) {
                            mem_copy_cpu2gpu_sync(grad_ptr_gpu, grad_ptr_cpu, static_cast<int>(sizeof(double) * count));
                        } else {
                            std::vector<double> zeros(static_cast<std::size_t>(count), 0.0);
                            mem_copy_cpu2gpu_sync(
                                grad_ptr_gpu,
                                zeros.data(),
                                static_cast<int>(sizeof(double) * count));
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int SimWrapper::get_learnable_grad_handle_from_variable_handle(int variable_handle) {
    return core_.get_learnable_grad_handle_from_variable_handle(variable_handle);
}

std::vector<int> SimWrapper::get_learnable_grad_handles_from_variable_handles(
    const std::vector<int>& variable_handles) {
    return core_.get_learnable_grad_handles_from_variable_handles(variable_handles);
}

int SimWrapper::create_optimizer(const string& optimizer_type) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    std::string lower = optimizer_type;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });

    OptimizerType type;
    if (lower == "sgd") {
        type = OptimizerType::SGD;
    } else if (lower == "momentum" || lower == "sgdmomentum" || lower == "sgd_momentum") {
        type = OptimizerType::Momentum;
    } else if (lower == "adam") {
        type = OptimizerType::Adam;
    } else {
        printf("Unsupported optimizer type: %s\n", optimizer_type.c_str());
        return -1;
    }
    return sim->create_optimizer(type);
}

int SimWrapper::optimizer_add_param(int optimizer_id, int weight_handle, int grad_handle, double impedance) {
    std::vector<int> weight_handles{weight_handle};
    std::vector<int> grad_handles{grad_handle};
    return optimizer_add_param_batch(optimizer_id, weight_handles, grad_handles, impedance);
}

int SimWrapper::optimizer_add_param_batch(int optimizer_id,
                                          const std::vector<int>& weight_handles,
                                          const std::vector<int>& grad_handles,
                                          double impedance) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (weight_handles.size() != grad_handles.size() || weight_handles.empty()) {
        printf("optimizer_add_param_batch: handle lists must be non-empty and have equal length\n");
        return -1;
    }

    std::vector<double*> weight_cpu;
    std::vector<double*> grad_cpu;
    std::vector<double*> weight_gpu;
    std::vector<double*> grad_gpu;
    weight_cpu.reserve(weight_handles.size());
    grad_cpu.reserve(weight_handles.size());
    weight_gpu.reserve(weight_handles.size());
    grad_gpu.reserve(weight_handles.size());

    for (size_t i = 0; i < weight_handles.size(); ++i) {
        int weight_handle = weight_handles[i];
        int grad_handle = grad_handles[i];

        double* weight_cpu_ptr = nullptr;
        double* weight_gpu_ptr = nullptr;
        double* grad_cpu_ptr = nullptr;
        double* grad_gpu_ptr = nullptr;
        if (!core_.get_cached_pointers(weight_handle, weight_cpu_ptr, weight_gpu_ptr)) {
            printf("optimizer_add_param_batch: invalid weight handle %d\n", weight_handle);
            return -1;
        }
        if (!core_.get_cached_pointers(grad_handle, grad_cpu_ptr, grad_gpu_ptr)) {
            printf("optimizer_add_param_batch: invalid grad handle %d\n", grad_handle);
            return -1;
        }

        if (weight_cpu_ptr == nullptr || grad_cpu_ptr == nullptr) {
            printf("optimizer_add_param_batch: null CPU pointer detected (index=%zu)\n", i);
            return -1;
        }

        weight_cpu.push_back(weight_cpu_ptr);
        grad_cpu.push_back(grad_cpu_ptr);

        if (sim->mode == GPU) {
            if (weight_gpu_ptr == nullptr || grad_gpu_ptr == nullptr) {
                printf("optimizer_add_param_batch: null GPU pointer detected in GPU mode (index=%zu)\n", i);
                return -1;
            }
            weight_gpu.push_back(weight_gpu_ptr);
            grad_gpu.push_back(grad_gpu_ptr);
        }
    }

    return sim->register_optimizer_param_batch(optimizer_id,
                                               weight_cpu,
                                               grad_cpu,
                                               weight_gpu,
                                               grad_gpu,
                                               impedance);
}

int SimWrapper::configure_optimizer(int optimizer_id, double momentum, double beta1, double beta2, double epsilon) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    OptimizerHyperParams params;
    params.momentum = momentum;
    params.beta1 = beta1;
    params.beta2 = beta2;
    params.epsilon = epsilon;
    return sim->configure_optimizer(optimizer_id, params);
}

int SimWrapper::optimizer_step(int optimizer_id, double learning_rate, double record_time, double dt_step) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return sim->optimizer_step(optimizer_id, learning_rate, record_time, dt_step);
}

int SimWrapper::optimizer_add_external_grads(int optimizer_id,
                                            const std::vector<int>& weight_handles,
                                            double impedance) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (weight_handles.empty()) {
        printf("optimizer_add_external_grads: empty weight_handles\n");
        return -1;
    }
    if (!(impedance > 0.0)) {
        printf("optimizer_add_external_grads: invalid impedance (%f)\n", impedance);
        return -1;
    }

    // Replace any existing external-grad binding for this optimizer_id.
    optimizer_external_grads_.erase(optimizer_id);
    auto [it, inserted] =
        optimizer_external_grads_.emplace(optimizer_id, OptimizerExternalGrads(sim->mode));
    (void)inserted;
    OptimizerExternalGrads& ext = it->second;
    ext.weight_handles = weight_handles;
    ext.impedance = impedance;
    ext.grads.resize(static_cast<int>(weight_handles.size()));

    // Zero-init CPU grads and sync to GPU if needed.
    std::fill_n(ext.grads.get_cpu_data(), ext.grads.size(), 0.0);
    if (sim->mode == GPU) {
        ext.grads.update_gpu_data_from_cpu();
    }

    // Register one optimizer param per weight handle (batch_size=1).
    for (size_t i = 0; i < weight_handles.size(); ++i) {
        const int weight_handle = weight_handles[i];
        double* weight_cpu_ptr = nullptr;
        double* weight_gpu_ptr = nullptr;
        if (!core_.get_cached_pointers(weight_handle, weight_cpu_ptr, weight_gpu_ptr)) {
            printf("optimizer_add_external_grads: invalid weight handle %d\n", weight_handle);
            return -1;
        }
        if (weight_cpu_ptr == nullptr) {
            printf("optimizer_add_external_grads: null CPU weight pointer (i=%zu)\n", i);
            return -1;
        }

        double* grad_cpu = ext.grads.get_cpu_data() + static_cast<int>(i);
        double* grad_gpu = nullptr;
        if (sim->mode == GPU) {
            if (weight_gpu_ptr == nullptr) {
                printf("optimizer_add_external_grads: null GPU weight pointer in GPU mode (i=%zu)\n", i);
                return -1;
            }
            grad_gpu = ext.grads.get_gpu_data() + static_cast<int>(i);
        }

        int rc = sim->register_optimizer_param(optimizer_id,
                                               weight_cpu_ptr,
                                               grad_cpu,
                                               weight_gpu_ptr,
                                               grad_gpu,
                                               impedance);
        if (rc < 0) {
            printf("optimizer_add_external_grads: register_optimizer_param failed (i=%zu)\n", i);
            return -1;
        }
    }

    return 0;
}

int SimWrapper::optimizer_set_external_grads_f32(int optimizer_id,
                                                nb::ndarray<float, nb::shape<-1>, nb::c_contig> grads) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    auto it = optimizer_external_grads_.find(optimizer_id);
    if (it == optimizer_external_grads_.end()) {
        printf("optimizer_set_external_grads_f32: optimizer %d has no external grads binding\n", optimizer_id);
        return -1;
    }
    OptimizerExternalGrads& ext = it->second;
    const int n = static_cast<int>(ext.weight_handles.size());
    if (n <= 0) {
        printf("optimizer_set_external_grads_f32: internal handle list is empty\n");
        return -1;
    }
    if (grads.ndim() != 1 || static_cast<int>(grads.shape(0)) != n) {
        printf("optimizer_set_external_grads_f32: grads shape mismatch (got %d expected %d)\n",
               static_cast<int>(grads.shape(0)), n);
        return -1;
    }

    double* out = ext.grads.get_cpu_data();
    const float* in = grads.data();
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<double>(in[i]);
    }
    if (sim->mode == GPU) {
        ext.grads.update_gpu_data_from_cpu();
    }
    return 0;
}

int SimWrapper::optimizer_clear_external_grads(int optimizer_id) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    auto it = optimizer_external_grads_.find(optimizer_id);
    if (it == optimizer_external_grads_.end()) {
        return 0;
    }
    OptimizerExternalGrads& ext = it->second;
    if (ext.grads.size() <= 0) {
        return 0;
    }
    std::fill_n(ext.grads.get_cpu_data(), ext.grads.size(), 0.0);
    if (sim->mode == GPU) {
        ext.grads.update_gpu_data_from_cpu();
    }
    return 0;
}

int SimWrapper::optimizer_reset_state(int optimizer_id) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return sim->optimizer_reset_state(optimizer_id);
}

std::tuple<long long, std::vector<double>, std::vector<double>, double, double, double> SimWrapper::optimizer_get_adam_state(
    int optimizer_id) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return {0, {}, {}, 0.9, 0.999, 1e-8};
    }
    long long step = 0;
    std::vector<double> m;
    std::vector<double> v;
    OptimizerHyperParams params;
    if (sim->optimizer_get_adam_state(optimizer_id, step, m, v, params) < 0) {
        return {0, {}, {}, 0.9, 0.999, 1e-8};
    }
    return {step, std::move(m), std::move(v), params.beta1, params.beta2, params.epsilon};
}

int SimWrapper::optimizer_set_adam_state(int optimizer_id,
                                        long long step_count,
                                        const std::vector<double>& m,
                                        const std::vector<double>& v,
                                        double beta1,
                                        double beta2,
                                        double epsilon) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    OptimizerHyperParams params;
    params.beta1 = beta1;
    params.beta2 = beta2;
    params.epsilon = epsilon;
    return sim->optimizer_set_adam_state(optimizer_id, step_count, m, v, params);
}

int SimWrapper::set_variable_value(double val, const string& mech, const string& var, int node_or_mech_idx) {
    return core_.set_variable_value(val, mech, var, node_or_mech_idx);
}

int SimWrapper::set_variable_value_with_array(double val, const string& mech, const string& var, int node_or_mech_idx, int array_index) {
    return core_.set_variable_value_with_array(val, mech, var, node_or_mech_idx, array_index);
}

double SimWrapper::call_mech_func(const string& mech_name, const string& func_name, nb::args args) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (mech_name == "VecStim" && func_name == "play") {
        VecEvent::setCurrentTime(sim->t);
    }
    auto result = FunctionRegistry::getInstance().call(mech_name + "." + func_name, args_to_vector(sim->mode, args));
    return 0;
}

std::string SimWrapper::resolve_mech_static_interface(const string& query) const {
    return StaticInterfaceRegistry::getInstance().resolve_interface(query);
}

bool SimWrapper::has_mech_static_interface(const string& mech_name) const {
    return StaticInterfaceRegistry::getInstance().has_interface(mech_name);
}

bool SimWrapper::has_mech_static_property(const string& mech_name, const string& property_name) const {
    return StaticInterfaceRegistry::getInstance().has_property(mech_name, property_name);
}

bool SimWrapper::has_mech_static_method(const string& mech_name, const string& method_name) const {
    return StaticInterfaceRegistry::getInstance().has_method(mech_name, method_name);
}

std::vector<std::string> SimWrapper::list_mech_static_interfaces() const {
    return StaticInterfaceRegistry::getInstance().list_interfaces();
}

std::vector<std::string> SimWrapper::list_mech_static_properties(const string& mech_name) const {
    return StaticInterfaceRegistry::getInstance().list_properties(mech_name);
}

std::vector<std::string> SimWrapper::list_mech_static_methods(const string& mech_name) const {
    return StaticInterfaceRegistry::getInstance().list_methods(mech_name);
}

std::string SimWrapper::get_mech_static_property_str(const string& mech_name, const string& property_name) const {
    return convert_any<std::string>(StaticInterfaceRegistry::getInstance().get_property(mech_name, property_name));
}

int SimWrapper::set_mech_static_property_str(const string& mech_name, const string& property_name, const string& value) {
    StaticInterfaceRegistry::getInstance().set_property(mech_name, property_name, value);
    return 0;
}

double SimWrapper::call_mech_static(const string& mech_name, const string& method_name, nb::args args) {
    StaticInterfaceRegistry::getInstance().call_method(mech_name, method_name, args_to_vector_no_mode(args));
    return 0;
}

// int SimWrapper::vecevent_play(int mech_idx, const vector<double>& data) {
//     if (sim == nullptr) {
//         printf("Simulate not initialized\n");
//         return -1;
//     }
    
//     printf("vecevent_play: mech_idx=%d len=%zu\n", mech_idx, data.size());
//     sim->vecevent_play(mech_idx, data.size(), data.data());
//     return 0;
// }

void SimWrapper::set_user_mod_num(int num) {
    core_.set_user_mod_num(num);
}

int SimWrapper::set_recorder_buffer_capacity(int capacity) {
    return core_.set_recorder_buffer_capacity(capacity);
}

int SimWrapper::set_recorder_storage_dtype(const string& dtype) {
    return core_.set_recorder_storage_dtype(dtype);
}

int SimWrapper::set_recorder_start_time(double start_time_ms) {
    return core_.set_recorder_start_time(start_time_ms);
}

int SimWrapper::set_recorder_stride(int stride_steps) {
    return core_.set_recorder_stride(stride_steps);
}

vector<double>& SimWrapper::get_spk_by_gid(int gid) {
    return sim->neuron_group_list[0]->presyn->vecdata_spk_output->get_cpu_data()[gid];
}

int SimWrapper::get_monitor_handle(const string& mech, const string& var, int node_or_mech_idx) {
    return get_monitor_handle_with_array(mech, var, node_or_mech_idx, 0);
}

int SimWrapper::get_monitor_handle_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index) {
    return core_.get_monitor_handle_with_array(mech, var, node_or_mech_idx, array_index);
}

// VecPlay相关方法实现
int SimWrapper::add_vecplay(const string& mech_name, const string& var_name, int instance_id,
                            const vector<double>& tvec, const vector<double>& yvec) {
    return core_.add_vecplay(mech_name, var_name, instance_id, tvec, yvec);
}

int SimWrapper::update_vecplay(const string& mech_name, const string& var_name, int instance_id,
                               const vector<double>& new_tvec, const vector<double>& new_yvec) {
    return core_.update_vecplay(mech_name, var_name, instance_id, new_tvec, new_yvec);
}

int SimWrapper::remove_vecplay(const string& mech_name, const string& var_name, int instance_id) {
    return core_.remove_vecplay(mech_name, var_name, instance_id);
}

bool SimWrapper::has_vecplay(const string& mech_name, const string& var_name, int instance_id) {
    return core_.has_vecplay(mech_name, var_name, instance_id);
}

vector<vector<string>> SimWrapper::get_all_vecplay_keys() {
    return core_.get_vecplay_keys();
}

int SimWrapper::set_vecplay_batch(const std::vector<std::string>& mech_names,
                                  const std::vector<std::string>& var_names,
                                  const std::vector<int>& instance_ids,
                                  nb::ndarray<double, nb::shape<-1>, nb::c_contig> tvec,
                                  nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> y_ct) {
    if (tvec.ndim() != 1 || y_ct.ndim() != 2) {
        printf("set_vecplay_batch: tvec must be 1D and y_ct must be 2D\n");
        return -1;
    }

    const int c = static_cast<int>(y_ct.shape(0));
    const int t = static_cast<int>(y_ct.shape(1));
    if (c < 0 || t <= 0) {
        printf("set_vecplay_batch: invalid y_ct shape (%d, %d)\n", c, t);
        return -1;
    }
    if (static_cast<int>(tvec.shape(0)) != t) {
        printf("set_vecplay_batch: tvec length mismatch (%zd vs %d)\n",
               static_cast<ssize_t>(tvec.shape(0)), t);
        return -1;
    }
    if (static_cast<int>(mech_names.size()) != c ||
        static_cast<int>(var_names.size()) != c ||
        static_cast<int>(instance_ids.size()) != c) {
        printf("set_vecplay_batch: key size mismatch (mech=%zu var=%zu id=%zu channels=%d)\n",
               mech_names.size(), var_names.size(), instance_ids.size(), c);
        return -1;
    }
    if (c == 0) {
        return 0;
    }

    const double* t_ptr = tvec.data();
    const double* y_ptr = y_ct.data();
    std::vector<double> t_values(t_ptr, t_ptr + t);
    std::vector<double> y_values(static_cast<size_t>(t));
    for (int j = 0; j < t; ++j) {
        const double tv = t_values[static_cast<size_t>(j)];
        if (!std::isfinite(tv) || tv < 0.0 || (j > 0 && tv < t_values[static_cast<size_t>(j - 1)])) {
            printf("set_vecplay_batch: invalid tvec at index %d (value=%f)\n", j, tv);
            return -1;
        }
    }

    for (int i = 0; i < c; ++i) {
        const double* row = y_ptr + static_cast<size_t>(i) * static_cast<size_t>(t);
        std::copy(row, row + t, y_values.begin());

        int rc = 0;
        if (!core_.has_vecplay(mech_names[i], var_names[i], instance_ids[i])) {
            rc = core_.add_vecplay(mech_names[i], var_names[i], instance_ids[i], t_values, y_values);
        } else {
            rc = core_.update_vecplay(mech_names[i], var_names[i], instance_ids[i], t_values, y_values);
        }
        if (rc != 0) {
            printf("set_vecplay_batch: failed at channel %d (%s.%s[%d])\n",
                   i, mech_names[i].c_str(), var_names[i].c_str(), instance_ids[i]);
            return -1;
        }
    }
    return 0;
}

int SimWrapper::compute_dw_from_mapping_v1_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> forward_rows,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> backward_rows,
    nb::ndarray<int, nb::shape<-1>, nb::c_contig> fwd_row_of_pair,
    nb::ndarray<int, nb::shape<-1>, nb::c_contig> bwd_row_of_pair,
    nb::ndarray<int, nb::shape<-1>, nb::c_contig> param_index_of_pair,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> scale_of_pair,
    int tf_use,
    double dt_fwd,
    double dt_bwd,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> out_dw,
    double t_start_fwd) {
    auto compute_dw_core = [&](const float* fwd_ptr,
                               int nf,
                               int tf,
                               const float* bwd_ptr,
                               int nb_row,
                               int tb,
                               const int* fwd_row_ptr,
                               const int* bwd_row_ptr,
                               const int* param_ptr,
                               const float* scale_ptr,
                               int p,
                               int nparam,
                               int tf_use_local,
                               double dt_fwd_local,
                               double dt_bwd_local,
                               float* out_ptr,
                               double t_start_fwd_local,
                               const char* who) -> int {
        if (p == 0 || nparam == 0) {
            std::fill_n(out_ptr, nparam, 0.0f);
            return 0;
        }
        if (tf_use_local <= 0 || tf_use_local > tf) {
            printf("%s: invalid tf_use=%d (tf=%d)\n", who, tf_use_local, tf);
            return -1;
        }
        if (!(dt_fwd_local > 0.0) || !(dt_bwd_local > 0.0)) {
            printf("%s: invalid dt_fwd/dt_bwd (%f, %f)\n", who, dt_fwd_local, dt_bwd_local);
            return -1;
        }
        if (!(t_start_fwd_local >= 0.0)) {
            printf("%s: t_start_fwd must be non-negative (%f)\n", who, t_start_fwd_local);
            return -1;
        }
        if (tb <= 0) {
            printf("%s: backward time length must be positive\n", who);
            return -1;
        }

        std::vector<double> accum(static_cast<size_t>(nparam), 0.0);
        std::vector<int> counts(static_cast<size_t>(nparam), 0);
        const double tmax_bwd = static_cast<double>(tb - 1) * dt_bwd_local;
        for (int i = 0; i < p; ++i) {
            const int rf = fwd_row_ptr[i];
            const int rb = bwd_row_ptr[i];
            const int pi = param_ptr[i];
            const float scale = scale_ptr[i];

            if (rf < 0 || rf >= nf || rb < 0 || rb >= nb_row || pi < 0 || pi >= nparam) {
                printf("%s: index out of range at pair %d\n", who, i);
                return -1;
            }

            const float* frow = fwd_ptr + static_cast<size_t>(rf) * static_cast<size_t>(tf);
            const float* brow = bwd_ptr + static_cast<size_t>(rb) * static_cast<size_t>(tb);
            double s = 0.0;
            for (int t = 0; t < tf_use_local; ++t) {
                const int rev_t = tf_use_local - 1 - t;
                const double tq = t_start_fwd_local + static_cast<double>(t) * dt_fwd_local;

                float bval = 0.0f;
                if (tq <= 0.0) {
                    bval = brow[0];
                } else if (tq >= tmax_bwd) {
                    bval = brow[tb - 1];
                } else {
                    const double pos = tq / dt_bwd_local;
                    const int i0 = static_cast<int>(std::floor(pos));
                    const int i1 = std::min(i0 + 1, tb - 1);
                    const double w = pos - static_cast<double>(i0);
                    bval = static_cast<float>(
                        (1.0 - w) * static_cast<double>(brow[i0]) +
                        w * static_cast<double>(brow[i1]));
                }
                s += static_cast<double>(bval) * static_cast<double>(-frow[rev_t]) *
                     static_cast<double>(scale);
            }
            accum[static_cast<size_t>(pi)] += s;
            counts[static_cast<size_t>(pi)] += 1;
        }

        for (int i = 0; i < nparam; ++i) {
            const int c = counts[static_cast<size_t>(i)];
            const double v = (c > 0) ? (accum[static_cast<size_t>(i)] / static_cast<double>(c)) : 0.0;
            out_ptr[i] = static_cast<float>(v);
        }
        return 0;
    };

    if (forward_rows.ndim() != 2 || backward_rows.ndim() != 2) {
        printf("compute_dw_from_mapping_v1_into: forward/backward must be 2D\n");
        return -1;
    }
    if (fwd_row_of_pair.ndim() != 1 || bwd_row_of_pair.ndim() != 1 ||
        param_index_of_pair.ndim() != 1 || scale_of_pair.ndim() != 1 || out_dw.ndim() != 1) {
        printf("compute_dw_from_mapping_v1_into: mapping/out must be 1D\n");
        return -1;
    }

    const int nf = static_cast<int>(forward_rows.shape(0));
    const int tf = static_cast<int>(forward_rows.shape(1));
    const int nb_row = static_cast<int>(backward_rows.shape(0));
    const int tb = static_cast<int>(backward_rows.shape(1));
    const int p = static_cast<int>(fwd_row_of_pair.shape(0));
    const int nparam = static_cast<int>(out_dw.shape(0));
    if (static_cast<int>(bwd_row_of_pair.shape(0)) != p ||
        static_cast<int>(param_index_of_pair.shape(0)) != p ||
        static_cast<int>(scale_of_pair.shape(0)) != p) {
        printf("compute_dw_from_mapping_v1_into: mapping size mismatch\n");
        return -1;
    }
    if (p == 0 || nparam == 0) {
        std::fill_n(out_dw.data(), nparam, 0.0f);
        return 0;
    }
    if (tf_use <= 0 || tf_use > tf) {
        printf("compute_dw_from_mapping_v1_into: invalid tf_use=%d (tf=%d)\n", tf_use, tf);
        return -1;
    }
    if (!(dt_fwd > 0.0) || !(dt_bwd > 0.0)) {
        printf("compute_dw_from_mapping_v1_into: invalid dt_fwd/dt_bwd (%f, %f)\n", dt_fwd, dt_bwd);
        return -1;
    }
    if (!(t_start_fwd >= 0.0)) {
        printf("compute_dw_from_mapping_v1_into: t_start_fwd must be non-negative (%f)\n", t_start_fwd);
        return -1;
    }
    if (tb <= 0) {
        printf("compute_dw_from_mapping_v1_into: backward time length must be positive\n");
        return -1;
    }

    const float* fwd_ptr = forward_rows.data();
    const float* bwd_ptr = backward_rows.data();
    const int* fwd_row_ptr = fwd_row_of_pair.data();
    const int* bwd_row_ptr = bwd_row_of_pair.data();
    const int* param_ptr = param_index_of_pair.data();
    const float* scale_ptr = scale_of_pair.data();
    return compute_dw_core(
        fwd_ptr,
        nf,
        tf,
        bwd_ptr,
        nb_row,
        tb,
        fwd_row_ptr,
        bwd_row_ptr,
        param_ptr,
        scale_ptr,
        p,
        nparam,
        tf_use,
        dt_fwd,
        dt_bwd,
        out_dw.data(),
        t_start_fwd,
        "compute_dw_from_mapping_v1_into");
}

double SimWrapper::compute_spk_readout_loss_and_grads_into(
    nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> spk_bct,
    nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> target_bot,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> w_oc,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> b_o,
    double k_out,
    double dt,
    nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> out_pred_bot,
    nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> out_dldspk_bct,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> out_dw_oc,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> out_db_o) {
    if (spk_bct.ndim() != 3 || target_bot.ndim() != 3 ||
        out_pred_bot.ndim() != 3 || out_dldspk_bct.ndim() != 3 ||
        w_oc.ndim() != 2 || out_dw_oc.ndim() != 2 || b_o.ndim() != 1 || out_db_o.ndim() != 1) {
        printf("compute_spk_readout_loss_and_grads_into: invalid ndim\n");
        return -1.0;
    }
    if (!(dt > 0.0)) {
        printf("compute_spk_readout_loss_and_grads_into: dt must be positive\n");
        return -1.0;
    }

    const int bsz = static_cast<int>(spk_bct.shape(0));
    const int ncell = static_cast<int>(spk_bct.shape(1));
    const int tlen = static_cast<int>(spk_bct.shape(2));
    const int bout = static_cast<int>(target_bot.shape(0));
    const int nout = static_cast<int>(target_bot.shape(1));
    const int tout = static_cast<int>(target_bot.shape(2));
    if (bsz <= 0 || ncell <= 0 || tlen <= 0 || nout <= 0) {
        printf("compute_spk_readout_loss_and_grads_into: empty shape\n");
        return -1.0;
    }
    if (bout != bsz || tout != tlen) {
        printf("compute_spk_readout_loss_and_grads_into: target shape mismatch\n");
        return -1.0;
    }
    if (static_cast<int>(w_oc.shape(0)) != nout || static_cast<int>(w_oc.shape(1)) != ncell) {
        printf("compute_spk_readout_loss_and_grads_into: w shape mismatch\n");
        return -1.0;
    }
    if (static_cast<int>(b_o.shape(0)) != nout) {
        printf("compute_spk_readout_loss_and_grads_into: b shape mismatch\n");
        return -1.0;
    }
    if (static_cast<int>(out_pred_bot.shape(0)) != bsz ||
        static_cast<int>(out_pred_bot.shape(1)) != nout ||
        static_cast<int>(out_pred_bot.shape(2)) != tlen) {
        printf("compute_spk_readout_loss_and_grads_into: out_pred shape mismatch\n");
        return -1.0;
    }
    if (static_cast<int>(out_dldspk_bct.shape(0)) != bsz ||
        static_cast<int>(out_dldspk_bct.shape(1)) != ncell ||
        static_cast<int>(out_dldspk_bct.shape(2)) != tlen) {
        printf("compute_spk_readout_loss_and_grads_into: out_dldspk shape mismatch\n");
        return -1.0;
    }
    if (static_cast<int>(out_dw_oc.shape(0)) != nout || static_cast<int>(out_dw_oc.shape(1)) != ncell) {
        printf("compute_spk_readout_loss_and_grads_into: out_dw shape mismatch\n");
        return -1.0;
    }
    if (static_cast<int>(out_db_o.shape(0)) != nout) {
        printf("compute_spk_readout_loss_and_grads_into: out_db shape mismatch\n");
        return -1.0;
    }

    const float alpha = static_cast<float>((1.0 - k_out) / dt);
    const float k = static_cast<float>(k_out);
    const float* spk = spk_bct.data();
    const float* tgt = target_bot.data();
    const float* w = w_oc.data();
    const float* b = b_o.data();
    float* pred = out_pred_bot.data();
    float* dldspk = out_dldspk_bct.data();
    float* dw = out_dw_oc.data();
    float* db = out_db_o.data();

    std::fill_n(dw, static_cast<size_t>(nout) * static_cast<size_t>(ncell), 0.0f);
    std::fill_n(db, static_cast<size_t>(nout), 0.0f);
    std::fill_n(dldspk, static_cast<size_t>(bsz) * static_cast<size_t>(ncell) * static_cast<size_t>(tlen), 0.0f);

    auto idx_bot = [nout, tlen](int ib, int io, int it) -> size_t {
        return (static_cast<size_t>(ib) * static_cast<size_t>(nout) + static_cast<size_t>(io)) * static_cast<size_t>(tlen) +
               static_cast<size_t>(it);
    };
    auto idx_bct = [ncell, tlen](int ib, int ic, int it) -> size_t {
        return (static_cast<size_t>(ib) * static_cast<size_t>(ncell) + static_cast<size_t>(ic)) * static_cast<size_t>(tlen) +
               static_cast<size_t>(it);
    };

    // Forward pass
    for (int ib = 0; ib < bsz; ++ib) {
        for (int io = 0; io < nout; ++io) {
            float p_prev = 0.0f;
            for (int it = 0; it < tlen; ++it) {
                double z = static_cast<double>(b[io]);
                for (int ic = 0; ic < ncell; ++ic) {
                    z += static_cast<double>(spk[idx_bct(ib, ic, it)]) *
                         static_cast<double>(w[static_cast<size_t>(io) * static_cast<size_t>(ncell) + static_cast<size_t>(ic)]);
                }
                const float p_cur = k * p_prev + alpha * static_cast<float>(z);
                pred[idx_bot(ib, io, it)] = p_cur;
                p_prev = p_cur;
            }
        }
    }

    // Backward through time
    double loss = 0.0;
    std::vector<float> grad_next(static_cast<size_t>(bsz) * static_cast<size_t>(nout), 0.0f);
    std::vector<float> grad_cur(static_cast<size_t>(bsz) * static_cast<size_t>(nout), 0.0f);

    for (int it = tlen - 1; it >= 0; --it) {
        // grad wrt pred_t
        for (int ib = 0; ib < bsz; ++ib) {
            for (int io = 0; io < nout; ++io) {
                const size_t bot = idx_bot(ib, io, it);
                const float err = pred[bot] - tgt[bot];
                loss += 0.5 * static_cast<double>(err) * static_cast<double>(err);
                const size_t bo = static_cast<size_t>(ib) * static_cast<size_t>(nout) + static_cast<size_t>(io);
                grad_cur[bo] = err + k * grad_next[bo];
            }
        }

        // grad wrt z_t = alpha * grad_cur
        for (int ib = 0; ib < bsz; ++ib) {
            for (int io = 0; io < nout; ++io) {
                const size_t bo = static_cast<size_t>(ib) * static_cast<size_t>(nout) + static_cast<size_t>(io);
                const float gz = alpha * grad_cur[bo];
                db[io] += gz;
                for (int ic = 0; ic < ncell; ++ic) {
                    const size_t wc = static_cast<size_t>(io) * static_cast<size_t>(ncell) + static_cast<size_t>(ic);
                    dw[wc] += gz * spk[idx_bct(ib, ic, it)];
                    dldspk[idx_bct(ib, ic, it)] += gz * w[wc];
                }
            }
        }

        std::copy(grad_cur.begin(), grad_cur.end(), grad_next.begin());
    }

    return loss;
}

// 变量handle缓存相关实现
int SimWrapper::get_variable_handle(const string& mech, const string& var, int node_or_mech_idx) {
    return core_.get_variable_handle_with_array(mech, var, node_or_mech_idx, 0);
}

int SimWrapper::get_variable_handle_with_array(const string& mech,
                                               const string& var,
                                               int node_or_mech_idx,
                                               int array_index) {
    return core_.get_variable_handle_with_array(mech,
                                                var,
                                                node_or_mech_idx,
                                                array_index);
}

int SimWrapper::get_variable_handle_with_array_by_loc(const string& mech,
                                                      const string& var,
                                                      int gid,
                                                      const string& section_name,
                                                      double loc,
                                                      int array_index,
                                                      int slot) {
    return core_.get_variable_handle_with_array_by_loc(mech,
                                                       var,
                                                       gid,
                                                       section_name,
                                                       loc,
                                                       array_index,
                                                       slot);
}

double SimWrapper::get_handle_resolved_loc(int handle) const {
    return core_.get_handle_resolved_loc(handle);
}

int SimWrapper::get_handle_resolved_segment_index(int handle) const {
    return core_.get_handle_resolved_segment_index(handle);
}

double SimWrapper::get_variable_by_handle(int handle) {
    return core_.get_variable_by_handle(handle);
}

int SimWrapper::set_dense_blocks_f32(nb::list blocks) {
    clear_dense_blocks();
    if (blocks.size() == 0) {
        return 0;
    }

    std::vector<neurong::runtime_api::learn::LearnRuntime::DenseBlockHostView> views;
    views.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        nb::handle item = blocks[i];
        nb::ndarray<float, nb::ndim<3>, nb::c_contig> arr =
            nb::cast<nb::ndarray<float, nb::ndim<3>, nb::c_contig>>(item);
        const int bn0 = static_cast<int>(arr.shape(0));
        const int bn1 = static_cast<int>(arr.shape(1));
        const int k0 = static_cast<int>(arr.shape(2));
        if (bn0 <= 0 || bn1 <= 0 || k0 <= 0) {
            printf("set_dense_blocks_f32: invalid shape in block %zu\n", i);
            clear_dense_blocks();
            return -1;
        }
        if (bn0 != bn1) {
            printf("set_dense_blocks_f32: block %zu is not square (%d vs %d)\n", i, bn0, bn1);
            clear_dense_blocks();
            return -1;
        }
        neurong::runtime_api::learn::LearnRuntime::DenseBlockHostView v;
        v.data = arr.data();
        v.bn = bn0;
        v.k_len = k0;
        views.push_back(v);
    }

    const int k_len = learn_.set_dense_blocks_f32(std::span<const neurong::runtime_api::learn::LearnRuntime::DenseBlockHostView>(
        views.data(), views.size()));
    if (k_len < 0) {
        clear_dense_blocks();
        return -1;
    }
    return k_len;
}

void SimWrapper::clear_dense_blocks() {
    learn_.clear_dense_blocks();
}

int SimWrapper::simulate_output_vs_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
    const std::vector<int>& output_v_handles,
    double tstop_ms,
    double v_init) {
    nb::gil_scoped_release release;

    if (output_vs_tn.ndim() != 2) {
        printf("simulate_output_vs_into: output_vs_tn must be 2D\n");
        return -1;
    }
    const int n_output = static_cast<int>(output_v_handles.size());
    if (n_output <= 0) {
        printf("simulate_output_vs_into: output_v_handles is empty\n");
        return -1;
    }
    if (static_cast<int>(output_vs_tn.shape(1)) != n_output) {
        printf("simulate_output_vs_into: output_vs_tn shape mismatch\n");
        return -1;
    }
    const int total_steps = static_cast<int>(output_vs_tn.shape(0)) - 1;
    if (total_steps <= 0) {
        printf("simulate_output_vs_into: total_steps must be positive\n");
        return -1;
    }
    return learn_.simulate_output_vs_into(output_vs_tn.data(),
                                         static_cast<int>(output_vs_tn.shape(0)),
                                         n_output,
                                         std::span<const int>(output_v_handles.data(), output_v_handles.size()),
                                         tstop_ms,
                                         v_init);
}

int SimWrapper::simulate_response_presence_into(
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> output,
    const std::vector<int>& output_v_handles,
    double tstop_ms,
    double v_init,
    double window_start_ms,
    double window_end_ms) {
    nb::gil_scoped_release release;
    if (output.ndim() != 1 || output.shape(0) != output_v_handles.size()) {
        printf("simulate_response_presence_into: output shape mismatch\n");
        return -1;
    }
    return learn_.simulate_response_presence_into(
        output.data(), static_cast<int>(output.shape(0)), output_v_handles,
        tstop_ms, v_init, window_start_ms, window_end_ms);
}

int SimWrapper::simulate_and_capture_mapped_signals_cached(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
    const std::vector<int>& output_v_handles,
    const std::vector<int>& pure_i_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pure_i_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> pure_i_scale,
    const std::vector<int>& didv_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didv_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> didv_scale,
    const std::vector<int>& didvpre_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didvpre_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> didvpre_scale,
    double tstop_ms,
    int k_mul,
    bool percise,
    double v_init) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("simulate_and_capture_mapped_signals_cached: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("simulate_and_capture_mapped_signals_cached: only supported on GPU\n");
        return -1;
    }
    if (k_mul <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: k_mul must be positive\n");
        return -1;
    }
    if (sim->dt <= 0.0) {
        printf("simulate_and_capture_mapped_signals_cached: invalid dt\n");
        return -1;
    }
    if (output_vs_tn.ndim() != 2) {
        printf("simulate_and_capture_mapped_signals_cached: output_vs_tn must be 2D\n");
        return -1;
    }

    const int n_output = static_cast<int>(output_v_handles.size());
    if (n_output <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: output_v_handles is empty\n");
        return -1;
    }
    if (static_cast<int>(output_vs_tn.shape(1)) != n_output) {
        printf("simulate_and_capture_mapped_signals_cached: output_vs_tn shape mismatch\n");
        return -1;
    }
    const int total_steps = static_cast<int>(output_vs_tn.shape(0)) - 1;
    if (total_steps <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: total_steps must be positive\n");
        return -1;
    }
    const int ksteps_total = static_cast<int>(total_steps / k_mul);
    if (ksteps_total <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: ksteps_total must be positive\n");
        return -1;
    }

    // Infer N from the mapping destinations (must be compatible with training code).
    int N = 0;
    for (size_t i = 0; i < pure_i_dest.shape(0); ++i) {
        const int32_t d = pure_i_dest.data()[i];
        if (d + 1 > N) {
            N = d + 1;
        }
    }
    if (N <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: inferred N is non-positive\n");
        return -1;
    }

    if (static_cast<int>(pure_i_handles.size()) != static_cast<int>(pure_i_dest.shape(0)) ||
        static_cast<int>(pure_i_handles.size()) != static_cast<int>(pure_i_scale.shape(0))) {
        printf("simulate_and_capture_mapped_signals_cached: pure_i handles/dest/scale size mismatch\n");
        return -1;
    }
    if (percise) {
        if (static_cast<int>(didv_handles.size()) != static_cast<int>(didv_dest.shape(0)) ||
            static_cast<int>(didv_handles.size()) != static_cast<int>(didv_scale.shape(0))) {
            printf("simulate_and_capture_mapped_signals_cached: didv handles/dest/scale size mismatch\n");
            return -1;
        }
        if (static_cast<int>(didvpre_handles.size()) != static_cast<int>(didvpre_dest.shape(0)) ||
            static_cast<int>(didvpre_handles.size()) != static_cast<int>(didvpre_scale.shape(0))) {
            printf("simulate_and_capture_mapped_signals_cached: didvpre handles/dest/scale size mismatch\n");
            return -1;
        }
    }

    // Validate scatter destinations to avoid memory corruption.
    for (size_t i = 0; i < pure_i_dest.shape(0); ++i) {
        const int32_t d = pure_i_dest.data()[i];
        if (d < 0 || d >= N) {
            printf("simulate_and_capture_mapped_signals_cached: pure_i_dest out of range at %zu (%d)\n", i, d);
            return -1;
        }
    }
    if (percise) {
        for (size_t i = 0; i < didv_dest.shape(0); ++i) {
            const int32_t d = didv_dest.data()[i];
            if (d < 0 || d >= N) {
                printf("simulate_and_capture_mapped_signals_cached: didv_dest out of range at %zu (%d)\n", i, d);
                return -1;
            }
        }
        for (size_t i = 0; i < didvpre_dest.shape(0); ++i) {
            const int32_t d = didvpre_dest.data()[i];
            if (d < 0 || d >= N) {
                printf("simulate_and_capture_mapped_signals_cached: didvpre_dest out of range at %zu (%d)\n", i, d);
                return -1;
            }
        }
    }

    // Basic sanity check for tstop_ms (we derive loop count from output buffer shape).
    const double tstop_from_steps = static_cast<double>(total_steps) * sim->dt;
    if (tstop_ms > 0.0 && std::abs(tstop_ms - tstop_from_steps) > (sim->dt * 0.5)) {
        printf("simulate_and_capture_mapped_signals_cached: tstop mismatch (tstop_ms=%f, steps*dt=%f)\n",
               tstop_ms, tstop_from_steps);
        return -1;
    }

    return learn_.simulate_and_capture_mapped_signals_cached(output_vs_tn.data(),
                                                            total_steps + 1,
                                                            n_output,
                                                            std::span<const int>(output_v_handles.data(), output_v_handles.size()),
                                                            std::span<const int>(pure_i_handles.data(), pure_i_handles.size()),
                                                            std::span<const int32_t>(pure_i_dest.data(), pure_i_dest.shape(0)),
                                                            std::span<const float>(pure_i_scale.data(), pure_i_scale.shape(0)),
                                                            std::span<const int>(didv_handles.data(), didv_handles.size()),
                                                            std::span<const int32_t>(didv_dest.data(), didv_dest.shape(0)),
                                                            std::span<const float>(didv_scale.data(), didv_scale.shape(0)),
                                                            std::span<const int>(didvpre_handles.data(), didvpre_handles.size()),
                                                            std::span<const int32_t>(didvpre_dest.data(), didvpre_dest.shape(0)),
                                                            std::span<const float>(didvpre_scale.data(), didvpre_scale.shape(0)),
                                                            tstop_ms,
                                                            k_mul,
                                                            percise,
                                                            v_init);

    
}

int SimWrapper::replay_compute_dw_dx_from_cached_signals_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_to,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pinput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dx_lr_it,
    double dt_ms,
    bool percise,
    double grad_scale,
    double eps,
    double grad_l2norm_threshold,
    int clip_strategy,
    int clip_check_every) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("replay_compute_dw_dx_from_cached_signals_into: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("replay_compute_dw_dx_from_cached_signals_into: only supported on GPU\n");
        return -1;
    }
    if (dLtdv_lr_to.ndim() != 2) {
        printf("replay_compute_dw_dx_from_cached_signals_into: dLtdv_lr_to must be 2D\n");
        return -1;
    }
    const int ksteps_total = static_cast<int>(dLtdv_lr_to.shape(0));
    const int n_output = static_cast<int>(poutput.shape(0));
    const int n_input = static_cast<int>(pinput.shape(0));
    const int N = static_cast<int>(pre_of_col.shape(0));
    if (ksteps_total <= 0 || n_output <= 0 || n_input <= 0 || N <= 0) {
        printf("replay_compute_dw_dx_from_cached_signals_into: empty shapes\n");
        return -1;
    }
    if (static_cast<int>(dLtdv_lr_to.shape(1)) != n_output) {
        printf("replay_compute_dw_dx_from_cached_signals_into: dLtdv_lr_to shape mismatch\n");
        return -1;
    }
    if (dw_out_n.ndim() != 1 || static_cast<int>(dw_out_n.shape(0)) != N) {
        printf("replay_compute_dw_dx_from_cached_signals_into: dw_out_n shape mismatch\n");
        return -1;
    }
    if (dx_lr_it.ndim() != 2 || static_cast<int>(dx_lr_it.shape(0)) != n_input ||
        static_cast<int>(dx_lr_it.shape(1)) != ksteps_total) {
        printf("replay_compute_dw_dx_from_cached_signals_into: dx_lr_it shape mismatch\n");
        return -1;
    }

    return learn_.replay_compute_dw_dx_from_cached_signals_into(dLtdv_lr_to.data(),
                                                               ksteps_total,
                                                               n_output,
                                                               poutput.data(),
                                                               pinput.data(),
                                                               n_input,
                                                               pre_of_col.data(),
                                                               dw_out_n.data(),
                                                               dx_lr_it.data(),
                                                               dt_ms,
                                                               percise,
                                                               grad_scale,
                                                               eps,
                                                               grad_l2norm_threshold,
                                                               clip_strategy,
                                                               clip_check_every);

}

int SimWrapper::replay_compute_dw_from_cached_signals_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_to,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
    double dt_ms,
    bool percise,
    double grad_scale,
    double eps,
    double grad_l2norm_threshold,
    int clip_strategy,
    int clip_check_every) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("replay_compute_dw_from_cached_signals_into: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("replay_compute_dw_from_cached_signals_into: only supported on GPU\n");
        return -1;
    }
    if (dLtdv_lr_to.ndim() != 2) {
        printf("replay_compute_dw_from_cached_signals_into: dLtdv_lr_to must be 2D\n");
        return -1;
    }
    if (poutput.ndim() != 1 || pre_of_col.ndim() != 1) {
        printf("replay_compute_dw_from_cached_signals_into: poutput/pre_of_col must be 1D\n");
        return -1;
    }

    const int ksteps_total = static_cast<int>(dLtdv_lr_to.shape(0));
    const int n_output = static_cast<int>(poutput.shape(0));
    const int N = static_cast<int>(pre_of_col.shape(0));
    if (ksteps_total <= 0 || n_output <= 0 || N <= 0) {
        printf("replay_compute_dw_from_cached_signals_into: empty shapes\n");
        return -1;
    }
    if (static_cast<int>(dLtdv_lr_to.shape(1)) != n_output) {
        printf("replay_compute_dw_from_cached_signals_into: dLtdv_lr_to shape mismatch\n");
        return -1;
    }
    if (dw_out_n.ndim() != 1 || static_cast<int>(dw_out_n.shape(0)) != N) {
        printf("replay_compute_dw_from_cached_signals_into: dw_out_n shape mismatch\n");
        return -1;
    }

    return learn_.replay_compute_dw_from_cached_signals_into(dLtdv_lr_to.data(),
                                                             ksteps_total,
                                                             n_output,
                                                             poutput.data(),
                                                             pre_of_col.data(),
                                                             dw_out_n.data(),
                                                             dt_ms,
                                                             percise,
                                                             grad_scale,
                                                             eps,
                                                             grad_l2norm_threshold,
                                                             clip_strategy,
                                                             clip_check_every);
}

std::vector<double> SimWrapper::get_variables_by_handles(const std::vector<int>& handles) {
    std::vector<double> values;
    values.reserve(handles.size());
    for (int handle : handles) {
        values.push_back(get_variable_by_handle(handle));
    }
    return values;
}

std::tuple<double, int> SimWrapper::debug_get_postsyn_delay_info(
    const string& mech,
    int instance_index) const {
    return core_.debug_get_postsyn_delay_info(mech, instance_index);
}

int SimWrapper::get_variables_by_handles_f32_into(const std::vector<int>& handles,
                                                  nb::ndarray<float, nb::shape<-1>, nb::c_contig> out) {
    if (out.ndim() != 1) {
        printf("get_variables_by_handles_f32_into: out must be 1D\n");
        return -1;
    }
    const int count = static_cast<int>(handles.size());
    if (static_cast<int>(out.shape(0)) != count) {
        printf("get_variables_by_handles_f32_into: out length mismatch (%zd vs %d)\n",
               static_cast<ssize_t>(out.shape(0)), count);
        return -1;
    }
    if (count == 0) {
        return 0;
    }

    float* out_cpu = out.data();
    return core_.get_variables_by_handles_f32(handles, out_cpu, count);
}

int SimWrapper::set_variable_by_handle(int handle, double value) {
    return core_.set_variable_by_handle(handle, value);
}

int SimWrapper::set_variables_by_handles(const std::vector<int>& handles, const std::vector<double>& values) {
    return core_.set_variables_by_handles(handles, values);
}

int SimWrapper::register_netstim_batch(const vector<tuple<int, int, int>>& handle_triplets,
                                       const NetStimBatchParams& params) {
    return core_.register_netstim_batch(handle_triplets,
                                        params.interval_scale,
                                        params.start_base,
                                        params.epsilon,
                                        params.number);
}

int SimWrapper::register_vecstim_batch(const vector<int>& mech_indices,
                                       const VecStimBatchParams& params) {
    return core_.register_vecstim_batch(mech_indices,
                                        params.spike_scale,
                                        params.start_base,
                                        params.epsilon,
                                        params.spike_count);
}

int SimWrapper::register_vecstim_batch_by_locs(const string& mech,
                                               const vector<int>& gids,
                                               const vector<string>& section_names,
                                               const vector<double>& locs,
                                               const VecStimBatchParams& params) {
    return core_.register_vecstim_batch_by_locs(mech,
                                                gids,
                                                section_names,
                                                locs,
                                                params.spike_scale,
                                                params.start_base,
                                                params.epsilon,
                                                params.spike_count);
}

int SimWrapper::set_vecstim_sequences(const vector<int>& mech_indices,
                                      const vector<std::int64_t>& event_offsets,
                                      const vector<double>& event_times_ms) {
    return core_.set_vecstim_sequences(mech_indices, event_offsets, event_times_ms);
}

int SimWrapper::set_input_batch_pixels(int batch_id, std::span<const double> pixels) {
    return core_.set_input_batch_pixels(batch_id, pixels);
}

// Gap Junction 管理方法实现

map<int, map<string, nb::object>> SimWrapper::get_all_gap_junctions() {
    map<int, map<string, nb::object>> result;

    const auto& gap_junctions = gap_routes_.get_all();
    for (const auto& [sid, gap_meta] : gap_junctions) {
        map<string, nb::object> gap_info;

        gap_info["src_mech"] = nb::cast(gap_meta.source.mech);
        gap_info["src_var"] = nb::cast(gap_meta.source.var);
        gap_info["src_idx"] = nb::cast(gap_meta.source.idx);
        gap_info["num_targets"] = nb::cast((int)gap_meta.targets.size());
        gap_info["is_dynamic"] = nb::cast(true);

        result[sid] = gap_info;
    }

    return result;
}

map<string, nb::object> SimWrapper::get_gap_junction(int sid) {
    map<string, nb::object> result;

    const auto* gap_meta = gap_routes_.get(sid);
    if (gap_meta == nullptr) {
        return result;
    }

    result["sid"] = nb::cast(sid);
    result["src_mech"] = nb::cast(gap_meta->source.mech);
    result["src_var"] = nb::cast(gap_meta->source.var);
    result["src_idx"] = nb::cast(gap_meta->source.idx);
    result["num_targets"] = nb::cast((int)gap_meta->targets.size());

    return result;
}

int SimWrapper::add_gap_source(int sid, const string& src_mech, const string& src_var, int src_idx) {
    return gap_routes_.add_source(src_mech, src_var, src_idx, sid, -1);
}

int SimWrapper::add_gap_target(int sid, const string& tgt_mech, const string& tgt_var, int tgt_idx) {
    return gap_routes_.add_target(sid, tgt_mech, tgt_var, tgt_idx, -1);
}

int SimWrapper::clear_all_gap_junctions() {
    return gap_routes_.clear(core_.sim());
}

int SimWrapper::get_next_available_sid() {
    return gap_routes_.get_next_available_sid();
}

int SimWrapper::add_gap_source_by_loc(int sid,
                                      const string& src_mech,
                                      const string& src_var,
                                      int gid,
                                      const string& section_name,
                                      double loc,
                                      int array_index,
                                      int slot) {
    return neurong_network::add_gap_source_by_loc(
        core_, gap_routes_, sid, src_mech, src_var, gid, section_name, loc, array_index, slot);
}

int SimWrapper::add_gap_target_by_loc(int sid,
                                      const string& tgt_mech,
                                      const string& tgt_var,
                                      int gid,
                                      const string& section_name,
                                      double loc,
                                      int array_index,
                                      int slot) {
    return neurong_network::add_gap_target_by_loc(
        core_, gap_routes_, sid, tgt_mech, tgt_var, gid, section_name, loc, array_index, slot);
}

int SimWrapper::finalize_gap_junctions() {
    return neurong_network::finalize_gap_junctions(core_, gap_routes_);
}

std::vector<double> SimWrapper::compute_transfer_impedance_dc_points(
    int gid,
    const std::string& source_section_name,
    double source_loc,
    const std::vector<std::string>& target_section_names,
    const std::vector<double>& target_locs,
    double inject_amp,
    double v_init) {
    std::vector<double> empty_result;
    if (target_section_names.size() != target_locs.size()) {
        printf("compute_transfer_impedance_dc_points: section/loc size mismatch (%zu vs %zu)\n",
               target_section_names.size(),
               target_locs.size());
        return empty_result;
    }
    if (target_section_names.empty()) {
        return empty_result;
    }
    if (sim == nullptr) {
        printf("compute_transfer_impedance_dc_points: simulator not initialized\n");
        return empty_result;
    }
    if (!(std::isfinite(inject_amp)) || std::fabs(inject_amp) <= 0.0) {
        printf("compute_transfer_impedance_dc_points: inject_amp must be non-zero\n");
        return empty_result;
    }

    const int source_node_idx_original = core_.resolve_node_index_by_loc(gid, source_section_name, source_loc);
    if (source_node_idx_original < 0) {
        printf("compute_transfer_impedance_dc_points: failed to resolve source node (gid=%d section=%s loc=%g)\n",
               gid,
               source_section_name.c_str(),
               source_loc);
        return empty_result;
    }
    const int source_node_idx = core_.resolve_node_index_internal_from_original(source_node_idx_original);
    if (source_node_idx < 0) {
        printf("compute_transfer_impedance_dc_points: failed to resolve source permuted node (node=%d)\n",
               source_node_idx_original);
        return empty_result;
    }

    std::vector<int> target_nodes;
    target_nodes.reserve(target_section_names.size());
    for (size_t i = 0; i < target_section_names.size(); ++i) {
        const auto& sec_name = target_section_names[i];
        const double loc = target_locs[i];
        const int node_idx_original = core_.resolve_node_index_by_loc(gid, sec_name, loc);
        if (node_idx_original < 0) {
            printf("compute_transfer_impedance_dc_points: failed to resolve target node at i=%zu (gid=%d section=%s loc=%g)\n",
                   i,
                   gid,
                   sec_name.c_str(),
                   loc);
            return empty_result;
        }
        const int target_node_idx = core_.resolve_node_index_internal_from_original(node_idx_original);
        if (target_node_idx < 0) {
            printf("compute_transfer_impedance_dc_points: failed to resolve target permuted node at i=%zu node=%d\n",
                   i,
                   node_idx_original);
            return empty_result;
        }
        target_nodes.push_back(target_node_idx);
    }

    std::vector<double> impedances;
    const int rc = sim->compute_transfer_impedance_dc_nodes(
        source_node_idx,
        target_nodes,
        inject_amp,
        v_init,
        impedances);
    if (rc != 0) {
        printf("compute_transfer_impedance_dc_points: matrix solve failed\n");
        return empty_result;
    }
    if (impedances.size() != target_nodes.size()) {
        printf("compute_transfer_impedance_dc_points: output size mismatch (%zu vs %zu)\n",
               impedances.size(),
               target_nodes.size());
        return empty_result;
    }
    return impedances;
}
