#ifndef SIM_WRAPPER_H
#define SIM_WRAPPER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "neuron.h"
#include "mechanism.h"
#include "simulate.h"
#include "utils.h"
#include "cuda_utils.h"
#include "coredat_structs.h"
#include "permute_order.h"
#include "read_coredat.h"
#include "cxxopts.hpp"
#include "coredat_to_innerdat.h"
#include "magic_enum/magic_enum.hpp"
#include "global_vars.h"
#include <set>
#include <netinet/in.h>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>
#include "args_to_any.hpp"
#include "runtime_api/core/SimRuntimeCore.h"
#include "runtime_api/learn/LearnRuntime.h"
#include "network/gap_route_builder.hpp"
namespace nb = nanobind;
using namespace std;

/**
 * @brief 神经元模拟器类，提供Python可调用的接口
 */
class SimWrapper {
	private:
	    neurong::runtime_api::core::SimRuntimeCore core_;
        neurong_network::GapRouteBuilder gap_routes_{};
        neurong::runtime_api::learn::LearnRuntime learn_{core_};
	    // Raw pointer view for legacy code paths in SimWrapper.cpp (especially learn/replay).
	    // Ownership lives in core_.
	    Simulate* sim = nullptr;
        int maybe_finalize_gap_routes();

	public:
    SimWrapper();
    ~SimWrapper();
    
    /**
     * @brief 设置数据路径
     * @param path 数据文件路径
     * @return 0表示成功，-1表示失败
     */
    int set_data_path(const string& path);
    
    /**
     * @brief 设置计算设备（CPU或GPU）
     * @param dev 设备名称："cpu"或"gpu"
     * @return 0表示成功，-1表示失败
     */
    int set_device(const string& dev);

    /**
     * @brief 设置模型来源（neuron: CoreNEURON导入, neurong: in-memory）
     * @param source 来源字符串
     * @return 0表示成功，-1表示失败
     */
    int set_source(const string& source);

    /**
     * @brief 获取当前模型来源
     * @return 来源字符串
     */
    string get_source() const;
    int load_mech_library(const string& path);
    std::vector<std::string> get_loaded_mech_libraries() const;
    
    /**
     * @brief 加载模型
     * @return 0表示成功，-1表示失败
     */
    int load_model();

    std::vector<std::tuple<std::string, int, int>> load_celltemplate_morphology(
        const nb::list& morph_templates,
        const nb::object& random_seed);
    std::vector<std::tuple<int, int, int>> apply_celltemplate_biophysics(
        const nb::list& biophys_templates,
        double celsius,
        const std::vector<int>& record_gids);
    std::vector<std::tuple<std::string, int, int>> get_celltemplate_gid_ranges() const;
    std::tuple<int, int> get_celltemplate_gid_range(const std::string& template_name) const;
    int load_network_real(const nb::list& connections);
    int load_network_artificial(const nb::list& connections);
    int load_network_artificial_packed(
        const std::vector<int>& pre_artificial_indices,
        const std::vector<int>& post_gids,
        const std::vector<int>& post_section_indices,
        const std::vector<double>& post_locs,
        const std::vector<double>& weights,
        const std::vector<double>& delays,
        const std::string& post_mech);
    int resolve_node_index(int gid, const string& section_name, int seg_index);
    int resolve_mech_index_by_loc(const string& mech, int gid, const string& section_name, double loc, int slot = 0);
    int resolve_segment_index_by_loc(int gid, const string& section_name, double loc);
    int resolve_node_index_by_loc(int gid, const string& section_name, double loc);
    int resolve_node_index_internal_from_original(int node_idx_original);
    int resolve_node_index_permuted_from_handle(int handle) const;
    int resolve_mech_index_permuted_from_handle(int handle) const;
    double resolve_segment_center_loc_by_loc(int gid, const string& section_name, double loc);
    double resolve_effective_loc_by_loc(const string& mech, int gid, const string& section_name, double loc);
    
    /**
     * @brief 初始化模拟
     * @param v_init 初始膜电位
     * @return 0表示成功，-1表示失败
     */
    int finitialize(double v_init);
    
    /**
     * @brief 运行模拟
     * @param tstop 模拟结束时间
     * @return 0表示成功，-1表示失败
     */
    int run(double tstop);

    /**
     * @brief 继续运行模拟（不重置t，从当前t开始推进runtime时间）
     * @param runtime 继续运行的时间长度
     * @return 0表示成功，-1表示失败
     */
    int continue_run(double runtime);

    /**
     * @brief 单步推进一次（推进一个dt）
     * @return 0表示成功，-1表示失败
     */
    int fadvance();

    /**
     * @brief 获取当前仿真时间t
     * @return 当前时间（ms）
     */
    double get_t() const;

    /**
     * @brief 将当前 recorder 缓冲区刷入IPC/HDF5（便于在step模式下读取monitor数据）
     * @return 0表示成功，-1表示失败
     */
    int flush_recorders();

    /**
     * @brief 设置是否输出spk文件
     * @param enable true表示开启，false表示关闭
     * @return 0表示成功
     */
    int set_spike_output_enabled(bool enable);

    /**
     * @brief 当前spk文件输出是否开启
     * @return true表示开启
     */
    bool is_spike_output_enabled() const;
    
    /**
     * @brief 设置排列类型
     * @param type 排列类型
     * @return 0表示成功
     */
    int set_permute_type(int type);
    
    /**
     * @brief 添加监视点
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return 监视点handle（在load_model时分配），-1表示失败
     */
    int add_monitor(const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 添加监视点（支持数组索引）
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return 监视点handle（在load_model时分配），-1表示失败
     */
    int add_monitor_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index);
    
    /**
     * @brief 设置时间步长
     * @param dt 时间步长
     * @return 0表示成功
     */
    int set_dt(double dt);
    
    /**
     * @brief 获取时间步长
     * @return 当前时间步长
     */
    double get_dt();
    
    /**
     * @brief 设置输出目录
     * @param dir 输出目录路径
     * @return 0表示成功，-1表示失败
     */
    int set_output_dir(const string& dir);
    
    /**
     * @brief 获取监视点的数据
     * @param handle 监视点handle
     * @return 包含监视点数据的向量，如果失败则返回空向量
     */
    vector<double> get_monitor_data(int handle);
    int get_monitor_sample_count(int handle);
    
    /**
     * @brief 获取多个监视点的数据
     * @param handles 监视点handle列表
     * @return 监视点handle到数据的映射
     */
    map<int, vector<double>> get_multiple_monitor_data(const vector<int>& handles);

    /**
     * @brief 批量获取多个监视点的数据（float32，写入预分配2D buffer）
     *
     * @param handles 监视点handle列表
     * @param out 预分配 float32 二维数组，shape=(len(handles), n_step)
     * @return 0表示成功，-1表示失败
     */
    int get_multiple_monitor_data_f32_into(const std::vector<int>& handles,
                                           nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> out);
    int get_multiple_monitor_data_f32_from_into(const std::vector<int>& handles,
                                                int start_step,
                                                nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> out);
    int add_window_observer(const string& mech, const string& var, int node_or_mech_idx);
    int add_window_observer_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index);
    int get_window_observer_handle(const string& mech, const string& var, int node_or_mech_idx);
    int get_window_observer_handle_with_array(const string& mech,
                                              const string& var,
                                              int node_or_mech_idx,
                                              int array_index);
    int begin_window_observer_recording(const std::vector<int>& handles, int step_count);
    int end_window_observer_recording();
    int get_window_observer_data_f32_into(const std::vector<int>& handles,
                                          nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> out);
    
    /**
     * @brief 设置变量值
     * @param val 要设置的值
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return 0表示成功，-1表示失败
     */
    int set_variable_value(double val, const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 设置变量值（支持数组索引）
     * @param val 要设置的值
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return 0表示成功，-1表示失败
     */
    int set_variable_value_with_array(double val, const string& mech, const string& var, int node_or_mech_idx, int array_index);

    /**
     * @brief 读取全局变量数据
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return 变量值
     */
    double get_variable_value(const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 读取全局变量数据（支持数组索引）
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return 变量值
     */
    double get_variable_value_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index);
    int get_group_len(int group_index);
    int prepare_vjp_from_handles(
        int group_index,
        const std::vector<int>& local_grad_handles,
        int step_count,
        const std::vector<string>& current_vjp_mechs);
    int set_spike_vjp_surrogate(const string& kind, double width_mv);
    int run_vjp(
        const std::vector<std::vector<double>>& local_grad_row_history);
    int run_vjp_from_forward_grads(nb::ndarray<double, nb::c_contig> local_grad_forward);
    int zero_learnable_grads_cpu();
    int get_learnable_grad_handle_from_variable_handle(int variable_handle);
    std::vector<int> get_learnable_grad_handles_from_variable_handles(const std::vector<int>& variable_handles);

    int create_optimizer(const string& optimizer_type);
    int optimizer_add_param(int optimizer_id, int weight_handle, int grad_handle, double impedance);
    int optimizer_add_param_batch(int optimizer_id,
                                  const std::vector<int>& weight_handles,
                                  const std::vector<int>& grad_handles,
                                  double impedance);
    int configure_optimizer(int optimizer_id, double momentum, double beta1, double beta2, double epsilon);
    int optimizer_step(int optimizer_id, double learning_rate, double record_time, double dt);
    int optimizer_add_external_grads(int optimizer_id, const std::vector<int>& weight_handles, double impedance);
    int optimizer_set_external_grads_f32(int optimizer_id,
                                         nb::ndarray<float, nb::shape<-1>, nb::c_contig> grads);
    int optimizer_clear_external_grads(int optimizer_id);
    int optimizer_reset_state(int optimizer_id);
    std::tuple<long long, std::vector<double>, std::vector<double>, double, double, double> optimizer_get_adam_state(
        int optimizer_id);
    int optimizer_set_adam_state(int optimizer_id,
                                 long long step_count,
                                 const std::vector<double>& m,
                                 const std::vector<double>& v,
                                 double beta1,
                                 double beta2,
                                 double epsilon);

    /**
     * @brief 调用自定义函数
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param args 参数列表
     * @return 变量值
     */
    double call_mech_func(const string& mech_name, const string& func_name, nb::args args);
    std::string resolve_mech_static_interface(const string& query) const;
    bool has_mech_static_interface(const string& mech_name) const;
    bool has_mech_static_property(const string& mech_name, const string& property_name) const;
    bool has_mech_static_method(const string& mech_name, const string& method_name) const;
    std::vector<std::string> list_mech_static_interfaces() const;
    std::vector<std::string> list_mech_static_properties(const string& mech_name) const;
    std::vector<std::string> list_mech_static_methods(const string& mech_name) const;
    std::string get_mech_static_property_str(const string& mech_name, const string& property_name) const;
    int set_mech_static_property_str(const string& mech_name, const string& property_name, const string& value);
    double call_mech_static(const string& mech_name, const string& method_name, nb::args args);

    // /**
    //  * @brief VecEvent播放
    //  * @param mech_idx 机制索引
    //  * @param data 数据数组
    //  * @return 0表示成功，-1表示失败
    //  */
    // int vecevent_play(int mech_idx, const vector<double>& data);
    
    /**
     * @brief 设置用户模块数量
     * @param num 模块数量
     */
    void set_user_mod_num(int num);

    /**
     * @brief 设置 recorder 的每通道缓冲容量（步数）
     * @param capacity 缓冲容量（>0）
     * @return 0表示成功，-1表示失败
     */
    int set_recorder_buffer_capacity(int capacity);
    /**
     * @brief 设置 recorder 存储精度（fp64/fp32）
     * @param dtype 支持 "fp64"/"fp32"
     * @return 0表示成功，-1表示失败
     */
    int set_recorder_storage_dtype(const string& dtype);
    /**
     * @brief 设置 recorder 记录起始时间（ms，包含该时刻）
     * @param start_time_ms 起始时间（>=0）
     * @return 0表示成功，-1表示失败
     */
    int set_recorder_start_time(double start_time_ms);
    /**
     * @brief 设置 recorder 记录步长（1=每步记录）
     * @param stride_steps 记录步长（>0）
     * @return 0表示成功，-1表示失败
     */
    int set_recorder_stride(int stride_steps);

    /**
     * @brief 获取某个gid对应的spk时间戳
     * @param gid 神经元ID
     * @return 神经元spk时间戳
     */
    vector<double>& get_spk_by_gid(int gid);
    
    /**
     * @brief 通过VarDescriptor查找对应的handle
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return handle，-1表示未找到
     */
    int get_monitor_handle(const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 通过VarDescriptor查找对应的handle（支持数组索引）
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return handle，-1表示未找到
     */
    int get_monitor_handle_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index);
    
    // VecPlay相关方法
    /**
     * @brief 添加VecPlay
     * @param mech_name 机制名称
     * @param var_name 变量名称
     * @param instance_id 实例ID
     * @param tvec 时间向量
     * @param yvec 值向量
     * @return 0表示成功，-1表示失败
     */
    int add_vecplay(const string& mech_name, const string& var_name, int instance_id,
                    const vector<double>& tvec, const vector<double>& yvec);
    
    /**
     * @brief 更新VecPlay
     * @param mech_name 机制名称
     * @param var_name 变量名称
     * @param instance_id 实例ID
     * @param new_tvec 新的时间向量
     * @param new_yvec 新的值向量
     * @return 0表示成功，-1表示失败
     */
    int update_vecplay(const string& mech_name, const string& var_name, int instance_id,
                       const vector<double>& new_tvec, const vector<double>& new_yvec);
    
    /**
     * @brief 删除VecPlay
     * @param mech_name 机制名称
     * @param var_name 变量名称
     * @param instance_id 实例ID
     * @return 0表示成功，-1表示失败
     */
    int remove_vecplay(const string& mech_name, const string& var_name, int instance_id);
    
    /**
     * @brief 检查VecPlay是否存在
     * @param mech_name 机制名称
     * @param var_name 变量名称
     * @param instance_id 实例ID
     * @return true表示存在，false表示不存在
     */
    bool has_vecplay(const string& mech_name, const string& var_name, int instance_id);
    
    /**
     * @brief 获取所有VecPlay键
     * @return 包含所有VecPlay键的向量，每个键为[mech_name, var_name, instance_id]
     */
    vector<vector<string>> get_all_vecplay_keys();

    /**
     * @brief 批量设置VecPlay（一次下发多通道时间序列）
     *
     * @param mech_names 通道机制名列表
     * @param var_names 通道变量名列表
     * @param instance_ids 通道实例ID列表
     * @param tvec 共享时间轴，shape=(T,)
     * @param y_ct 多通道数据，shape=(C,T)
     * @return 0表示成功，-1表示失败
     */
    int set_vecplay_batch(const std::vector<std::string>& mech_names,
                          const std::vector<std::string>& var_names,
                          const std::vector<int>& instance_ids,
                          nb::ndarray<double, nb::shape<-1>, nb::c_contig> tvec,
                          nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> y_ct);
    
    /**
     * @brief 获取变量的快速访问handle
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return 变量handle，-1表示失败
     */
    int get_variable_handle(const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 获取变量的快速访问handle（支持数组索引）
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return 变量handle，-1表示失败
     */
    int get_variable_handle_with_array(const string& mech,
                                       const string& var,
                                       int node_or_mech_idx,
                                       int array_index = 0);
    int get_variable_handle_with_array_by_loc(const string& mech,
                                              const string& var,
                                              int gid,
                                              const string& section_name,
                                              double loc,
                                              int array_index = 0,
                                              int slot = 0);
    double get_handle_resolved_loc(int handle) const;
    int get_handle_resolved_segment_index(int handle) const;
    
    /**
     * @brief 通过handle获取变量值
     * @param handle 变量handle
     * @return 变量值
     */
    double get_variable_by_handle(int handle);

    /**
     * @brief 批量通过handle获取变量值
     * @param handles 变量handle列表
     * @return 按输入顺序排列的变量值
     */
    std::vector<double> get_variables_by_handles(const std::vector<int>& handles);

    /**
     * @brief 批量通过handle获取变量值（float32，写入预分配buffer）
     *
     * 设计目的：降低 Python<->C++ 往返与单元素 GPU->CPU 拷贝开销，适用于频繁采样场景（如 replay 训练）。
     *
     * @param handles 变量handle列表
     * @param out 预分配的 float32 一维数组（长度必须等于 handles.size()）
     * @return 0表示成功，-1表示失败
     */
    int get_variables_by_handles_f32_into(const std::vector<int>& handles,
                                          nb::ndarray<float, nb::shape<-1>, nb::c_contig> out);
    std::tuple<double, int> debug_get_postsyn_delay_info(const string& mech, int instance_index) const;
    
    /**
     * @brief 通过handle设置变量值
     * @param handle 变量handle
     * @param value 要设置的值
     * @return 0表示成功，-1表示失败
     */
    int set_variable_by_handle(int handle, double value);

    /**
     * @brief 批量通过handle设置变量值
     * @param handles 变量handle列表
     * @param values 对应的目标值
     * @return 0表示成功，-1表示失败
     */
    int set_variables_by_handles(const std::vector<int>& handles, const std::vector<double>& values);

    /**
     * @brief 基于映射关系计算参数梯度（v1）。
     *
     * 语义：
     * - 对每个 pair：
     *     g_pair = sum_t interp(backward[row_b], t) * (-forward[row_f][rev_t]) * scale
     * - 按 param_index 聚合并做均值（mean over pairs in the same param bucket）
     *
     * @param forward_rows forward 信号矩阵，shape=(Nf, Tf)
     * @param backward_rows backward/replay 信号矩阵，shape=(Nb, Tb)
     * @param fwd_row_of_pair pair->forward row，shape=(P,)
     * @param bwd_row_of_pair pair->backward row，shape=(P,)
     * @param param_index_of_pair pair->param index，shape=(P,)
     * @param scale_of_pair 每个pair缩放系数，shape=(P,)
     * @param tf_use 实际使用的 forward 时间长度（<=Tf）
     * @param dt_fwd forward 时间步长
     * @param dt_bwd backward 时间步长
     * @param out_dw 输出梯度，shape=(Nparam,)
     * @return 0成功，-1失败
     */
    int compute_dw_from_mapping_v1_into(
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
        double t_start_fwd = 0.0);

    /**
     * @brief 计算spk读出loss及梯度（v1，MSE + leaky readout）
     *
     * pred_t = k_out * pred_{t-1} + (1-k_out)/dt * (spk_t @ w^T + b)
     * loss   = 0.5 * sum((target - pred)^2)
     *
     * 输出：
     * - out_pred_bot: 预测序列
     * - out_dldspk_bct: dL/dspk
     * - out_dw_oc, out_db_o: 读出层参数梯度
     *
     * @return loss 标量（double）
     */
    double compute_spk_readout_loss_and_grads_into(
        nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> spk_bct,
        nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> target_bot,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> w_oc,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> b_o,
        double k_out,
        double dt,
        nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> out_pred_bot,
        nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> out_dldspk_bct,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> out_dw_oc,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> out_db_o);

    // Gap Junction 管理相关方法
    /**
     * @brief 获取所有间隙连接信息
     * @return 包含所有间隙连接的映射，键为sid，值为包含源和目标信息的map
     */
    map<int, map<string, nb::object>> get_all_gap_junctions();
    
    /**
     * @brief 获取特定间隙连接信息
     * @param sid 源ID
     * @return 包含间隙连接信息的map，如果不存在则返回空map
     */
    map<string, nb::object> get_gap_junction(int sid);

    int add_gap_source(int sid, const string& src_mech, const string& src_var, int src_idx);
    int add_gap_target(int sid, const string& tgt_mech, const string& tgt_var, int tgt_idx);
    int add_gap_source_by_loc(int sid,
                              const string& src_mech,
                              const string& src_var,
                              int gid,
                              const string& section_name,
                              double loc,
                              int array_index = -1,
                              int slot = 0);
    int add_gap_target_by_loc(int sid,
                              const string& tgt_mech,
                              const string& tgt_var,
                              int gid,
                              const string& section_name,
                              double loc,
                              int array_index = -1,
                              int slot = 0);

    /**
     * @brief 清空所有间隙连接
     * @return 0表示成功
     */
    int clear_all_gap_junctions();
    int finalize_gap_junctions();
    
    /**
     * @brief 获取下一个可用的源ID
     * @return 下一个可用的sid
     */
    int get_next_available_sid();
    std::vector<double> compute_transfer_impedance_dc_points(
        int gid,
        const std::string& source_section_name,
        double source_loc,
        const std::vector<std::string>& target_section_names,
        const std::vector<double>& target_locs,
        double inject_amp,
        double v_init);

    // 输入刺激批量控制 -------------------------------------------------
    struct NetStimBatchParams {
        double interval_scale = 5.0;
        double start_base = 9.0;
        double epsilon = 0.01;
        double number = 100.0;
    };

    struct VecStimBatchParams {
        double spike_scale = 5.0;
        double start_base = 9.0;
        double epsilon = 0.01;
        int spike_count = 20;
    };

    /**
     * @brief 注册一组NetStim刺激器，返回批量更新ID
     */
    int register_netstim_batch(const vector<tuple<int, int, int>>& handle_triplets,
                               const NetStimBatchParams& params);

    /**
     * @brief 注册一组VecStim刺激器，返回批量更新ID
     */
    int register_vecstim_batch(const vector<int>& mech_indices,
                               const VecStimBatchParams& params);
    int register_vecstim_batch_by_locs(const string& mech,
                                       const vector<int>& gids,
                                       const vector<string>& section_names,
                                       const vector<double>& locs,
                                       const VecStimBatchParams& params);
    int set_vecstim_sequences(const vector<int>& mech_indices,
                              const vector<std::int64_t>& event_offsets,
                              const vector<double>& event_times_ms);

    /**
     * @brief 批量更新指定输入刺激批次的像素值
     */
    int set_input_batch_pixels(int batch_id, std::span<const double> pixels);

    // -----------------------------
    // Training/learning helper fast paths
    // -----------------------------
    // Upload dense blocks (list of float32 arrays shaped [bn, bn, K_len]) to GPU for later use.
    int set_dense_blocks_f32(nb::list blocks);
    void clear_dense_blocks();

    // Run simulation and capture only output v(t) into a preallocated time-major buffer:
    //   - output_vs_tn: (total_steps+1, n_output)
    //
    // Intended for streaming replay workflows where LR signals are captured/consumed on-device.
    int simulate_output_vs_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
        const std::vector<int>& output_v_handles,
        double tstop_ms,
        double v_init);
    int simulate_response_presence_into(
        nb::ndarray<double, nb::shape<-1>, nb::c_contig> output,
        const std::vector<int>& output_v_handles,
        double tstop_ms,
        double v_init,
        double window_start_ms,
        double window_end_ms);

    // Replay dv/dw on GPU and directly return dw/dx (Option 1: backend outputs dw/dx).
    //
    // Inputs are CPU numpy arrays; this function copies them to GPU and runs the replay loop.
    // Expected layouts:
    //   - it_lr_nt / ditdv_lr_nt / ditdvpre_lr_nt: (N, ksteps_total+1), C-contiguous
    //   - dLtdv_lr_ot: (ksteps_total, n_output), C-contiguous (time-major so each tick is contiguous)
    //   - poutput/pinput: indices into [0..N)
    //   - pre_of_col: length N, values in [-1..N) (use -1 for "no pre")
    //
    // Outputs:
    //   - dw_out_n: (N,)
    //   - dx_lr_it: (n_input, ksteps_total)
    // Run simulation and capture mapped/accumulated learning signals using GPU-side gather + scatter-add.
    // Output layout is time-major for contiguous writes:
    //   - output_vs_tn: (total_steps+1, n_output)
    //   - it_lr_tn/ditdv_lr_tn/ditdvpre_lr_tn: (ksteps_total+1, N)
    //
    // This API is intended to replace Python per-step loops when capturing training signals.
    // Single-pass training helper:
    // - Run simulation and capture mapped learning signals into *cached* GPU buffers (time-major: (ksteps_total+1, N)).
    // - Also capture output v(t) into the provided CPU buffer `output_vs_tn` (time-major: (total_steps+1, n_output)).
    //
    // This enables a "one simulation pass" workflow for objectives whose dL/dv depends on the full trajectory:
    // Python can compute dL/dv from output_vs, then call replay-from-cache without re-simulating or uploading signals.
    int simulate_and_capture_mapped_signals_cached(
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
        double v_init);

    // Replay dv/dw using cached signal buffers produced by `simulate_and_capture_mapped_signals_cached`.
    // Inputs:
    // - dLtdv_lr_ot: (ksteps_total, n_output), time-major
    // - poutput/pinput/pre_of_col: index metadata
    // Outputs:
    // - dw_out_n: (N,)
    // - dx_lr_it: (n_input, ksteps_total)
    int replay_compute_dw_dx_from_cached_signals_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_ot,
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
        int clip_check_every);

    // Replay dw only (no dx) from cached signal buffers produced by
    // `simulate_and_capture_mapped_signals_cached`.
    int replay_compute_dw_from_cached_signals_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_ot,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
        double dt_ms,
        bool percise,
        double grad_scale,
        double eps,
        double grad_l2norm_threshold,
        int clip_strategy,
        int clip_check_every);

    // Streaming replay: run simulation, maintain only a ring-buffer of LR signals on GPU,
    // and compute dw/dx without materializing full (N, ksteps_total+1) signal matrices on CPU.
    //
    // Notes:
    // - This fast path assumes inputs are provided via VecPlay (caller installs/updates vecplay beforehand).
    // - dLtdv is time-major: (ksteps_total, n_output), contiguous.
	private:

    struct OptimizerExternalGrads {
        std::vector<int> weight_handles;
        VecData<double> grads;
        double impedance = 1.0;

        explicit OptimizerExternalGrads(Mode mode) : grads(mode) {}
        OptimizerExternalGrads(OptimizerExternalGrads&&) = default;
        OptimizerExternalGrads& operator=(OptimizerExternalGrads&&) = default;
        OptimizerExternalGrads(const OptimizerExternalGrads&) = delete;
        OptimizerExternalGrads& operator=(const OptimizerExternalGrads&) = delete;
    };
    std::unordered_map<int, OptimizerExternalGrads> optimizer_external_grads_;
};

#endif // SIM_WRAPPER_H
