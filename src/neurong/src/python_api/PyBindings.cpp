#include "SimWrapper.h"
#include "neurong_shared/mech/cuda_utils.hpp"
#include <nanobind/nanobind.h>
#include <span>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include "celltemplate_builder.hpp"
#include "celltemplate_pybind_parsing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace nb = nanobind;

namespace {

nb::list pt3d_to_list(const std::vector<neurong_celltemplate::SectionPt3d>& pts) {
    nb::list out;
    for (const auto& p : pts) {
        out.append(nb::make_tuple(p.x_um, p.y_um, p.z_um, p.diam_um));
    }
    return out;
}

std::vector<neurong_celltemplate::SectionPt3d> pt3d_from_obj(nb::handle obj) {
    std::vector<neurong_celltemplate::SectionPt3d> out;
    if (obj.is_none()) {
        return out;
    }
    nb::sequence seq = nb::borrow<nb::sequence>(obj);
    const std::size_t n = nb::len(seq);
    out.reserve(n);
    for (nb::handle item : seq) {
        nb::sequence p = nb::borrow<nb::sequence>(item);
        neurong_celltemplate::SectionPt3d out_p{};
        out_p.x_um = nb::cast<float>(p[0]);
        out_p.y_um = nb::cast<float>(p[1]);
        out_p.z_um = nb::cast<float>(p[2]);
        out_p.diam_um = nb::cast<float>(p[3]);
        out.push_back(out_p);
    }
    return out;
}

std::string section_name_from_delete_arg(nb::handle section_obj) {
    using neurong_celltemplate::SectionSpec;
    if (nb::isinstance<SectionSpec>(section_obj)) {
        const auto sec = nb::cast<SectionSpec>(section_obj);
        return sec.name;
    }
    if (nb::isinstance<nb::str>(section_obj)) {
        return nb::cast<std::string>(section_obj);
    }
    throw std::runtime_error("delete_section/delete_section_subtree: 'section' must be neurong.section or str name");
}

std::unordered_map<std::string, std::size_t> build_name_to_index(
    const std::vector<neurong_celltemplate::SectionSpec>& sections) {
    std::unordered_map<std::string, std::size_t> out;
    out.reserve(sections.size());
    for (std::size_t i = 0; i < sections.size(); ++i) {
        const auto& name = sections[i].name;
        if (name.empty()) {
            throw std::runtime_error("delete_section: section name is empty");
        }
        const auto [it, inserted] = out.emplace(name, i);
        if (!inserted) {
            throw std::runtime_error("delete_section: duplicate section name: " + name);
        }
    }
    return out;
}

std::vector<neurong_celltemplate::SectionSpec> delete_section_specs(
    const std::vector<neurong_celltemplate::SectionSpec>& sections,
    const std::string& section_name,
    bool subtree) {
    const auto name_to_index = build_name_to_index(sections);
    const auto it = name_to_index.find(section_name);
    if (it == name_to_index.end()) {
        throw std::runtime_error("delete_section: section not found: " + section_name);
    }
    const std::size_t n = sections.size();
    std::vector<std::uint8_t> marked(n, 0);
    const auto root_idx = it->second;

    if (subtree) {
        std::vector<std::vector<std::size_t>> children;
        children.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& parent_name = sections[i].parent_name;
            if (parent_name.empty()) {
                continue;
            }
            const auto pit = name_to_index.find(parent_name);
            if (pit == name_to_index.end()) {
                throw std::runtime_error(
                    "delete_section_subtree: missing parent '" + parent_name + "' for section '" + sections[i].name +
                    "'");
            }
            children[pit->second].push_back(i);
        }

        std::stack<std::size_t> dfs;
        dfs.push(root_idx);
        marked[root_idx] = 1;
        while (!dfs.empty()) {
            const auto cur = dfs.top();
            dfs.pop();
            for (const auto ch : children[cur]) {
                if (marked[ch]) {
                    continue;
                }
                marked[ch] = 1;
                dfs.push(ch);
            }
        }
    } else {
        marked[root_idx] = 1;
    }

    std::vector<neurong_celltemplate::SectionSpec> out;
    out.reserve(n);
    std::unordered_set<std::string> kept_names;
    kept_names.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (marked[i]) {
            continue;
        }
        out.push_back(sections[i]);
        kept_names.emplace(sections[i].name);
    }

    for (auto& sec : out) {
        if (sec.parent_name.empty()) {
            sec.parentx = 1.0;
            continue;
        }
        if (kept_names.find(sec.parent_name) == kept_names.end()) {
            sec.parent_name.clear();
            sec.parentx = 1.0;
        }
    }

    return out;
}

nb::list sections_to_pylist(const std::vector<neurong_celltemplate::SectionSpec>& sections) {
    nb::list out;
    for (const auto& sec : sections) {
        out.append(sec);
    }
    return out;
}

}  // namespace

NB_MODULE(neurong, m) {
    m.def(
        "warmup_gpu",
        []() {
            void* ptr = nullptr;
            gpu_mem_allocate(&ptr, 1);
            gpu_mem_free(&ptr);
        },
        nb::call_guard<nb::gil_scoped_release>(),
        "Initialize the current CUDA device context without loading a model.");
    m.doc() = "神经元模拟器Python绑定";

    using neurong_celltemplate::SectionSpec;

    nb::class_<SectionSpec>(m, "section")
        .def(nb::init<const std::string&, const std::string&>(), nb::arg("name"), nb::arg("label"))
        .def_rw("name", &SectionSpec::name)
        .def(
            "connect",
            [](SectionSpec& child, nb::handle parent, double parentx) {
                if (parentx < 0.0 || parentx > 1.0) {
                    throw std::runtime_error("section.connect: parentx must be in [0, 1]");
                }
                if (parent.is_none()) {
                    child.parent_name.clear();
                } else if (nb::isinstance<SectionSpec>(parent)) {
                    const auto& p = nb::cast<const SectionSpec&>(parent);
                    child.parent_name = p.name;
                } else {
                    child.parent_name = nb::cast<std::string>(parent);
                }
                child.parentx = parentx;
            },
            "Connect this section to a parent section (or None) at parentx.",
            nb::arg("parent").none(),
            nb::arg("parentx") = 1.0)
        .def_prop_ro(
            "parent",
            [](const SectionSpec& sec) -> nb::object {
                if (sec.parent_name.empty()) {
                    return nb::none();
                }
                return nb::str(sec.parent_name.c_str());
            },
            "Read-only parent section name. Use section.connect() to change topology.")
        .def_prop_ro(
            "parentx",
            [](const SectionSpec& sec) { return sec.parentx; },
            "Read-only parent attach position. Use section.connect() to change topology.")
        .def_prop_ro(
            "label",
            [](const SectionSpec& sec) -> nb::object { return nb::str(sec.label.c_str()); })
        .def_rw("nseg", &SectionSpec::nseg)
        .def_rw("L_um", &SectionSpec::L_um)
        .def_rw("diam_um", &SectionSpec::diam_um)
        .def_prop_rw(
            "pt3d",
            [](const SectionSpec& sec) { return pt3d_to_list(sec.pt3d); },
            [](SectionSpec& sec, nb::handle obj) { sec.pt3d = pt3d_from_obj(obj); },
            nb::arg("pt3d").none())
        .def("__repr__",
             [](const SectionSpec& sec) {
                 return "<neurong.section name='" + sec.name + "'>";
             });

    m.def(
        "load_swc_sections",
        [](const std::string& swc_file) {
            const auto sections = neurong_celltemplate::load_swc_sections(swc_file);
            nb::list out;
            for (const auto& sec : sections) {
                out.append(sec);
            }
            return out;
        },
        "Load an SWC file and return a list of neurong.section objects (topology + geometry).",
        nb::arg("swc_file"));

    m.def(
        "load_asc_sections",
        [](const std::string& asc_file) {
            const auto sections = neurong_celltemplate::load_asc_sections(asc_file);
            nb::list out;
            for (const auto& sec : sections) {
                out.append(sec);
            }
            return out;
        },
        "Load a Neurolucida ASC file and return a list of neurong.section objects (topology + geometry).",
        nb::arg("asc_file"));

    m.def(
        "sections_with_label",
        [](const nb::list& sections, const std::string& label) {
            const auto section_specs = neurong_pybind::parse_sections(sections);
            const auto out = neurong_celltemplate::sections_with_label(section_specs, label);
            return sections_to_pylist(out);
        },
        "Return sections that match the given label name. Unknown labels return an empty list.",
        nb::arg("sections"),
        nb::arg("label"));

    m.def(
        "section_distance_um",
        [](const nb::list& sections,
           const std::string& section_name1,
           double loc1,
           const std::string& section_name2,
           double loc2) {
            const auto section_specs = neurong_pybind::parse_sections(sections);
            const auto layout = neurong_celltemplate::build_section_distance_layout(section_specs);
            const double d_um = neurong_morph::distance_between_locs_um(
                layout, section_name1, loc1, section_name2, loc2);
            if (!std::isfinite(d_um)) {
                throw std::runtime_error(
                    "section_distance_um failed: invalid section name/loc or inconsistent topology");
            }
            return d_um;
        },
        "Compute NEURON-like topology distance (um) between two section locations on a section list.",
        nb::arg("sections"),
        nb::arg("section_name1"),
        nb::arg("loc1"),
        nb::arg("section_name2"),
        nb::arg("loc2"));

    m.def(
        "section_diam_um_by_loc",
        [](const nb::list& sections, const std::string& section_name, double loc) {
            const auto section_specs = neurong_pybind::parse_sections(sections);
            const auto layout = neurong_celltemplate::build_section_distance_layout(section_specs);
            double diam_um = 0.0;
            if (!neurong_morph::resolve_diam_um_by_loc(layout, section_name, loc, &diam_um) || !std::isfinite(diam_um)) {
                throw std::runtime_error(
                    "section_diam_um_by_loc failed: invalid section name/loc or inconsistent geometry");
            }
            return diam_um;
        },
        "Resolve section diameter (um) at loc using NEURON-like pt3d/diam semantics.",
        nb::arg("sections"),
        nb::arg("section_name"),
        nb::arg("loc"));

    m.def(
        "section_segment_index_by_loc",
        [](const nb::list& sections, const std::string& section_name, double loc) {
            const auto section_specs = neurong_pybind::parse_sections(sections);
            const auto layout = neurong_celltemplate::build_section_distance_layout(section_specs);
            std::int32_t seg_idx = -1;
            if (!neurong_morph::resolve_segment_index_by_loc(layout, section_name, loc, &seg_idx)) {
                throw std::runtime_error(
                    "section_segment_index_by_loc failed: invalid section name/loc or section nseg");
            }
            return static_cast<int>(seg_idx);
        },
        "Resolve segment index for (section_name, loc) on a section list (maps to [0, nseg-1]).",
        nb::arg("sections"),
        nb::arg("section_name"),
        nb::arg("loc"));

    m.def(
        "delete_section",
        [](const nb::list& sections, nb::handle section) {
            const auto section_specs = neurong_pybind::parse_sections(sections);
            const auto section_name = section_name_from_delete_arg(section);
            const auto out = delete_section_specs(section_specs, section_name, false);
            return sections_to_pylist(out);
        },
        "Delete a single section (NEURON delete_section semantics): descendants are preserved and become roots.",
        nb::arg("sections"),
        nb::arg("section"));

    m.def(
        "delete_section_subtree",
        [](const nb::list& sections, nb::handle section) {
            const auto section_specs = neurong_pybind::parse_sections(sections);
            const auto section_name = section_name_from_delete_arg(section);
            const auto out = delete_section_specs(section_specs, section_name, true);
            return sections_to_pylist(out);
        },
        "Delete a section subtree (target section and all descendants).",
        nb::arg("sections"),
        nb::arg("section"));

    m.def(
        "delete_section_label",
        [](const nb::list& sections, const std::string& label) {
            auto current = neurong_pybind::parse_sections(sections);
            while (true) {
                const auto it = std::find_if(
                    current.begin(), current.end(), [&](const neurong_celltemplate::SectionSpec& sec) {
                        return sec.label == label;
                    });
                if (it == current.end()) {
                    break;
                }
                current = delete_section_specs(current, it->name, false);
            }
            return sections_to_pylist(current);
        },
        "Delete all sections with the given label. Each delete uses NEURON delete_section semantics.",
        nb::arg("sections"),
        nb::arg("label"));

    nb::class_<SimWrapper>(m, "Sim")
        .def(nb::init<>())
        .def("set_data_path", &SimWrapper::set_data_path, "设置数据路径")
        .def("set_device", &SimWrapper::set_device, "设置计算设备")
        .def("set_source", &SimWrapper::set_source, "设置模型来源：neuron(coreneuron) 或 neurong(in-memory)")
        .def("get_source", &SimWrapper::get_source, "获取当前模型来源")
        .def("load_mech_library", &SimWrapper::load_mech_library, "加载外部 mech 动态库", nb::arg("path"))
        .def("get_loaded_mech_libraries", &SimWrapper::get_loaded_mech_libraries, "获取已加载 mech 动态库路径")
        .def("load_model", &SimWrapper::load_model, "加载模型")
        .def("load_celltemplate_morphology",
             &SimWrapper::load_celltemplate_morphology,
             "加载多个celltemplate morphology到同一个Sim（每项仅接受name/num_cells/sections）",
             nb::arg("morph_templates"),
             nb::arg("random_seed") = nb::none())
        .def("apply_celltemplate_biophysics",
             &SimWrapper::apply_celltemplate_biophysics,
             "将biophysical语义应用到已加载morphology（每项接受template_name或gid_selector；也允许同时提供template_name+gid_selector='all'表示template内all；"
             "同时支持同一morphology template内按gid_selector分段差异化biophys；v_init必填，其他包含inserts/ion_species/specificCapacitance/resistivity/iclamp；"
             "iclamp子项仅接受cells/section/x/delay/dur/amp，cells支持'all'/单个int/int列表）",
             nb::arg("biophys_templates"),
             nb::arg("celsius"),
             nb::arg("record_gids") = std::vector<int>{})
        .def("get_celltemplate_gid_ranges",
             &SimWrapper::get_celltemplate_gid_ranges,
             "获取已加载celltemplate的gid范围，返回[(template_name, begin, end_exclusive), ...]")
        .def("get_celltemplate_gid_range",
             &SimWrapper::get_celltemplate_gid_range,
             "按template_name获取gid范围，返回(begin, end_exclusive)",
             nb::arg("template_name"))
        .def("load_network_real",
             &SimWrapper::load_network_real,
             "加载real pre连接：字段 [id]+preGid+[preSection+preSectionLocation]+threshold+post...；"
             "若connections中存在artificial pre则报错；id可省略（按顺序自动分配）",
             nb::arg("connections"))
        .def("load_network_artificial",
             &SimWrapper::load_network_artificial,
             "加载artificial pre连接：字段 [id]+[preGid]+preMech+[preSection+preSectionLocation]+post...；"
             "不允许threshold；若省略preGid则默认postGid，若省略preSection/preSectionLocation则默认post端位置；"
             "若connections中存在real pre则报错；id可省略（按顺序自动分配）",
             nb::arg("connections"))
        .def("load_network_artificial_packed",
             &SimWrapper::load_network_artificial_packed,
             "使用预解析artificial实例和primitive数组批量加载连接",
             nb::arg("pre_artificial_indices"),
             nb::arg("post_gids"),
             nb::arg("post_section_indices"),
             nb::arg("post_locs"),
             nb::arg("weights"),
             nb::arg("delays"),
             nb::arg("post_mech"))
        .def("resolve_node_index",
             &SimWrapper::resolve_node_index,
             "根据(gid, section_name, seg_index)解析原始node索引")
        .def("resolve_mech_index_by_loc",
             &SimWrapper::resolve_mech_index_by_loc,
             "根据(gid, section_name, loc, slot)解析机制实例索引",
             nb::arg("mech"),
             nb::arg("gid"),
             nb::arg("section_name"),
             nb::arg("loc"),
             nb::arg("slot") = 0)
        .def("resolve_segment_index_by_loc",
             &SimWrapper::resolve_segment_index_by_loc,
             "根据(gid, section_name, loc)解析segment索引（范围[0, nseg-1]）")
        .def("resolve_node_index_internal_from_original",
             &SimWrapper::resolve_node_index_internal_from_original,
             "将原始node索引映射到组内内部索引",
             nb::arg("node_idx_original"))
        .def("resolve_node_index_permuted_from_handle",
             &SimWrapper::resolve_node_index_permuted_from_handle,
             "将变量handle解析到组内内部(permuted)node索引",
             nb::arg("handle"))
        .def("resolve_mech_index_permuted_from_handle",
             &SimWrapper::resolve_mech_index_permuted_from_handle,
             "将机制变量handle解析到运行时内部(permuted)mech索引",
             nb::arg("handle"))
        .def("resolve_segment_center_loc_by_loc",
             &SimWrapper::resolve_segment_center_loc_by_loc,
             "根据(gid, section_name, loc)解析到segment后返回该segment中心loc")
        .def("resolve_effective_loc_by_loc",
             &SimWrapper::resolve_effective_loc_by_loc,
             "按机制语义解析有效loc：non-global返回segment中心loc，global返回clamp后的loc")
        .def("resolve_node_index_by_loc",
             &SimWrapper::resolve_node_index_by_loc,
             "根据(gid, section_name, loc)按NEURON语义映射到段并解析原始node索引")
        .def("finitialize", &SimWrapper::finitialize, "初始化模拟")
        .def("run", &SimWrapper::run, "运行模拟")
        .def("continue_run", &SimWrapper::continue_run, "继续运行模拟（不重置t）")
        .def("fadvance", &SimWrapper::fadvance, "单步推进一次（推进一个dt）")
        .def("get_t", &SimWrapper::get_t, "获取当前仿真时间t")
        .def("flush_recorders", &SimWrapper::flush_recorders, "将record缓冲区刷入输出缓冲（step模式可用）")
        .def("set_spike_output_enabled", &SimWrapper::set_spike_output_enabled, "设置是否输出spk文件")
        .def("is_spike_output_enabled", &SimWrapper::is_spike_output_enabled, "获取spk文件输出状态")
        .def("set_permute_type", &SimWrapper::set_permute_type, "设置排列类型")
        .def("add_monitor", &SimWrapper::add_monitor, "添加监视点")
        .def("add_monitor_with_array", &SimWrapper::add_monitor_with_array, "添加监视点（支持数组索引）")
        .def("set_dt", &SimWrapper::set_dt, "设置时间步长")
        .def("get_dt", &SimWrapper::get_dt, "获取时间步长")
        .def("set_output_dir", &SimWrapper::set_output_dir, "设置输出目录") 
        .def("get_monitor_data", &SimWrapper::get_monitor_data, "获取监视点数据")
        .def("get_monitor_sample_count", &SimWrapper::get_monitor_sample_count, "获取监视点已记录样本数")
        .def("get_multiple_monitor_data", &SimWrapper::get_multiple_monitor_data, "获取多个监视点的数据")
        .def("get_multiple_monitor_data_f32_into",
             &SimWrapper::get_multiple_monitor_data_f32_into,
             "批量获取多个监视点的数据（float32，写入预分配2D buffer）")
        .def("get_multiple_monitor_data_f32_from_into",
             &SimWrapper::get_multiple_monitor_data_f32_from_into,
             "批量获取多个监视点从指定样本offset开始的数据（float32，写入预分配2D buffer）")
        .def("add_window_observer", &SimWrapper::add_window_observer, "添加训练窗口观测点")
        .def("add_window_observer_with_array", &SimWrapper::add_window_observer_with_array, "添加训练窗口观测点（支持数组索引）")
        .def("get_window_observer_handle", &SimWrapper::get_window_observer_handle, "获取训练窗口观测点handle")
        .def("get_window_observer_handle_with_array",
             &SimWrapper::get_window_observer_handle_with_array,
             "获取训练窗口观测点handle（支持数组索引）")
        .def("begin_window_observer_recording",
             &SimWrapper::begin_window_observer_recording,
             "开始当前训练窗口观测")
        .def("end_window_observer_recording",
             &SimWrapper::end_window_observer_recording,
             "结束当前训练窗口观测")
        .def("get_window_observer_data_f32_into",
             &SimWrapper::get_window_observer_data_f32_into,
             "读取当前训练窗口观测数据（float32，写入预分配2D buffer）")
        .def("set_variable_value", &SimWrapper::set_variable_value, "设置变量值")
        .def("set_variable_value_with_array", &SimWrapper::set_variable_value_with_array, "设置变量值（支持数组索引）")
        .def("get_variable_value", &SimWrapper::get_variable_value, "获取变量值")
        .def("get_variable_value_with_array", &SimWrapper::get_variable_value_with_array, "获取变量值（支持数组索引）")
        .def("get_group_len",
             &SimWrapper::get_group_len,
             "返回指定group的内部节点数",
             nb::arg("group_index"))
        .def("prepare_vjp_from_handles",
             &SimWrapper::prepare_vjp_from_handles,
             "准备当前VJP反传计划；Python传入电压变量handles，后端解析内部VJP rows",
             nb::arg("group_index"),
             nb::arg("local_grad_handles"),
             nb::arg("step_count"),
             nb::arg("current_vjp_mechs"))
        .def("set_spike_vjp_surrogate",
             &SimWrapper::set_spike_vjp_surrogate,
             "设置已注册的 Spike VJP surrogate",
             nb::arg("kind"),
             nb::arg("width_mv"))
        .def("run_vjp",
             &SimWrapper::run_vjp,
             "运行最近一次prepare_vjp准备好的VJP反传",
             nb::arg("local_grad_row_history"))
        .def("run_vjp_from_forward_grads",
             &SimWrapper::run_vjp_from_forward_grads,
             "用forward-order ndarray梯度运行最近一次prepare_vjp准备好的VJP反传",
             nb::arg("local_grad_forward"))
        .def("zero_learnable_grads_cpu",
             &SimWrapper::zero_learnable_grads_cpu,
             "清空当前模型中所有CPU learnable parameter gradient buffer")
        .def("get_learnable_grad_handle_from_variable_handle",
             &SimWrapper::get_learnable_grad_handle_from_variable_handle,
             "从变量handle派生对应learnable grad handle",
             nb::arg("variable_handle"))
        .def("get_learnable_grad_handles_from_variable_handles",
             &SimWrapper::get_learnable_grad_handles_from_variable_handles,
             "批量从变量handles派生对应learnable grad handles",
             nb::arg("variable_handles"))
        .def("create_optimizer", &SimWrapper::create_optimizer, "创建优化器并返回ID")
        .def("optimizer_add_param", &SimWrapper::optimizer_add_param, "向优化器注册参数")
        .def("optimizer_add_param_batch", &SimWrapper::optimizer_add_param_batch, "批量向优化器注册参数")
        .def("configure_optimizer", &SimWrapper::configure_optimizer, "配置优化器超参数",
             nb::arg("optimizer_id"), nb::arg("momentum") = 0.9, nb::arg("beta1") = 0.9,
             nb::arg("beta2") = 0.999, nb::arg("epsilon") = 1e-8)
        .def("optimizer_step", &SimWrapper::optimizer_step, "执行优化器更新")
        .def("optimizer_add_external_grads",
             &SimWrapper::optimizer_add_external_grads,
             "向优化器注册外部梯度槽位（每个weight_handle对应一个独立参数；batch_size=1）")
        .def("optimizer_set_external_grads_f32",
             &SimWrapper::optimizer_set_external_grads_f32,
             "设置外部梯度（float32数组，自动转换为double并同步到GPU）")
        .def("optimizer_clear_external_grads",
             &SimWrapper::optimizer_clear_external_grads,
             "清空外部梯度（置0，并同步到GPU）")
        .def("optimizer_reset_state", &SimWrapper::optimizer_reset_state, "重置优化器内部状态（例如Adam动量）")
        .def("optimizer_get_adam_state",
             &SimWrapper::optimizer_get_adam_state,
             "获取Adam状态 (step, m, v, beta1, beta2, epsilon)")
        .def("optimizer_set_adam_state",
             &SimWrapper::optimizer_set_adam_state,
             "设置Adam状态 (optimizer_id, step, m, v, beta1, beta2, epsilon)")
        .def("call_mech_func", &SimWrapper::call_mech_func, "调用自定义函数")
        .def("resolve_mech_static_interface", &SimWrapper::resolve_mech_static_interface, "解析mechanism静态接口名称")
        .def("has_mech_static_interface", &SimWrapper::has_mech_static_interface, "检查mechanism静态接口是否存在")
        .def("has_mech_static_property", &SimWrapper::has_mech_static_property, "检查mechanism静态属性是否存在")
        .def("has_mech_static_method", &SimWrapper::has_mech_static_method, "检查mechanism静态方法是否存在")
        .def("list_mech_static_interfaces", &SimWrapper::list_mech_static_interfaces, "列出mechanism静态接口")
        .def("list_mech_static_properties", &SimWrapper::list_mech_static_properties, "列出mechanism静态属性")
        .def("list_mech_static_methods", &SimWrapper::list_mech_static_methods, "列出mechanism静态方法")
        .def("get_mech_static_property_str", &SimWrapper::get_mech_static_property_str, "读取mechanism静态字符串属性")
        .def("set_mech_static_property_str", &SimWrapper::set_mech_static_property_str, "设置mechanism静态字符串属性")
        .def("call_mech_static", &SimWrapper::call_mech_static, "调用mechanism静态方法")
        // .def("vecevent_play", &SimWrapper::vecevent_play, "VecEvent播放")
        .def("set_user_mod_num", &SimWrapper::set_user_mod_num, "设置用户模块数量")
        .def("set_recorder_buffer_capacity",
             &SimWrapper::set_recorder_buffer_capacity,
             "设置recorder每通道缓冲容量（步数）",
             nb::arg("capacity"))
        .def("set_recorder_storage_dtype",
             &SimWrapper::set_recorder_storage_dtype,
             "设置recorder存储精度（fp64/fp32）",
             nb::arg("dtype"))
        .def("set_recorder_start_time",
             &SimWrapper::set_recorder_start_time,
             "设置recorder记录起始时间（ms）",
             nb::arg("start_time_ms"))
        .def("set_recorder_stride",
             &SimWrapper::set_recorder_stride,
             "设置recorder记录步长（1=每步）",
             nb::arg("stride_steps"))
        .def("get_spk_by_gid", &SimWrapper::get_spk_by_gid, "获取某个gid对应的spk时间戳",
            nb::rv_policy::reference_internal)
        .def("get_monitor_handle", &SimWrapper::get_monitor_handle, "获取监视点handle")
        .def("get_monitor_handle_with_array", &SimWrapper::get_monitor_handle_with_array, "获取监视点handle（支持数组索引）")
        // VecPlay相关方法
        .def("add_vecplay", &SimWrapper::add_vecplay, "添加VecPlay")
        .def("update_vecplay", &SimWrapper::update_vecplay, "更新VecPlay")
        .def("remove_vecplay", &SimWrapper::remove_vecplay, "删除VecPlay")
        .def("has_vecplay", &SimWrapper::has_vecplay, "检查VecPlay是否存在")
        .def("set_vecplay_batch",
             &SimWrapper::set_vecplay_batch,
             "批量设置VecPlay（共享tvec + 多通道y矩阵）",
             nb::arg("mech_names"),
             nb::arg("var_names"),
             nb::arg("instance_ids"),
             nb::arg("tvec"),
             nb::arg("y_ct"))
        .def("get_all_vecplay_keys", &SimWrapper::get_all_vecplay_keys, "获取所有VecPlay键")
        // 变量handle相关方法
        .def("get_variable_handle",
             &SimWrapper::get_variable_handle,
             "获取变量的快速访问handle（等价于array_index=0）",
             nb::arg("mech"),
             nb::arg("var"),
             nb::arg("index"))
        .def("get_variable_handle_with_array",
             &SimWrapper::get_variable_handle_with_array,
             "获取变量的快速访问handle（global使用node索引，非global机制使用机制实例索引）",
             nb::arg("mech"),
             nb::arg("var"),
             nb::arg("index"),
             nb::arg("array_index") = 0)
        .def("get_variable_handle_with_array_by_loc",
             &SimWrapper::get_variable_handle_with_array_by_loc,
             "按(gid, section, loc, slot, array_index)解析并缓存变量handle",
             nb::arg("mech"),
             nb::arg("var"),
             nb::arg("gid"),
             nb::arg("section_name"),
             nb::arg("loc"),
             nb::arg("array_index") = 0,
             nb::arg("slot") = 0)
        .def("get_handle_resolved_loc",
             &SimWrapper::get_handle_resolved_loc,
             "查询by-loc handle的resolved loc（non-global为segment中心loc）",
             nb::arg("handle"))
        .def("get_handle_resolved_segment_index",
             &SimWrapper::get_handle_resolved_segment_index,
             "查询by-loc handle的segment index（global/未知返回-1）",
             nb::arg("handle"))
        .def("get_variable_by_handle", &SimWrapper::get_variable_by_handle, "通过handle获取变量值")
        .def("get_variables_by_handles", &SimWrapper::get_variables_by_handles, "批量通过handle获取变量值")
        .def("get_variables_by_handles_f32_into", &SimWrapper::get_variables_by_handles_f32_into, "批量通过handle获取变量值（float32，写入预分配buffer）")
        .def("debug_get_postsyn_delay_info",
             &SimWrapper::debug_get_postsyn_delay_info,
             "测试/调试用：读取postsyn实例的(delay, delay_step)",
             nb::arg("mech"),
             nb::arg("instance_index"))
        .def("set_variable_by_handle", &SimWrapper::set_variable_by_handle, "通过handle设置变量值")
        .def("set_variables_by_handles", &SimWrapper::set_variables_by_handles, "批量通过handle设置变量值")
        .def("compute_dw_from_mapping_v1_into",
             &SimWrapper::compute_dw_from_mapping_v1_into,
             "基于映射计算参数梯度（v1，含线性插值+时间反转+按param均值归约）",
             nb::arg("forward_rows"),
             nb::arg("backward_rows"),
             nb::arg("fwd_row_of_pair"),
             nb::arg("bwd_row_of_pair"),
             nb::arg("param_index_of_pair"),
             nb::arg("scale_of_pair"),
             nb::arg("tf_use"),
             nb::arg("dt_fwd"),
             nb::arg("dt_bwd"),
             nb::arg("out_dw"),
             nb::arg("t_start_fwd") = 0.0)
        .def("compute_spk_readout_loss_and_grads_into",
             &SimWrapper::compute_spk_readout_loss_and_grads_into,
             "计算spk读出loss及梯度（MSE + leaky readout）",
             nb::arg("spk_bct"),
             nb::arg("target_bot"),
             nb::arg("w_oc"),
             nb::arg("b_o"),
             nb::arg("k_out"),
             nb::arg("dt"),
             nb::arg("out_pred_bot"),
             nb::arg("out_dldspk_bct"),
             nb::arg("out_dw_oc"),
             nb::arg("out_db_o"))
        // Training/learning helper fast paths
        .def("set_dense_blocks_f32", &SimWrapper::set_dense_blocks_f32, "上传分块矩阵到GPU（可选）")
        .def("clear_dense_blocks", &SimWrapper::clear_dense_blocks, "清理分块矩阵GPU缓存")
        .def("simulate_output_vs_into",
             &SimWrapper::simulate_output_vs_into,
             "运行仿真并采集输出v(t)（写入预分配buffer）",
             nb::arg("output_vs_tn"),
             nb::arg("output_v_handles"),
             nb::arg("tstop_ms"),
             nb::arg("v_init"))
        .def("simulate_response_presence_into",
             &SimWrapper::simulate_response_presence_into,
             "Run simulation and reduce Growing response-presence on GPU",
             nb::arg("output"),
             nb::arg("output_v_handles"),
             nb::arg("tstop_ms"),
             nb::arg("v_init"),
             nb::arg("window_start_ms"),
             nb::arg("window_end_ms"))
        .def("simulate_and_capture_mapped_signals_cached",
             &SimWrapper::simulate_and_capture_mapped_signals_cached,
             "运行仿真并采集训练信号（缓存到GPU；仅输出output_vs）",
             nb::arg("output_vs_tn"),
             nb::arg("output_v_handles"),
             nb::arg("pure_i_handles"),
             nb::arg("pure_i_dest"),
             nb::arg("pure_i_scale"),
             nb::arg("didv_handles"),
             nb::arg("didv_dest"),
             nb::arg("didv_scale"),
             nb::arg("didvpre_handles"),
             nb::arg("didvpre_dest"),
             nb::arg("didvpre_scale"),
             nb::arg("tstop_ms"),
             nb::arg("k_mul"),
             nb::arg("percise"),
             nb::arg("v_init"))
        .def("replay_compute_dw_dx_from_cached_signals_into",
             &SimWrapper::replay_compute_dw_dx_from_cached_signals_into,
             "Replay dv/dw（使用GPU缓存的signal矩阵），直接输出 dw/dx（写入预分配buffer）",
             nb::arg("dLtdv_lr_ot"),
             nb::arg("poutput"),
             nb::arg("pinput"),
             nb::arg("pre_of_col"),
             nb::arg("dw_out_n"),
             nb::arg("dx_lr_it"),
             nb::arg("dt_ms"),
             nb::arg("percise"),
             nb::arg("grad_scale") = 1.0,
             nb::arg("eps") = 1e-6,
             nb::arg("grad_l2norm_threshold") = 1e6,
             nb::arg("clip_strategy") = 1,
             nb::arg("clip_check_every") = 1)
        .def("replay_compute_dw_from_cached_signals_into",
             &SimWrapper::replay_compute_dw_from_cached_signals_into,
             "Replay dv/dw（使用GPU缓存的signal矩阵），只输出 dw（不计算 dx）",
             nb::arg("dLtdv_lr_ot"),
             nb::arg("poutput"),
             nb::arg("pre_of_col"),
             nb::arg("dw_out_n"),
             nb::arg("dt_ms"),
             nb::arg("percise"),
             nb::arg("grad_scale") = 1.0,
             nb::arg("eps") = 1e-6,
             nb::arg("grad_l2norm_threshold") = 1e6,
             nb::arg("clip_strategy") = 1,
             nb::arg("clip_check_every") = 1)
        // Gap Junction相关方法
        .def("get_all_gap_junctions", &SimWrapper::get_all_gap_junctions, "获取所有间隙连接信息")
        .def("get_gap_junction", &SimWrapper::get_gap_junction, "获取特定间隙连接信息")
        .def("add_gap_source",
             &SimWrapper::add_gap_source,
             "添加间隙连接源（直接传backend row/index）",
             nb::arg("sid"),
             nb::arg("src_mech"),
             nb::arg("src_var"),
             nb::arg("src_idx"))
        .def("add_gap_target",
             &SimWrapper::add_gap_target,
             "添加间隙连接目标（直接传backend row/index）",
             nb::arg("sid"),
             nb::arg("tgt_mech"),
             nb::arg("tgt_var"),
             nb::arg("tgt_idx"))
        .def("add_gap_source_by_loc",
             &SimWrapper::add_gap_source_by_loc,
             "添加间隙连接源（前端仅传gid/section/loc，后端解析并缓存）",
             nb::arg("sid"),
             nb::arg("src_mech"),
             nb::arg("src_var"),
             nb::arg("gid"),
             nb::arg("section_name"),
             nb::arg("loc"),
             nb::arg("array_index") = -1,
             nb::arg("slot") = 0)
        .def("add_gap_target_by_loc",
             &SimWrapper::add_gap_target_by_loc,
             "添加间隙连接目标（前端仅传gid/section/loc，后端解析并缓存）",
             nb::arg("sid"),
             nb::arg("tgt_mech"),
             nb::arg("tgt_var"),
             nb::arg("gid"),
             nb::arg("section_name"),
             nb::arg("loc"),
             nb::arg("array_index") = -1,
             nb::arg("slot") = 0)
        .def("clear_all_gap_junctions", &SimWrapper::clear_all_gap_junctions, "清空所有间隙连接")
        .def("finalize_gap_junctions", &SimWrapper::finalize_gap_junctions, "完成间隙连接路由构建")
        .def("get_next_available_sid", &SimWrapper::get_next_available_sid, "获取下一个可用的源ID")
        .def("compute_transfer_impedance_dc_points",
             &SimWrapper::compute_transfer_impedance_dc_points,
             "DC注入方式计算转移阻抗（输入gid/section/loc，按NEURON语义映射到段）",
             nb::arg("gid"),
             nb::arg("source_section_name"),
             nb::arg("source_loc"),
             nb::arg("target_section_names"),
             nb::arg("target_locs"),
             nb::arg("inject_amp") = 1.0,
             nb::arg("v_init") = 0.0)
        .def("register_netstim_batch",
             [](SimWrapper& self,
                const std::vector<std::tuple<int, int, int>>& handles,
                double interval_scale,
                double start_base,
                double epsilon,
                double number) {
                 SimWrapper::NetStimBatchParams params;
                 params.interval_scale = interval_scale;
                 params.start_base = start_base;
                 params.epsilon = epsilon;
                 params.number = number;
                 return self.register_netstim_batch(handles, params);
             },
             nb::arg("handle_triplets"),
             nb::arg("interval_scale") = 5.0,
             nb::arg("start_base") = 9.0,
             nb::arg("epsilon") = 0.01,
             nb::arg("number") = 100.0)
        .def("register_vecstim_batch",
             [](SimWrapper& self,
                const std::vector<int>& mech_indices,
                double spike_scale,
                double start_base,
                double epsilon,
                int spike_count) {
                 SimWrapper::VecStimBatchParams params;
                 params.spike_scale = spike_scale;
                 params.start_base = start_base;
                 params.epsilon = epsilon;
                 params.spike_count = spike_count;
                 return self.register_vecstim_batch(mech_indices, params);
             },
             nb::arg("mech_indices"),
             nb::arg("spike_scale") = 5.0,
             nb::arg("start_base") = 9.0,
             nb::arg("epsilon") = 0.01,
             nb::arg("spike_count") = 20)
        .def("register_vecstim_batch_by_locs",
             [](SimWrapper& self,
                const std::string& mech,
                const std::vector<int>& gids,
                const std::vector<std::string>& section_names,
                const std::vector<double>& locs,
                double spike_scale,
                double start_base,
                double epsilon,
                int spike_count) {
                 SimWrapper::VecStimBatchParams params;
                 params.spike_scale = spike_scale;
                 params.start_base = start_base;
                 params.epsilon = epsilon;
                 params.spike_count = spike_count;
                 return self.register_vecstim_batch_by_locs(
                     mech, gids, section_names, locs, params);
             },
             nb::arg("mech"),
             nb::arg("gids"),
             nb::arg("section_names"),
             nb::arg("locs"),
             nb::arg("spike_scale") = 5.0,
             nb::arg("start_base") = 9.0,
             nb::arg("epsilon") = 0.01,
             nb::arg("spike_count") = 20)
        .def("set_vecstim_sequences",
             &SimWrapper::set_vecstim_sequences,
             nb::arg("mech_indices"),
             nb::arg("event_offsets"),
             nb::arg("event_times_ms"))
        .def("set_input_batch_pixels",
             [](SimWrapper& self,
                int batch_id,
                nb::ndarray<double, nb::shape<-1>, nb::c_contig> pixels) {
                 std::span<const double> span(pixels.data(), pixels.shape(0));
                 return self.set_input_batch_pixels(batch_id, span);
             },
             nb::arg("batch_id"),
             nb::arg("pixels"));
}
