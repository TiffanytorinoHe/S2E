from __future__ import annotations

from ._common import *  # noqa: F403
from .wrappers import ObjWrapper, MonitorWrapper, RecorderWrapper, VecPlayWrapper, WindowRecorderWrapper
import os
import time


class _StaticMethodProxy:
    def __init__(self, client, mech_name, method_name):
        self._client = client
        self._mech_name = mech_name
        self._method_name = method_name

    def __call__(self, *args):
        return self._client.call_mech_static(self._mech_name, self._method_name, *args)


class _StaticInterfaceProxy:
    def __init__(self, client, mech_name):
        super().__setattr__("_client", client)
        super().__setattr__("_mech_name", mech_name)

    def __getattr__(self, name):
        if self._client.has_mech_static_property(self._mech_name, name):
            return self._client.get_mech_static_property_str(self._mech_name, name)
        if self._client.has_mech_static_method(self._mech_name, name):
            return _StaticMethodProxy(self._client, self._mech_name, name)
        raise AttributeError(f"'{self._mech_name}' static interface has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        if self._client.has_mech_static_property(self._mech_name, name):
            self._client.set_mech_static_property_str(self._mech_name, name, str(value))
            return
        raise AttributeError(f"'{self._mech_name}' static interface has no writable property '{name}'")


class NeurongManager:
    """Neurong模拟器的单例管理类"""
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NeurongManager, cls).__new__(cls)
            cls._client = neurong.Sim()
            cls._client.set_spike_output_enabled(False)
            cls._instance.pending_monitor_wrappers = []  # 待初始化的监控器包装器
            cls._instance.pending_window_recorder_wrappers = []
            cls._instance.pending_obj_wrappers = []      # 待初始化的对象包装器
            cls._instance.pending_vecplay_wrappers = []  # 待初始化的VecPlay包装器
            # Registry to track all created wrappers for export functionality
            cls._instance._all_monitor_wrappers = []     # 所有监控器包装器
            cls._instance._all_window_recorder_wrappers = []
            cls._instance._all_obj_wrappers = []         # 所有对象包装器
            cls._instance._all_vecplay_wrappers = []     # 所有VecPlay包装器
            # 设置默认参数
            cls._instance.device = "gpu"
            cls._instance.permute_type = 3
            cls._instance._device_explicitly_set = False
            cls._instance._permute_explicitly_set = False
            cls._instance._dt = 0.05  # 默认时间步长
            cls._instance._recorder_start_time_ms = 0.0
            cls._instance._recorder_stride_steps = 1
            cls._instance._model_loaded = False  # 模型加载状态
            cls._instance._optimizer_id = None
            cls._instance._optimizer_type = None
            cls._instance._pending_input_batches = []
            cls._instance._input_batches = {}
            cls._instance._active_input_mode = None
            cls._instance._netstim_default = {
                "interval_scale": 5.0,
                "start_base": 9.0,
                "epsilon": 0.01,
                "number": 100.0,
            }
            cls._instance._vecstim_default = {
                "spike_scale": 5.0,
                "start_base": 9.0,
                "epsilon": 0.01,
                "spike_count": 20,
            }
            cls._instance._static_interface_cache = {}
        return cls._instance
    
    @property
    def client(self):
        """获取neurong客户端"""
        return self._client

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cache = self.__dict__.setdefault("_static_interface_cache", {})
        if name in cache:
            return cache[name]
        mech_name = self._client.resolve_mech_static_interface(name)
        if mech_name:
            proxy = _StaticInterfaceProxy(self._client, mech_name)
            cache[name] = proxy
            return proxy
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    @property
    def dt(self):
        """获取时间步长"""
        return self._client.get_dt()
    
    @dt.setter
    def dt(self, value):
        """设置时间步长"""
        self._client.set_dt(value)
    
    def set_default_device(self, device="gpu"):
        """设置默认设备"""
        self._apply_device_setting(device, explicit=True, apply_client=False)
    
    def set_default_permute_type(self, permute_type=3):
        """设置默认排列类型"""
        self._apply_permute_setting(permute_type, explicit=True, apply_client=False)

    def set_recorder_buffer_capacity(self, capacity: int):
        """设置recorder每通道缓冲容量（单位：步）"""
        cap = int(capacity)
        if cap <= 0:
            raise ValueError(f"recorder buffer capacity must be positive, got {cap}")
        if not hasattr(self._client, "set_recorder_buffer_capacity"):
            raise RuntimeError("current neurong backend does not support set_recorder_buffer_capacity")
        rc = int(self._client.set_recorder_buffer_capacity(cap))
        if rc != 0:
            raise RuntimeError(f"set_recorder_buffer_capacity failed with code {rc}")
        return rc

    def set_recorder_storage_dtype(self, dtype: str):
        """设置recorder内部存储精度（fp64/fp32）"""
        dtype_norm = str(dtype).strip().lower()
        if dtype_norm not in {"fp64", "float64", "f64", "fp32", "float32", "f32"}:
            raise ValueError(f"unsupported recorder dtype: {dtype}")
        if not hasattr(self._client, "set_recorder_storage_dtype"):
            raise RuntimeError("current neurong backend does not support set_recorder_storage_dtype")
        rc = int(self._client.set_recorder_storage_dtype(dtype_norm))
        if rc != 0:
            raise RuntimeError(f"set_recorder_storage_dtype failed with code {rc}")
        return rc

    def set_recorder_start_time(self, start_time_ms: float):
        """设置recorder记录起始时间（单位：ms，包含该时刻）"""
        start = float(start_time_ms)
        if start < 0.0:
            raise ValueError(f"recorder start time must be non-negative, got {start}")
        if not hasattr(self._client, "set_recorder_start_time"):
            raise RuntimeError("current neurong backend does not support set_recorder_start_time")
        rc = int(self._client.set_recorder_start_time(start))
        if rc != 0:
            raise RuntimeError(f"set_recorder_start_time failed with code {rc}")
        self._recorder_start_time_ms = start
        return rc

    def set_recorder_stride(self, stride_steps: int):
        """设置recorder记录步长（1表示每个step都记录）"""
        stride = int(stride_steps)
        if stride <= 0:
            raise ValueError(f"recorder stride must be positive, got {stride}")
        if not hasattr(self._client, "set_recorder_stride"):
            raise RuntimeError("current neurong backend does not support set_recorder_stride")
        rc = int(self._client.set_recorder_stride(stride))
        if rc != 0:
            raise RuntimeError(f"set_recorder_stride failed with code {rc}")
        self._recorder_stride_steps = stride
        return rc

    def is_device_explicitly_set(self):
        """是否已由用户显式设置device"""
        return bool(getattr(self, "_device_explicitly_set", False))

    def is_permute_type_explicitly_set(self):
        """是否已由用户显式设置permute_type"""
        return bool(getattr(self, "_permute_explicitly_set", False))

    def _apply_device_setting(self, device, *, explicit=True, apply_client=True):
        if device is None:
            raise ValueError("device cannot be None")
        normalized = str(device).lower()
        self.device = normalized
        if explicit:
            self._device_explicitly_set = True
        if apply_client:
            self._client.set_device(normalized)

    def _apply_permute_setting(self, permute_type, *, explicit=True, apply_client=True):
        value = int(permute_type)
        self.permute_type = value
        if explicit:
            self._permute_explicitly_set = True
        if apply_client:
            self._client.set_permute_type(value)
    
    def setup_and_load_model(self, export_path, dt=0.05, v_init=-62.5):
        """
        统一处理NEURON模型导出和neurong模型加载
        
        Parameters:
        - export_path: 模型导出路径
        - dt: 时间步长
        - v_init: 初始电压
        """
        from neuron import h

        profile_setup = os.environ.get("NEURONG_PROFILE_SETUP", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        setup_t0 = time.perf_counter()
        last_t = setup_t0

        def mark_setup(name):
            nonlocal last_t
            if not profile_setup:
                return
            now = time.perf_counter()
            print(
                f"[setup_profile] {name}: delta={now - last_t:.6f}s total={now - setup_t0:.6f}s",
                flush=True,
            )
            last_t = now
        
        # 1. NEURON模型导出
        # 确保ParallelContext正确设置
        pc = h.ParallelContext()
        pc.setup_transfer()
        pc.set_maxstep(10)
        h.dt = dt
        h.finitialize(v_init)
        mark_setup("neuron_finitialize")
        
        # 导出模型
        pc.nrnbbcore_write(export_path)
        mark_setup("nrnbbcore_write")
        
        # 2. 配置neurong
        self.set_data_path(export_path)
        self.set_device(self.device, explicit=False)
        self.set_permute_type(self.permute_type, explicit=False)
        mark_setup("configure_client")
        
        # 3. 初始化所有对象包装器
        self._initialize_obj_wrappers()
        mark_setup("initialize_obj_wrappers")
        
        # 4. 初始化所有监控器包装器
        self._initialize_monitor_wrappers()
        mark_setup("initialize_monitor_wrappers")

        self._initialize_window_recorder_wrappers()
        mark_setup("initialize_window_recorder_wrappers")
        
        # 5. 初始化所有VecPlay包装器
        self._initialize_vecplay_wrappers()
        mark_setup("initialize_vecplay_wrappers")
        
        # 6. 加载neurong模型
        self.load_model()
        mark_setup("load_model")
        self._initialize_input_batches()
        mark_setup("initialize_input_batches")
    
    def create_obj_wrapper(self, obj, var_names=None):
        """
        创建对象包装器（延迟初始化）- 用于读写当前时刻的变量值
        
        ⚠️ 重要: 必须保持对返回的包装器的引用！
        不要丢弃返回值，neurong暂不支持自动资源回收。
        
        正确用法:
            wrapper = neurong_manager.create_obj_wrapper(iclamp)
            wrapper = neurong_manager.create_obj_wrapper(soma(0.5))
            
        错误用法:
            neurong_manager.create_obj_wrapper(iclamp)  # 返回值被丢弃！
        
        Parameters:
        -----------
        obj : NEURON object
            要包装的NEURON对象 (mech 或 segment)
        var_names : sequence[str], optional
            只需要访问少数变量时可传入白名单，避免初始化时遍历整个
            NEURON HocObject 属性集合。大规模 synapse wrapper 会明显受益。
            
        Returns:
        --------
        ObjWrapper
            对象包装器，必须保持引用
        """
        wrapper = ObjWrapper(obj, self._client, var_names=var_names)
        
        # Add to registry for export functionality
        self._all_obj_wrappers.append(wrapper)
        
        # 如果模型已经加载，立即初始化
        if hasattr(self, '_model_loaded') and self._model_loaded:
            wrapper._initialize()
            # print(f"Immediately initialized obj wrapper: {wrapper.get_mech_name()}")
        else:
            # 否则添加到待初始化列表
            self.pending_obj_wrappers.append(wrapper)
        
        return wrapper
    
    def create_recorder(self, obj, var_name="v", array_index=0):
        """
        创建变量记录器（延迟初始化）- 用于记录时间序列数据

        这是推荐的新API，取代旧的create_monitor_wrapper。

        ⚠️ 重要: 必须保持对返回的包装器的引用！
        不要丢弃返回值，neurong暂不支持自动资源回收。

        正确用法:
            recorder = neurong_manager.create_recorder(soma(0.5))
            recorder = neurong_manager.create_recorder(iclamp, "amp")
            recorder_array = neurong_manager.create_recorder(iclamp, "amp", 1)  # 数组索引1

        错误用法:
            neurong_manager.create_recorder(soma(0.5))  # 返回值被丢弃！

        Parameters:
        -----------
        obj : NEURON object
            要记录的NEURON对象 (segment 或 mech)
        var_name : str, optional
            要记录的变量名，默认为"v"
        array_index : int, optional
            数组索引，默认为0

        Returns:
        --------
        RecorderWrapper
            变量记录器对象，必须保持引用
        """
        wrapper = RecorderWrapper(obj, var_name, self._client, array_index)

        # Add to registry for export functionality
        self._all_monitor_wrappers.append(wrapper)  # 内部仍使用monitor列表

        # 如果模型已经加载，立即初始化
        if hasattr(self, '_model_loaded') and self._model_loaded:
            wrapper._initialize()
        else:
            # 否则添加到待初始化列表
            self.pending_monitor_wrappers.append(wrapper)

        return wrapper

    def create_monitor_wrapper(self, obj, var_name="v", array_index=0):
        """
        创建监控器包装器（兼容旧API）- 用于记录时间序列数据

        ⚠️ 已弃用: 推荐使用 create_recorder()，此方法仅为向后兼容保留

        ⚠️ 重要: 必须保持对返回的包装器的引用！
        不要丢弃返回值，neurong暂不支持自动资源回收。

        正确用法:
            monitor = neurong_manager.create_monitor_wrapper(soma(0.5))
            monitor = neurong_manager.create_monitor_wrapper(iclamp, "amp")
            monitor_array = neurong_manager.create_monitor_wrapper(iclamp, "amp", 1)  # 数组索引1

        错误用法:
            neurong_manager.create_monitor_wrapper(soma(0.5))  # 返回值被丢弃！

        Parameters:
        -----------
        obj : NEURON object
            要监控的NEURON对象 (segment 或 mech)
        var_name : str, optional
            要监控的变量名，默认为"v"
        array_index : int, optional
            数组索引，默认为0

        Returns:
        --------
        RecorderWrapper (兼容MonitorWrapper)
            监控器包装器对象，必须保持引用
        """
        # 直接调用新API
        return self.create_recorder(obj, var_name, array_index)

    def create_window_recorder(self, obj, var_name="v", array_index=0):
        """Create a segment-local observer for training losses.

        Unlike create_recorder(), this does not keep a full history and is read
        through SegmentRunner/OnlineVJPLearner windows.
        """
        wrapper = WindowRecorderWrapper(obj, var_name, self._client, array_index)
        self._all_window_recorder_wrappers.append(wrapper)
        if hasattr(self, '_model_loaded') and self._model_loaded:
            wrapper._initialize()
        else:
            self.pending_window_recorder_wrappers.append(wrapper)
        return wrapper
    
    def create_vecplay_wrapper(self, obj, var_name="amp"):
        """
        创建VecPlay包装器（延迟初始化）- 用于动态播放时间序列数据
        
        ⚠️ 重要: 必须保持对返回的包装器的引用！
        不要丢弃返回值，neurong暂不支持自动资源回收。
        
        正确用法:
            vecplay = neurong_manager.create_vecplay_wrapper(iclamp, "amp")
            vecplay = neurong_manager.create_vecplay_wrapper(soma(0.5), "v")
            
        错误用法:
            neurong_manager.create_vecplay_wrapper(iclamp)  # 返回值被丢弃！
        
        Parameters:
        -----------
        obj : NEURON object
            要控制的NEURON对象 (mech 或 segment)
        var_name : str, optional
            要控制的变量名，默认为"amp"
            
        Returns:
        --------
        VecPlayWrapper
            VecPlay包装器对象，必须保持引用
        """
        wrapper = VecPlayWrapper(obj, var_name, self._client)
        
        # Add to registry for export functionality
        self._all_vecplay_wrappers.append(wrapper)
        
        # 如果模型已经加载，立即初始化
        if hasattr(self, '_model_loaded') and self._model_loaded:
            wrapper._initialize()
        else:
            # 否则添加到待初始化列表
            self.pending_vecplay_wrappers.append(wrapper)
        
        return wrapper

    def set_vecplay_batch(self, vecplay_wrappers, tvec, y_ct):
        """
        批量设置VecPlay（共享tvec + 多通道y矩阵）。

        Parameters
        ----------
        vecplay_wrappers
            VecPlayWrapper 列表，长度为 C
        tvec
            时间轴，shape=(T,)
        y_ct
            多通道值矩阵，shape=(C, T)
        """
        if not vecplay_wrappers:
            return 0

        tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y_ct, dtype=np.float64, order="C")
        if y_arr.ndim != 2:
            raise ValueError(f"set_vecplay_batch expects y_ct shape (C, T), got {y_arr.shape}")
        if int(y_arr.shape[0]) != len(vecplay_wrappers):
            raise ValueError(f"set_vecplay_batch channel mismatch: expected {len(vecplay_wrappers)}, got {y_arr.shape[0]}")
        if int(y_arr.shape[1]) != int(tvec_arr.shape[0]):
            raise ValueError(f"set_vecplay_batch tvec mismatch: T={tvec_arr.shape[0]}, y_ct={y_arr.shape[1]}")

        mech_names = []
        var_names = []
        instance_ids = []
        for wrapper in vecplay_wrappers:
            if wrapper is None:
                raise ValueError("set_vecplay_batch does not accept None wrappers")
            if not wrapper.initialized:
                raise RuntimeError("VecPlay wrapper not initialized. Call setup_and_load_model() first.")
            mech_names.append(wrapper._mech_name)
            var_names.append(wrapper.var_name)
            instance_ids.append(int(wrapper._instance_id))

        if hasattr(self._client, "set_vecplay_batch"):
            rc = self._client.set_vecplay_batch(mech_names, var_names, instance_ids, tvec_arr, y_arr)
            if rc != 0:
                raise RuntimeError("set_vecplay_batch failed")
            return rc

        for i, wrapper in enumerate(vecplay_wrappers):
            wrapper.play(tvec_arr, y_arr[i])
        return 0

    def _initialize_obj_wrappers(self):
        """初始化所有对象包装器"""
        for wrapper in self.pending_obj_wrappers:
            wrapper._initialize()
            # print(f"Initialized obj wrapper: {wrapper.get_mech_name()}")

        # 清空待初始化列表
        self.pending_obj_wrappers.clear()
    
    def _initialize_monitor_wrappers(self):
        """初始化所有监控器包装器"""
        for wrapper in self.pending_monitor_wrappers:
            wrapper._initialize()
            # print(f"Initialized monitor wrapper for {wrapper._mech_name}.{wrapper.var_name}")

        # 清空待初始化列表
        self.pending_monitor_wrappers.clear()

    def _initialize_window_recorder_wrappers(self):
        for wrapper in self.pending_window_recorder_wrappers:
            wrapper._initialize()
        self.pending_window_recorder_wrappers.clear()

    def _initialize_vecplay_wrappers(self):
        """初始化所有VecPlay包装器"""
        for wrapper in self.pending_vecplay_wrappers:
            wrapper._initialize()

        # 清空待初始化列表
        self.pending_vecplay_wrappers.clear()

    def configure_input_stimulus_defaults(self, mode="netstim", **params):
        """配置输入刺激的默认参数"""
        mode = (mode or "netstim").lower()
        if mode == "netstim":
            target = self._netstim_default
        elif mode == "vecstim":
            target = self._vecstim_default
        else:
            raise ValueError(f"Unsupported input stimulus mode: {mode}")

        for key, value in params.items():
            if value is None:
                continue
            target[key] = value

    def register_input_stimulators(self, wrappers, mode="netstim", config=None):
        """
        注册输入刺激包装器，供Neurong批量更新

        wrappers: 包装器列表
        mode: "netstim" 或 "vecstim"
        config: 覆盖默认参数的字典
        """
        if not wrappers:
            return
        mode = (mode or "netstim").lower()
        valid_wrappers = [w for w in wrappers if w is not None]
        if not valid_wrappers:
            return

        record = {
            "mode": mode,
            "wrappers": valid_wrappers,
            "config": dict(config) if config else None,
        }

        if self._model_loaded:
            self._create_input_batch(record)
        else:
            self._pending_input_batches.append(record)

    def _initialize_input_batches(self):
        """在模型加载完成后初始化所有待注册的输入刺激批次"""
        if not self._pending_input_batches:
            return
        for record in self._pending_input_batches:
            self._create_input_batch(record)
        self._pending_input_batches.clear()

    def _create_input_batch(self, record):
        mode = record["mode"]
        wrappers = record["wrappers"]
        if not wrappers:
            return

        if mode == "netstim":
            params = dict(self._netstim_default)
        elif mode == "vecstim":
            params = dict(self._vecstim_default)
        else:
            raise ValueError(f"Unsupported input stimulus mode: {mode}")

        if record["config"]:
            params.update(record["config"])

        if mode == "netstim":
            handle_triplets = []
            for wrapper in wrappers:
                interval_handle = wrapper.get_handle("interval")
                start_handle = wrapper.get_handle("start")
                number_handle = wrapper.get_handle("number")
                handle_triplets.append((interval_handle, start_handle, number_handle))

            batch_id = self._client.register_netstim_batch(
                handle_triplets,
                params.get("interval_scale", 5.0),
                params.get("start_base", 9.0),
                params.get("epsilon", 0.01),
                params.get("number", 100.0),
            )
        elif mode == "vecstim":
            mech_indices = [wrapper.get_node_index() for wrapper in wrappers]
            batch_id = self._client.register_vecstim_batch(
                mech_indices,
                params.get("spike_scale", 5.0),
                params.get("start_base", 9.0),
                params.get("epsilon", 0.01),
                int(params.get("spike_count", 20)),
            )
        else:
            return

        if batch_id < 0:
            raise RuntimeError(f"Failed to register input stimulus batch for mode {mode}")

        self._input_batches[mode] = {
            "batch_id": batch_id,
            "size": len(wrappers),
            "config": params,
        }
        self._active_input_mode = mode

    def set_input_stimulus(self, values, mode=None):
        """
        批量设置输入刺激。返回True表示已由NeuronG处理。
        """
        if not self._model_loaded or not self._input_batches:
            return False

        mode = (mode or self._active_input_mode or "netstim").lower()
        if mode not in self._input_batches:
            return False

        batch_info = self._input_batches[mode]
        pixels = np.asarray(values, dtype=np.float64).reshape(-1)
        if pixels.size != batch_info["size"]:
            raise ValueError(
                f"Input size mismatch for {mode}: expected {batch_info['size']}, got {pixels.size}"
            )
        self._client.set_input_batch_pixels(batch_info["batch_id"], pixels)
        return True

    def set_data_path(self, path):
        """设置数据路径"""
        self._client.set_data_path(path)
    
    def set_device(self, device, *, explicit=True):
        """设置设备"""
        self._apply_device_setting(device, explicit=explicit, apply_client=True)
    
    def set_permute_type(self, permute_type, *, explicit=True):
        """设置排列类型"""
        self._apply_permute_setting(permute_type, explicit=explicit, apply_client=True)

    def enable_spike_output(self):
        """开启spk文件输出"""
        self._client.set_spike_output_enabled(True)

    def disable_spike_output(self):
        """关闭spk文件输出"""
        self._client.set_spike_output_enabled(False)

    def is_spike_output_enabled(self):
        """查询spk文件输出状态"""
        return self._client.is_spike_output_enabled()
    
    def load_model(self):
        """加载模型"""
        self._model_loaded = False
        result = self._client.load_model()
        if isinstance(result, int) and result < 0:
            raise RuntimeError(f"load_model failed with return code {result}")
        self._model_loaded = True
    
    def add_monitor(self, monitor_type, variable, node_idx, array_index=0):
        """添加监控器"""
        return self._client.add_monitor_with_array(monitor_type, variable, node_idx, array_index)
    
    def get_monitor_data(self, monitor_id):
        """获取监控数据"""
        return self._client.get_monitor_data(monitor_id)

    def get_multiple_monitor_data(self, monitor_ids):
        """批量获取监控数据（返回handle->list映射）"""
        if not monitor_ids:
            return {}
        handle_list = [int(h) for h in monitor_ids]
        return self._client.get_multiple_monitor_data(handle_list)

    def get_monitor_sample_count(self, monitor_id):
        """返回单个 monitor 当前已刷入 readout buffer 的样本数。"""
        if not hasattr(self._client, "get_monitor_sample_count"):
            return len(self.get_monitor_data(monitor_id))
        return int(self._client.get_monitor_sample_count(int(monitor_id)))

    def get_monitor_cursor(self, monitor_ids):
        """返回一组同步 monitor 的当前样本 cursor。"""
        handle_list = [int(h) for h in monitor_ids]
        if not handle_list:
            return 0
        counts = [self.get_monitor_sample_count(h) for h in handle_list]
        first = counts[0]
        if any(c != first for c in counts):
            raise RuntimeError(f"monitor cursor mismatch: {counts[:8]}")
        return int(first)

    def get_multiple_monitor_data_f32(self, monitor_ids, n_steps=None, start_step=0):
        """
        批量获取监控数据（float32 2D numpy）。

        Parameters
        ----------
        monitor_ids
            监控句柄列表
            n_steps
            输出步数。若为None，则回退到dict/list路径后按最长序列自动推断。
        start_step
            readout buffer 起始样本 offset。在线 segment 用它读取新增数据。

        Returns
        -------
        numpy.ndarray
            shape=(len(monitor_ids), n_steps), dtype=float32
        """
        if not monitor_ids:
            if n_steps is None:
                return np.zeros((0, 0), dtype=np.float32)
            return np.zeros((0, int(n_steps)), dtype=np.float32)

        handle_list = [int(h) for h in monitor_ids]
        start = int(start_step)
        if start < 0:
            raise ValueError(f"start_step must be non-negative, got {start}")

        if n_steps is not None and start == 0 and hasattr(self._client, "get_multiple_monitor_data_f32_into"):
            out = np.empty((len(handle_list), int(n_steps)), dtype=np.float32)
            try:
                rc = self._client.get_multiple_monitor_data_f32_into(handle_list, out)
            except AttributeError:
                rc = None
            if rc is None:
                pass
            elif rc != 0:
                raise RuntimeError("get_multiple_monitor_data_f32_into failed")
            else:
                return out
        if n_steps is not None and hasattr(self._client, "get_multiple_monitor_data_f32_from_into"):
            out = np.empty((len(handle_list), int(n_steps)), dtype=np.float32)
            rc = self._client.get_multiple_monitor_data_f32_from_into(handle_list, start, out)
            if rc != 0:
                raise RuntimeError("get_multiple_monitor_data_f32_from_into failed")
            return out

        data_map = self._client.get_multiple_monitor_data(handle_list)
        if n_steps is None:
            n_steps = max((max(0, len(data_map.get(h, [])) - start) for h in handle_list), default=0)
        out = np.zeros((len(handle_list), int(n_steps)), dtype=np.float32)
        for i, handle in enumerate(handle_list):
            values = np.asarray(data_map.get(handle, []), dtype=np.float32)
            values = values[start:]
            n = min(int(n_steps), int(values.size))
            if n > 0:
                out[i, :n] = values[:n]
        return out

    def begin_window_recording(self, window_ids, n_steps):
        ids = [int(h) for h in window_ids]
        if not ids:
            return 0
        rc = self._client.begin_window_observer_recording(ids, int(n_steps))
        if rc != 0:
            raise RuntimeError("begin_window_observer_recording failed")
        return rc

    def end_window_recording(self):
        rc = self._client.end_window_observer_recording()
        if rc != 0:
            raise RuntimeError("end_window_observer_recording failed")
        return rc

    def get_window_recording_f32(self, window_ids, n_steps):
        ids = [int(h) for h in window_ids]
        out = np.empty((len(ids), int(n_steps)), dtype=np.float32)
        if not ids:
            return out
        rc = self._client.get_window_observer_data_f32_into(ids, out)
        if rc != 0:
            raise RuntimeError("get_window_observer_data_f32_into failed")
        return out
    
    def set_dt(self, dt):
        """设置时间步长（兼容旧接口）"""
        self.dt = dt
    
    def finitialize(self, v_init):
        """初始化"""
        self._client.finitialize(v_init)
    
    def run(self, runtime):
        """运行仿真"""
        self._client.run(runtime)

    def continue_run(self, runtime):
        """继续运行仿真（不重置t，从当前t开始推进runtime时间）"""
        self._client.continue_run(runtime)

    def fadvance(self):
        """单步推进一次（推进一个dt）"""
        self._client.fadvance()

    def get_t(self):
        """获取当前仿真时间t"""
        return self._client.get_t()

    def flush_recorders(self):
        """将record缓冲区刷入输出缓冲（step模式可用）"""
        return self._client.flush_recorders()
    
    def get_variable_value(self, mech_name, var_name, row_id, array_index=0):
        """获取变量值"""
        return self._client.get_variable_value_with_array(mech_name, var_name, row_id, array_index)

    def set_variable_value(self, value, mech_name, var_name, row_id, array_index=0):
        """设置变量值"""
        self._client.set_variable_value_with_array(value, mech_name, var_name, row_id, array_index)

    def get_variable_array(self, mech_name, var_name, row_id, array_length):
        """获取整个数组变量的值"""
        return [self._client.get_variable_value_with_array(mech_name, var_name, row_id, i)
                for i in range(array_length)]

    def get_variables_by_handles(self, handles):
        """批量读取变量值"""
        if not handles:
            return []
        handle_list = [int(h) for h in handles]
        return self._client.get_variables_by_handles(handle_list)

    def get_variables_by_handles_f32(self, handles):
        """
        批量读取变量值（float32 numpy array）。

        性能目标：避免 GPU 模式下逐个 handle 的小额 GPU->CPU 拷贝，以及 Python list 装箱开销。
        若底层不支持，则回退到 `get_variables_by_handles()` 再转 numpy。
        """
        if not handles:
            return np.zeros((0,), dtype=np.float32)
        handle_list = [int(h) for h in handles]
        if hasattr(self._client, "get_variables_by_handles_f32_into"):
            out = np.empty((len(handle_list),), dtype=np.float32)
            self._client.get_variables_by_handles_f32_into(handle_list, out)
            return out
        vals = self._client.get_variables_by_handles(handle_list)
        return np.asarray(vals, dtype=np.float32)

    def get_learnable_grad_handles_from_variable_handles(self, variable_handles):
        """从变量handles批量派生对应learnable grad handles。"""
        if not variable_handles:
            return []
        handle_list = [int(h) for h in variable_handles]
        return self._client.get_learnable_grad_handles_from_variable_handles(handle_list)

    def debug_get_postsyn_delay_info(self, mech_name, instance_index):
        """测试/调试用：读取 postsyn 实例的 (delay_ms, delay_step)。"""
        return self._client.debug_get_postsyn_delay_info(str(mech_name), int(instance_index))

    def set_variables_by_handles(self, handles, values):
        """批量写入变量值"""
        if not handles:
            return 0
        handle_list = [int(h) for h in handles]
        value_list = [float(v) for v in values]
        return self._client.set_variables_by_handles(handle_list, value_list)

    def compute_dw_from_mapping_v1(
        self,
        forward_rows,
        backward_rows,
        fwd_row_of_pair,
        bwd_row_of_pair,
        param_index_of_pair,
        scale_of_pair,
        tf_use,
        dt_fwd,
        dt_bwd,
        n_param,
        t_start_fwd=0.0,
        force_fp64=False,
    ):
        """
        基于映射关系计算参数梯度（v1）。

        `force_fp64=True` 用 fp64 + numpy 回退路径计算（数值对照/调试用）；
        默认 False 走 fp32 + C++ 快路径。

        Notes
        -----
        `t_start_fwd` 用于与旧训练脚本语义对齐：
        旧实现会先在完整 lr 窗口插值，再取末尾 `tf_use` 个点参与 dw 计算。
        这里可通过设置 `t_start_fwd = (lr_steps - tf_use) * dt_fwd` 复现该行为。

        Returns
        -------
        numpy.ndarray
            shape=(n_param,), dtype=float32 (or float64 when force_fp64=True).
        """
        scalar_dtype = np.float64 if force_fp64 else np.float32
        fwd = np.asarray(forward_rows, dtype=scalar_dtype, order="C")
        bwd = np.asarray(backward_rows, dtype=scalar_dtype, order="C")
        fwd_idx = np.asarray(fwd_row_of_pair, dtype=np.int32, order="C")
        bwd_idx = np.asarray(bwd_row_of_pair, dtype=np.int32, order="C")
        pidx = np.asarray(param_index_of_pair, dtype=np.int32, order="C")
        scale = np.asarray(scale_of_pair, dtype=scalar_dtype, order="C")
        tf_use = int(tf_use)
        n_param = int(n_param)
        t_start_fwd = float(t_start_fwd)
        dt_fwd = float(dt_fwd)
        dt_bwd = float(dt_bwd)
        out = np.zeros((n_param,), dtype=scalar_dtype)

        if tf_use <= 0 or tf_use > int(fwd.shape[1]):
            raise ValueError(f"compute_dw_from_mapping_v1 invalid tf_use={tf_use} for tf={fwd.shape[1]}")
        if n_param < 0:
            raise ValueError(f"compute_dw_from_mapping_v1 invalid n_param={n_param}")
        if dt_fwd <= 0.0 or dt_bwd <= 0.0:
            raise ValueError(f"compute_dw_from_mapping_v1 invalid dt_fwd/dt_bwd: {dt_fwd}, {dt_bwd}")
        if t_start_fwd < 0.0:
            raise ValueError(f"compute_dw_from_mapping_v1 invalid t_start_fwd={t_start_fwd}")

        def _fallback_numpy():
            if n_param == 0 or fwd_idx.size == 0:
                return out
            nf, tf = fwd.shape
            nb_row, tb = bwd.shape
            if bwd_idx.size != fwd_idx.size or pidx.size != fwd_idx.size or scale.size != fwd_idx.size:
                raise ValueError("compute_dw_from_mapping_v1 mapping size mismatch")
            if tb <= 0:
                raise ValueError("compute_dw_from_mapping_v1 backward time length must be positive")

            t_query = t_start_fwd + np.arange(tf_use, dtype=np.float64) * dt_fwd
            t_bwd = np.arange(tb, dtype=np.float64) * dt_bwd
            accum = np.zeros((n_param,), dtype=np.float64)
            counts = np.zeros((n_param,), dtype=np.int64)

            for i in range(int(fwd_idx.size)):
                rf = int(fwd_idx[i])
                rb = int(bwd_idx[i])
                pi = int(pidx[i])
                if rf < 0 or rf >= nf or rb < 0 or rb >= nb_row or pi < 0 or pi >= n_param:
                    raise ValueError(f"compute_dw_from_mapping_v1 index out of range at pair {i}")
                b_interp = np.interp(t_query, t_bwd, bwd[rb].astype(np.float64, copy=False))
                s = np.sum(
                    b_interp * (-fwd[rf, :tf_use].astype(np.float64, copy=False)[::-1]) * float(scale[i]),
                    dtype=np.float64,
                )
                accum[pi] += s
                counts[pi] += 1

            nz = counts > 0
            out64 = np.zeros((n_param,), dtype=np.float64)
            out64[nz] = accum[nz] / counts[nz]
            return out64.astype(scalar_dtype, copy=False)

        if (not force_fp64) and hasattr(self._client, "compute_dw_from_mapping_v1_into"):
            try:
                rc = self._client.compute_dw_from_mapping_v1_into(
                    fwd,
                    bwd,
                    fwd_idx,
                    bwd_idx,
                    pidx,
                    scale,
                    tf_use,
                    dt_fwd,
                    dt_bwd,
                    out,
                    t_start_fwd,
                )
            except (AttributeError, TypeError):
                return _fallback_numpy()
            if rc != 0:
                raise RuntimeError("compute_dw_from_mapping_v1_into failed")
            return out

        return _fallback_numpy()

    def compute_spk_readout_loss_and_grads(
        self,
        spk_bct,
        target_bot,
        w_oc,
        b_o,
        *,
        k_out,
        dt,
    ):
        """
        计算spk读出loss及梯度（MSE + leaky readout）。

        Returns
        -------
        tuple
            (loss: float, pred_bot, dldspk_bct, dw_oc, db_o)
        """
        spk = np.asarray(spk_bct, dtype=np.float32, order="C")
        target = np.asarray(target_bot, dtype=np.float32, order="C")
        w = np.asarray(w_oc, dtype=np.float32, order="C")
        b = np.asarray(b_o, dtype=np.float32, order="C")
        if spk.ndim != 3 or target.ndim != 3:
            raise ValueError(f"spk/target must be 3D, got {spk.shape} / {target.shape}")
        if w.ndim != 2 or b.ndim != 1:
            raise ValueError(f"w/b must be 2D/1D, got {w.shape} / {b.shape}")
        if float(dt) <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        n_batch, n_cell, t_len = spk.shape
        if target.shape[0] != n_batch or target.shape[2] != t_len:
            raise ValueError(f"target shape mismatch: {target.shape} vs ({n_batch}, n_out, {t_len})")
        if w.shape[1] != n_cell or w.shape[0] != target.shape[1]:
            raise ValueError(f"w shape mismatch: {w.shape}, expected ({target.shape[1]}, {n_cell})")
        if b.shape[0] != target.shape[1]:
            raise ValueError(f"b shape mismatch: {b.shape}, expected ({target.shape[1]},)")

        pred = np.empty_like(target, dtype=np.float32, order="C")
        dldspk = np.zeros_like(spk, dtype=np.float32, order="C")
        dw = np.zeros_like(w, dtype=np.float32, order="C")
        db = np.zeros_like(b, dtype=np.float32, order="C")
        if hasattr(self._client, "compute_spk_readout_loss_and_grads_into"):
            try:
                loss = float(
                    self._client.compute_spk_readout_loss_and_grads_into(
                        spk,
                        target,
                        w,
                        b,
                        float(k_out),
                        float(dt),
                        pred,
                        dldspk,
                        dw,
                        db,
                    )
                )
            except AttributeError:
                loss = None
            if loss is not None:
                if loss < 0.0:
                    raise RuntimeError("compute_spk_readout_loss_and_grads_into failed")
                return loss, pred, dldspk, dw, db

        spk64 = spk.astype(np.float64, copy=False)
        target64 = target.astype(np.float64, copy=False)
        w64 = w.astype(np.float64, copy=False)
        b64 = b.astype(np.float64, copy=False)

        n_batch, n_cell, t_len = spk64.shape
        n_out = w64.shape[0]
        alpha = float((1.0 - float(k_out)) / float(dt))

        pred64 = np.zeros((n_batch, n_out, t_len), dtype=np.float64)
        z0 = spk64[:, :, 0] @ w64.T + b64[None, :]
        pred64[:, :, 0] = alpha * z0
        for t in range(1, t_len):
            zt = spk64[:, :, t] @ w64.T + b64[None, :]
            pred64[:, :, t] = float(k_out) * pred64[:, :, t - 1] + alpha * zt

        err = pred64 - target64
        loss = float(0.5 * np.sum(err * err))

        grad_pred = np.zeros_like(pred64)
        grad_next = np.zeros((n_batch, n_out), dtype=np.float64)
        for t in range(t_len - 1, -1, -1):
            grad_cur = err[:, :, t] + float(k_out) * grad_next
            grad_pred[:, :, t] = grad_cur
            grad_next = grad_cur

        grad_z = alpha * grad_pred
        dw64 = np.einsum("bot,bct->oc", grad_z, spk64, optimize=True)
        db64 = np.sum(grad_z, axis=(0, 2))
        dldspk64 = np.einsum("bot,oc->bct", grad_z, w64, optimize=True)

        return (
            loss,
            pred64.astype(np.float32, copy=False),
            dldspk64.astype(np.float32, copy=False),
            dw64.astype(np.float32, copy=False),
            db64.astype(np.float32, copy=False),
        )

    def create_optimizer(self, optimizer_type="sgd"):
        """创建优化器并返回ID"""
        if not self._model_loaded:
            raise RuntimeError("Model must be loaded before creating optimizer")
        optimizer_id = self._client.create_optimizer(optimizer_type)
        if optimizer_id < 0:
            raise RuntimeError(f"Failed to create optimizer of type {optimizer_type}")
        self._optimizer_id = optimizer_id
        self._optimizer_type = optimizer_type.lower()
        return optimizer_id

    def optimizer_add_param(self, optimizer_id, weight_handle, grad_handle, impedance):
        """向优化器注册参数"""
        if optimizer_id < 0:
            raise ValueError("optimizer_id must be non-negative")
        result = self._client.optimizer_add_param(optimizer_id, weight_handle, grad_handle, impedance)
        if result < 0:
            raise RuntimeError(f"Failed to register optimizer param (weight_handle={weight_handle}, grad_handle={grad_handle})")
        return result

    def optimizer_add_param_batch(self, optimizer_id, weight_handles, grad_handles, impedance):
        """批量向优化器注册一组参数（用于batch训练）"""
        if optimizer_id < 0:
            raise ValueError("optimizer_id must be non-negative")
        if len(weight_handles) != len(grad_handles):
            raise ValueError("weight_handles and grad_handles must have the same length")
        if len(weight_handles) == 0:
            raise ValueError("optimizer_add_param_batch requires at least one handle")

        weight_list = [int(h) for h in weight_handles]
        grad_list = [int(h) for h in grad_handles]

        result = self._client.optimizer_add_param_batch(optimizer_id, weight_list, grad_list, impedance)
        if result < 0:
            raise RuntimeError("Failed to register batch optimizer params")
        return result

    def configure_optimizer(self, optimizer_id, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """配置优化器超参数"""
        if optimizer_id < 0:
            raise ValueError("optimizer_id must be non-negative")
        result = self._client.configure_optimizer(optimizer_id, momentum, beta1, beta2, epsilon)
        if result < 0:
            raise RuntimeError("Failed to configure optimizer")
        return result

    def optimizer_step(self, optimizer_id, learning_rate, record_time, dt):
        """执行优化器更新"""
        if optimizer_id < 0:
            raise ValueError("optimizer_id must be non-negative")
        result = self._client.optimizer_step(optimizer_id, learning_rate, record_time, dt)
        if result < 0:
            raise RuntimeError("Optimizer step failed")
        return result
    
    def get_spk_by_gid(self, gid):
        """获取某个gid对应的spk时间戳"""
        return self._client.get_spk_by_gid(gid)

    def enhanced_export_model(self, export_path, dt=0.05, v_init=-62.5):
        """
        Enhanced export functionality that saves both NEURON model and wrapper metadata
        
        This method:
        1. Exports NEURON model using existing functionality
        2. Initializes all wrappers to extract metadata
        3. Saves wrapper metadata and configuration to JSON files
        
        Parameters:
        -----------
        export_path : str
            Directory path for export
        dt : float, optional
            Time step (default: 0.05)
        v_init : float, optional
            Initial voltage (default: -62.5)
            
        Files created:
        --------------
        - neurong_metadata.json: Wrapper information
        - neurong_config.json: Configuration (device, permute_type, dt, v_init)
        - *.dat files: NEURON exported data
        """
        import neurong_export
        from neuron import h
        
        # Create export directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)
        
        # 1. Standard NEURON model export
        pc = h.ParallelContext()
        pc.setup_transfer()
        pc.set_maxstep(10)
        h.dt = dt
        h.finitialize(v_init)
        
        # Export model
        pc.nrnbbcore_write(export_path)
        
        # 2. Configure neurong for metadata extraction
        self.set_data_path(export_path)
        self.set_device(self.device, explicit=False)
        self.set_permute_type(self.permute_type, explicit=False)
        
        # 3. Initialize all wrappers to extract metadata
        self._initialize_obj_wrappers()
        self._initialize_monitor_wrappers()
        self._initialize_window_recorder_wrappers()
        self._initialize_vecplay_wrappers()
        
        # 4. Collect all wrapper metadata
        metadata = {
            "version": "1.0",
            "export_timestamp": neurong_export.datetime.datetime.now().isoformat(),
            "wrappers": {
                "monitors": [],
                "window_recorders": [],
                "obj_wrappers": [],
                "vecplay_wrappers": []
            }
        }
        
        # Serialize all monitor wrappers (both pending and previously initialized)
        all_monitors = list(self.pending_monitor_wrappers) + [w for w in self._all_monitor_wrappers if w not in self.pending_monitor_wrappers]
        for wrapper in all_monitors:
            if wrapper.initialized:
                metadata["wrappers"]["monitors"].append(
                    neurong_export.serialize_monitor_wrapper(wrapper)
                )

        all_window_recorders = (
            list(self.pending_window_recorder_wrappers)
            + [w for w in self._all_window_recorder_wrappers if w not in self.pending_window_recorder_wrappers]
        )
        for wrapper in all_window_recorders:
            if wrapper.initialized:
                metadata["wrappers"]["window_recorders"].append(
                    neurong_export.serialize_window_recorder_wrapper(wrapper)
                )
        
        # Serialize all object wrappers
        all_obj_wrappers = list(self.pending_obj_wrappers) + [w for w in self._all_obj_wrappers if w not in self.pending_obj_wrappers]
        for wrapper in all_obj_wrappers:
            if wrapper.initialized:
                metadata["wrappers"]["obj_wrappers"].append(
                    neurong_export.serialize_obj_wrapper(wrapper)
                )
        
        # Serialize all VecPlay wrappers
        all_vecplay_wrappers = list(self.pending_vecplay_wrappers) + [w for w in self._all_vecplay_wrappers if w not in self.pending_vecplay_wrappers]
        for wrapper in all_vecplay_wrappers:
            if wrapper.initialized:
                metadata["wrappers"]["vecplay_wrappers"].append(
                    neurong_export.serialize_vecplay_wrapper(wrapper)
                )
        
        # 5. Save metadata to file
        metadata_path = os.path.join(export_path, "neurong_metadata.json")
        neurong_export.save_metadata_to_file(metadata, metadata_path)
        
        # 6. Save configuration
        config = {
            "device": self.device,
            "permute_type": self.permute_type,
            "dt": dt,
            "v_init": v_init
        }
        config_path = os.path.join(export_path, "neurong_config.json")
        neurong_export.save_config_to_file(config, config_path)
        
        # 7. Load neurong model
        self.load_model()

    def load_from_export(self, export_path):
        """
        Load model and recreate wrappers from exported metadata
        
        This method allows loading models without NEURON by:
        1. Loading configuration from JSON
        2. Loading wrapper metadata from JSON
        3. Creating standalone wrappers
        4. Loading neurong model
        
        Parameters:
        -----------
        export_path : str
            Directory containing exported model and metadata
            
        Returns:
        --------
        dict
            Dictionary containing recreated wrappers organized by type:
            {
                'monitors': [list of MonitorWrapper objects],
                'obj_wrappers': [list of ObjWrapper objects], 
                'vecplay_wrappers': [list of VecPlayWrapper objects]
            }
        """
        import neurong_export
        # Check for metadata files
        metadata_path = os.path.join(export_path, "neurong_metadata.json")
        config_path = os.path.join(export_path, "neurong_config.json")
        
        if not os.path.exists(metadata_path):
            print(f"Warning: No metadata found at {metadata_path}")
            print("Falling back to standard load behavior")
            # Fallback to existing behavior
            self.set_data_path(export_path)
            self.set_device(self.device, explicit=False)
            self.set_permute_type(self.permute_type, explicit=False)
            self.load_model()
            return {"monitors": [], "window_recorders": [], "obj_wrappers": [], "vecplay_wrappers": []}
        
        # Load metadata and configuration
        metadata = neurong_export.load_metadata_from_file(metadata_path)
        if not neurong_export.validate_metadata(metadata):
            raise ValueError(f"Invalid metadata format in {metadata_path}")

        # Load configuration if available
        if os.path.exists(config_path):
            config = neurong_export.load_config_from_file(config_path)
            if neurong_export.validate_config(config):
                self.set_device(config["device"], explicit=False)
                self.set_permute_type(config["permute_type"], explicit=False)
            else:
                raise ValueError(f"Invalid configuration format in {config_path}")
        else:
            print("Warning: No configuration file found, using current settings")
        
        # Configure neurong
        self.set_data_path(export_path)
        
        # Load neurong model first
        self.load_model()
        
        # Recreate wrappers from metadata
        recreated_wrappers = {
            "monitors": [],
            "window_recorders": [],
            "obj_wrappers": [],
            "vecplay_wrappers": []
        }

        # Recreate monitor wrappers
        for monitor_data in metadata["wrappers"]["monitors"]:
            wrapper = neurong_export.create_standalone_monitor_wrapper(monitor_data, self._client)
            # Register the monitor with neurong
            monitor_id = self._client.add_monitor_with_array(
                wrapper._mech_name,
                wrapper.var_name,
                wrapper._node_or_mech_idx,
                wrapper.array_index,
            )
            wrapper.monitor_id = monitor_id
            recreated_wrappers["monitors"].append(wrapper)

        for window_data in metadata["wrappers"].get("window_recorders", []):
            wrapper = neurong_export.create_standalone_window_recorder_wrapper(window_data, self._client)
            window_id = self._client.add_window_observer_with_array(
                wrapper._mech_name,
                wrapper.var_name,
                wrapper._node_or_mech_idx,
                wrapper.array_index,
            )
            wrapper.window_observer_id = window_id
            recreated_wrappers["window_recorders"].append(wrapper)

        # Recreate object wrappers
        for obj_data in metadata["wrappers"]["obj_wrappers"]:
            wrapper = neurong_export.create_standalone_obj_wrapper(obj_data, self._client)
            recreated_wrappers["obj_wrappers"].append(wrapper)

        # Recreate VecPlay wrappers
        for vecplay_data in metadata["wrappers"]["vecplay_wrappers"]:
            wrapper = neurong_export.create_standalone_vecplay_wrapper(vecplay_data, self._client)
            recreated_wrappers["vecplay_wrappers"].append(wrapper)

        
        return recreated_wrappers

    @classmethod
    def disable_destructor_warnings(cls):
        """手动禁用包装器析构警告"""
        global _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED
        _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED = False
    
    @classmethod  
    def enable_destructor_warnings(cls):
        """手动启用包装器析构警告"""
        global _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED
        _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED = True
