#include "ion_table.h"
#include "var_struct.cuh"

#include <stdexcept>
using namespace std;
static unordered_map<string,EionData> EionTable;
static unordered_map<string, IonMeta> IonMetaTable;

void reg_ion(Mode mode, const std::string& name, double **cpu_data, double **gpu_data,int nnodes){
    auto &eionData = EionTable[name];
    eionData.nnodes = nnodes;
    eionData.cpu_data = cpu_data;
    if(mode == Mode::GPU){
        int Eion_Var_Size = magic_enum::enum_count<EionVarNames>();
        if (eionData.gpu_data_oh_host) {
            delete[] eionData.gpu_data_oh_host;
            eionData.gpu_data_oh_host = nullptr;
        }
        eionData.gpu_data_oh_host = new double*[Eion_Var_Size];
        cudaMemcpy(eionData.gpu_data_oh_host,gpu_data,Eion_Var_Size*sizeof(double*),cudaMemcpyDeviceToHost);
    }

}
EionData& get_ion_vars(const std::string& name,bool must_exist){
    if(must_exist){
        assert(EionTable.find(name) != EionTable.end());
    }
    return EionTable[name];
}

void register_ion_meta(const std::string& name, const IonMeta& meta) {
    IonMetaTable[name] = meta;
}

bool has_ion_meta(const std::string& name) {
    return IonMetaTable.find(name) != IonMetaTable.end();
}

IonMeta get_ion_meta(const std::string& name, bool must_exist) {
    auto it = IonMetaTable.find(name);
    if (it == IonMetaTable.end()) {
        if (must_exist) {
            throw std::runtime_error("ion meta not found for ion " + name);
        }
        return IonMeta{};
    }
    return it->second;
}
