#include <../../nrnconf.h>
#include <stdlib.h>
#include <errno.h>
#include "neuron.h"
#include "section.h"
#include "nrniv_mf.h"
#include "multisplit.h"
#include "multicore.h"
#include "nrncvode.h"

#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
/*
extern std::vector<Memb_list> memb_list;
static const char *MechNames[] = {
    "Exp2Syn_Excitatory_ori",
    "Exp2Syn_Excitatory",
    "Exp2Syn_Inhibitory",
    "Ca_LVAst", "Ca_HVA", 
    "Nap_Et2", "NaTa_t", "NaTs2_t", 
    "K_Pst", "K_Tst", 
    "Ih", "Im", 
    "SKv3_1", "hh_grad", "CaDynamics_E2", "SK_E2"
};
*/

/*
static std::map<std::string, std::vector<int> > mech2data;
mech2data["Exp2Syn_Excitatory_ori"] = {9,13,14,15};
mech2data["Exp2Syn_Excitatory"] = {5,8,6,9};
mech2data["Exp2Syn_Inhibitory"] = {4,6,7,8};
mech2data["Ca_LVAst"] = {3,4,7,6,9,5,8};
mech2data["Ca_HVA"] = {3,4,7,6,9,5,8};
mech2data["Nap_Et2"] = {3,4,7,6,9,5,8};
mech2data["NaTa_t"] = {3,4,7,6,9,5,8};
mech2data["NaTs2_t"] = {3,4,7,6,9,5,8};
mech2data["K_Pst"] = {3,4,7,6,9,5,8};
mech2data["K_Tst"] = {3,4,7,6,9,5,8};
mech2data["Ih"] = {4,5,7,6};
mech2data["Im"] = {3,4,6,5};
mech2data["SKv3_1"] = {3,4,6,5};
mech2data["hh_grad"] = {5,6,10,13,12,15,11,14}; 
mech2data["CaDynamics_E2"] = {10,5,6};
mech2data["SK_E2"] = {3,4,6,5};
*/

static std::map<std::string, std::vector<int> > mech2data = {
    {"Exp2Syn_Excitatory_ori", {10,13,15,16,17,8,9}},
    {"Exp2Syn_Excitatory", {5,8,6,9,4}},
    {"Exp2Syn_Inhibitory", {5,6,7,8,9,3,4}},
    {"tmgExSyn", {10,12,11,13,14,8,9}},
    {"Syn_Pseudo",{6,7,8,9,10,4,5}},
    {"Exp2Syn_back", {5,6,7,8,9,3,4}},
    {"Exp2Syn_Excitatory_back", {10,13,15,16,17,8,9}},

    {"hh_grad", {7,8,5,6,10,13,16,12,15,18,11,14,17,32,33}},
    {"hh3", {7,8,10,11,22,25,28,31,34,24,27,30,33,36,23,26,29,32,35,57,58}},
    {"hha_old",{42,43,8,7,12,13,14,15,16,17,18,19,20,21,22,23,32,33}},
    {"hha2",{42,43,8,7,12,13,14,15,16,17,18,19,20,21,22,23,32,33}},

    {"Ca_LVAst", {2,3,4,7,6,9,5,8,14}},
    {"Ca_HVA", {2,3,4,7,6,9,5,8,14}},
    {"calH",{3,1,6,9,8,11,7,10,16}},
    {"car",{3,6,7,10,9,12,8,11,1}},
    {"somacar",{2,6,7,10,9,12,8,11,17}},
    {"ca", {16,1,6,7,8,9,10,11,17}},
    {"h",{5,8,9,10,11,0}},
    {"nap",{4,5,6,8,7,11}},
    {"nap_poirazi", {10,3,4,5,6,8}},
    {"Nap_Et2", {2,3,4,7,6,9,5,8,14}},
    {"NaTa_t", {2,3,4,7,6,9,5,8,14}},
    {"NaTs2_t", {2,3,4,7,6,9,5,8,14}},
    {"NaTg", {6,7,8,11,10,13,9,12,18}},
    {"nax",{1,13,2,5,4,7,3,6,12}},
    {"na3",{2,19,3,6,9,5,8,11,4,7,10,18}},
    {"na3dend",{19,20,3,6,9,5,8,11,4,7,10,18}},
    {"na3notrunk",{19,20,3,6,9,5,8,11,4,7,10,18}},
    {"na", {16,1,6,7,8,9,10,11,17}},
    {"K_Pst", {2,3,4,7,6,9,5,8,14}},
    {"K_Tst", {2,3,4,7,6,9,5,8,14}},
    {"Ih", {3,4,5,7,6,1}},
    {"Im", {2,3,4,6,5,9}},
    {"SKv3_1", {2,3,4,6,5,9}},
    {"kdr",{3,4,5,7,6,10}},
    {"Kdr",{2,3,7,9,8,4}},
    {"kdr2",{2,3,4,6,5,9}},
    {"km",{3,1,6,4,5,9}},
    {"km2",{2,3,6,8,7,13}},
    {"km_zach", {11,1,4,6,5,12}},
    {"kv", {2,3,6,8,7,13}},
    {"kad",{4,9,10,13,12,15,11,14,20}},
    {"kap",{4,5,6,9,8,11,7,10,16}},
    {"kap_poirazi", {17,1,2,3,4,5,6,7,12}},
    {"kad_poirazi", {17,1,2,3,4,5,6,7,12}},
    {"kad_maria", {4,9,10,13,12,15,11,14,24}},
    {"kap_maria", {4,9,10,13,12,15,11,14,24}},
    {"kdf",{20,3,9,12,11,14,10,13,19}},
    {"sca",{19,4,9,12,11,14,10,13,20}},
    {"it", {17,2,7,8,9,10,11,12,18}},
    {"it2",{2,3,8,11,10,13,9,12,18}},
    {"ih",{3,4,7,9,8,0}},
    {"hd",{5,6,7,9,8,0}},
    {"h2",{3,4,5,7,6,0}},
    {"naf_kai",{11,12,13,14,17,18,15,16,23}},
    {"kaf_kai",{4,5,6,7,10,11,8,9,16}},
    {"kas_kai",{4,5,6,7,10,11,8,9,16}},
    {"kdr_kai",{2,3,4,6,5,9}},
    {"kir_kai",{4,5,6,8,7,11}},
    {"calH_tas", {6,13,7,10,8,11,9,12,18}},
    {"car_tas", {2,9,3,6,4,7,5,8,18}},
    {"somacar_tas",{2,3,4,7,5,8,6,9,18}},
    {"h_tas", {5,6,7,8,9,1}}, 
    {"kad_tas", {3,2,4,7,5,8,6,9,18}},
    {"kap_tas", {3,2,4,7,5,8,6,9,18}},
    {"km_tas", {3,2,4,5,6,9}},
    {"kvs",{8,9,10,11,12,15}},
    {"nas",{13,14,15,18,16,19,17,20,25}},
    {"nap_tas", {13,7,4,5,6,10}},



    {"SK_E2", {2,3,4,6,5,9}},
    {"kcas",{7,1,9,10,11,2}},
    {"kcas2",{13,2,5,7,6,14}},
    {"kca_zach", {13,2,5,6,7,14}},
    {"sk_kai",{2,3,4,6,5,10}},
    {"kca_tas", {6,5,7,8,9,12}},

    

    {"mykca", {3,4,5,7,8,6,12}},
    {"kca",{5,2,11,8,9,10,15}},
    {"bk", {3,4,5,7,8,6,13}},
    {"mykca_tas", {3,4,5,6,7,8,13}},

    {"cat",{2,8,9,10,11,12,13,14,15,16,18,17,19}},
    {"cal",{3,5,6,7,8,9,10,11,12}},
    {"car_kai",{2,3,4,5,6,7,8,9,10,11,13,12,14}},
    {"can_kai",{4,5,6,7,8,9,10,11,12,13,15,14,16}},
    {"caq_kai",{2,3,4,5,6,7,8,9,10}},
    {"cal12",{4,5,6,7,8,9,10,11,12,13,15,14,16}},
    {"cal13",{4,5,6,7,8,9,10,11,12,13,15,14,16}},
    {"cat32",{2,3,4,5,6,7,8,9,10,11,13,12,14}},
    {"cat33",{2,3,4,5,6,7,8,9,10,11,13,12,14}},
    {"cat_tas", {2,4,5,6,7,8,9,10,11,12,14,13,15}},
    {"cal_tas", {2,4,5,6,7,8,9,10,11}},


    {"CaDynamics_E2", {10,5,6}},
    {"cad", {7,4,5}},
    {"cad2", {7,1,2}},
    {"cad_muellner", {6,4,5}},
    {"cadyn",{6,5,4}},
    {"caldyn",{6,5,4}},
    {"cad_tas", {8,5,6}}
};

int show_mech_type(const char* mechname){
    int type = nrn_get_mechtype(mechname);
    return type;
}

void print_threads_mech() {
    NrnThreadMembList* tml;
    for (int i = 0; i < nrn_nthread; ++i) {
        NrnThread& nt = nrn_threads[i];
        py::print("********* Thread *********", i);
        for(tml = nt.tml; tml; tml = tml->next) {
            py::print(tml->index, "--", memb_func[tml->index].sym->name);
            py::print("Nodecount: ", tml->ml->nodecount);

            py::print("ml_list Nodecount: ", nt._ml_list[tml->index]->nodecount);
        }
    }
}

py::array_t<int> get_ml_nodeindices(const char* mechname){
    int type = nrn_get_mechtype(mechname);
    if (type < 0 || type >= n_memb_func) {
        py::print("mech_data_rec: no mechanism named %s\n", mechname);
        return py::array_t<double>();
    }

    /*only for single thread*/
    NrnThread& nt = nrn_threads[0];
    Memb_list* ml = nt._ml_list[type]; 
    if (ml == NULL) {
        py::print(mechname, " not exist~~");
        return py::array_t<int>();
    }
    py::array_t<int> nodeindices(ml->nodecount);
    std::memcpy(nodeindices.mutable_data(), ml->nodeindices, ml->nodecount * sizeof(int));

    return nodeindices;
}
py::array_t<double> syn_data_rec(const char* mechname) {
    int type = nrn_get_mechtype(mechname);
    int type_exc = nrn_get_mechtype("Exp2Syn_Excitatory_ori");
    int type_inh = nrn_get_mechtype("Exp2Syn_Inhibitory");
    int type_tmp = nrn_get_mechtype("tmgExSyn");
    int type_pseudo = nrn_get_mechtype("Syn_Pseudo");
    int type_back = nrn_get_mechtype("Exp2Syn_back");
    int type_exc_back = nrn_get_mechtype("Exp2Syn_Excitatory_back");
    if (type == type_exc || type == type_inh || type == type_tmp || type == type_pseudo || type == type_back || type == type_exc_back){
        NrnThread& nt = nrn_threads[0];
        Memb_list* ml = nt._ml_list[type]; 

        if (ml == NULL) return py::array_t<double>();

        auto& data_idx = mech2data.at(mechname);
        std::vector<double> rec(nt.end * 4); // i--0, g -- 1, didv--3, djdv--4, w--5 

        for(int i = 0; i < ml->nodecount; ++i){
            int idx = ml->nodeindices[i];
            rec[idx * 4] -= ml->data(i, data_idx[0]);
            rec[idx * 4 + 1] += ml->data(i, data_idx[1]) * ml->data(i, data_idx[5]);
            rec[idx * 4 + 2] += ml->data(i, data_idx[3]) * ml->data(i, data_idx[5]);
            rec[idx * 4 + 3] += ml->data(i, data_idx[4]) * ml->data(i, data_idx[5]);
        } 
        py::array_t<double> res(nt.end * 4);
        std::memcpy(res.mutable_data(), rec.data(), nt.end * 4 * sizeof(double));
        return res;
    }
    else{
        py::print("syn_data_rec: no syn named %s\n", mechname);
        return py::array_t<double>();
    }
}
py::array_t<double> mech_data_rec(const char* mechname) {

    int type = nrn_get_mechtype(mechname);
    if (type < 0 || type >= n_memb_func) {
        py::print("mech_data_rec: no mechanism named %s\n", mechname);
        return py::array_t<double>();
    }

    /*only for single thread*/
    NrnThread& nt = nrn_threads[0];
    Memb_list* ml = nt._ml_list[type];
    if (ml == NULL) return py::array_t<double>();
    
    auto& data_idx = mech2data.at(mechname);

    /*
    py::print(ml->nodecount);
    py::print(data_idx.size());
    */
    
    std::vector<double> arr(ml->nodecount * data_idx.size());
    for(int i = 0; i < ml->nodecount; ++i) {
        for(int j = 0; j < data_idx.size(); ++j) {
            arr[i * data_idx.size() + j] = ml->data(i,data_idx[j]);
        }
    }
    py::array_t<double> rec(ml->nodecount * data_idx.size());
    std::memcpy(rec.mutable_data(), arr.data(), ml->nodecount * data_idx.size() * sizeof(double));
    return rec;
}


PYBIND11_MODULE(data_c2py, m) {
    m.doc() = "Mechanism data recorder";
    m.def("print_threads_mech", &print_threads_mech, "Print threads mechanism");
    m.def("show_mech_type", &show_mech_type, "Show mechanism type");
    m.def("get_ml_nodeindices", &get_ml_nodeindices, "Get mechanism nodeindices");
    m.def("syn_data_rec", &syn_data_rec, "Synapse data recorder");
    m.def("mech_data_rec", &mech_data_rec, "Mechanism data recorder");
}