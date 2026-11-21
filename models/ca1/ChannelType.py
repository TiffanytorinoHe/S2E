#Category = ['hh_ori', 'hh_like', 'cad', 'sk', '']
""" 
Ion channel category 2023.12.15
'': not channel
'hh_ori': original Hodgkin-Huxley model
'hh_like': hh-like model, rates depend on v
'sk', SK-like channel, rates depend on [Ca]_i (c)
'kca': ca-dependent k channel, rates depend on c and v
'ghk': ion channel whose g = hh-like func * nonlinear func ghk(v,c)
'cad': ca dynamics

(category, ion, num of states)
"""
mech_type = dict()
mech_type['ca_ion'] = ('', None)
mech_type['na_ion'] = ('',None)
mech_type['k_ion'] = ('', None)
mech_type['pas'] = ('', None)

mech_type['hh_grad'] = ('hh_ori', 'hh', 3)
mech_type['hh3'] = ('hh_ori', 'hh', 5)
mech_type['hha_old'] = ('hh_ori', 'hh', 4)
mech_type['hha2'] = ('hh_ori', 'hh', 4)

mech_type['Ca_LVAst'] = ('hh_like', 'ca', 2)
mech_type['Ca_HVA'] = ('hh_like', 'ca', 2)
mech_type['Nap_Et2'] = ('hh_like', 'na', 2)
mech_type['NaTa_t'] = ('hh_like', 'na', 2)
mech_type['NaTs2_t'] = ('hh_like', 'na', 2)
mech_type['NaTg'] =   ('hh_like', 'na', 2)
mech_type['K_Pst'] = ('hh_like', 'k', 2)
mech_type['K_Tst'] = ('hh_like', 'k', 2)
mech_type['kad'] = ('hh_like', 'k', 2)
mech_type['kad_poirazi'] = ('hh_like', 'k', 2)
mech_type['kap'] = ('hh_like', 'k', 2)
mech_type['kap_poirazi'] = ('hh_like', 'k', 2)
mech_type['kdf'] = ('hh_like', 'k', 2)
mech_type['Ih'] = ('hh_like', 'hcn', 1)
mech_type['Im'] = ('hh_like', 'k', 1)
mech_type['SKv3_1'] = ('hh_like', 'k', 1)
mech_type['kdr'] = ('hh_like', 'k', 1)
mech_type['Kdr'] = ('hh_like', 'k', 1)
mech_type['kdr2'] = ('hh_like', 'k', 1)
mech_type['nax'] = ('hh_like', 'na', 2)
mech_type['na3'] = ('hh_like', 'na', 3)
mech_type['na3dend'] = ('hh_like', 'na', 3)
mech_type['na3notrunk'] = ('hh_like', 'na', 3)
mech_type['calH'] = ('hh_like', 'ca', 2)
mech_type['car'] = ('hh_like', 'ca', 2)
mech_type['somacar'] = ('hh_like', 'ca', 2)
mech_type['sca'] = ('hh_like', 'ca', 2)
mech_type['it2'] = ('hh_like', 'ca', 2)
mech_type['h'] = ('hh_like', 'hcn', 1)
mech_type['km'] = ('hh_like', 'k', 1)
mech_type['km2'] = ('hh_like', 'k', 1)
mech_type['km_zach'] = ('hh_like', 'k', 1)
mech_type['nap'] = ('hh_like', 'na', 1)
mech_type['nap_poirazi'] = ('hh_like', 'na', 1)
mech_type['ih'] = ('hh_like', 'hcn', 1)
mech_type['hd'] = ('hh_like', 'hcn', 1)
mech_type['h2'] = ('hh_like', 'hcn', 1)

mech_type['mykca'] = ('kca', 'k', 1)
mech_type['kca'] = ('kca', 'k', 1)

mech_type['CaDynamics_E2'] = ('cad', 'ca', 1)
mech_type['cad'] = ('cad', 'ca', 1)
mech_type['cad2'] = ('cad', 'ca', 1)

mech_type['SK_E2'] = ('sk', 'k', 1)
mech_type['kcas'] = ('sk', 'k', 1)
mech_type['kcas2'] = ('sk', 'k', 1)

mech_type['cat'] = ('ghk', 'Ca', 2) # Ca not ca
mech_type['cal'] = ('ghk', 'ca', 1)


