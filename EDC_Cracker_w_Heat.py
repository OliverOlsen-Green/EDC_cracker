import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
import re
import pandas as pd
from collections import defaultdict
from scipy.integrate import quad
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors

R_gas = 8.314 * (10 ** -3) #Kj/molK
T = 535 #inlet to furnace temp (k)
A_kinetic = np.array([5.90e15,1.3e10,1.20e10,1.0e9,5e8,2e8,9.10e7,1.20e11,5e8,
              2e7,3e8,5e14,2.10e14,2e13,1.70e10,1.2e10,1.70e10,1e8,1e10,1e10,1.60e14]) # (m^3(n-1)/kmol(n-1)s)
Ea = np.array([342,7,34,42,45,48,0,56,31,30,61,90,84,70,4,6,15,20,13,12,70]) #Kj/mol
p = 18 
n = np.array([0.30,0,0,0,0,0,0,0,0,0.00045,0,0.00034,0,0,0,0,1.55e-06,0])
Molar_mass = np.array([99,35.5,63.5,98,36.5,61.5,62.5,64.5,98,99,132.5,133.5,88.5,125,26,97,78,12]) 

T_ref = 273
T_flue = 1170 #temp of the flue gas in the furnace
enthalpy_formation = np.array([-130.45,121,0,0,-92.3,0,37.2,-135.7,0,-127.5,0,-185.5,120,-188.15,0,0,0,0])

M_process = 29.85
n_flue = (np.array([0,(12.5*0.21),12.5,25]) * 1)
% viscosity_flue = np.array([0,(144.5 * (10 ** -6)),(51 * (10 ** -6)),(115.1 * (10 ** -6))]) # W/m^2K

a = np.array([20.486,0,0,0,30.291,0,5.949,-0.553,0,12.472,0,6.322,0,-3.444,26.821,12.954,0,-33.917])
b = np.array([23.130e-02,0,0,0,-7.201e-03,0,20.193e-02,26.063e-02,0,26.959e-02,0,34.307e-02,0,45.594e-02,75.781e-03,16.232e-02,0,47.436e-02])
c = np.array([-1.438e-04,0,0,0,12.460e-06,0,-1.536e-04,-1.840e-04,0,-2.0505e-04,0,-2.958e-04,0,-2.981e-04,-5.007e-05,-1.302e-04,0,-3.017e-04])
d = np.array([33.888e-09,0,0,0,-3.898e-09,0,47.730e-09,55.475e-09,0,63.011e-09,0,97.929e-09,0,82.564e-09,14.122e-09,42.077e-09,0,71.301e-09])
L_PFR = 413.82 # coil length (m)
D_inner= 0.220 # i.d of coil (m)
D_outer = 0.233
As_outer = np.pi * D_outer * L_PFR
As_inner = np.pi * D_inner * L_PFR
A_inner = np.pi * ((D_inner / 2) ** 2)
A_outer = np.pi * ((D_outer / 2) ** 2)
volumetric_flow_flue = 44.9
velocity_flue = 0.89
conductivity_flue = (np.array([0,75.8,55.1,63.7]) * (10 ** -3))

T_emissivity = np.array([1000,1500])
a_emissivity = np.array([[2.6367,0.2712,-0.0804,0.030],
                        [2.7178,0.3386,-0.0990,-0.0030]
                        ])
T_constants_process = np.array([273,298,373,473,573,673,773,873])

thermal_conductivity = np.array([[7.33,0,0,0,13.10,0,10.49,0,0,0,0,0,0,0,0,0,0,0],
                                 [8.58,0,0,0,14.41,0,12.11,0,0,0,0,0,0,0,0,0,0,0],
                                 [12.75,0,0,0,18.15,0,17.38,0,0,0,0,0,0,0,0,0,0,0],
                                 [19.21,0,0,0,22.76,0,25.18,0,0,0,0,0,0,0,0,0,0,0],
                                 [26.55,0,0,0,26.95,0,33.68,0,0,0,0,0,0,0,0,0,0,0],
                                 [34.64,0,0,0,30.78,0,42.69,0,0,0,0,0,0,0,0,0,0,0],
                                 [43.35,0,0,0,34.28,0,52.10,0,0,0,0,0,0,0,0,0,0,0],
                                 [52.59,0,0,0,37.49,0,61.81,0,0,0,0,0,0,0,0,0,0,0]
                                ]) * (10 ** -3) W/m^2K

T_constants_coil = np.array([20,100,200,300,400,500,600,700,800,900,1000]) + 273
thermal_conductivity_coil = np.array([11.5,13.0,14.7,16.3,17.9,19.5,21.1,22.8,24.7,27.1,31.9])

interp_thermal = interp1d(T_constants_coil, thermal_conductivity_coil,axis=0,kind='linear')
k_coil = interp_thermal(400)

boltzmann = 5.67 * (10 ** -8)
L = 5.8

def calc_mol_fraction(n):
    mol_fraction = n / np.sum(n)
    return mol_fraction

def k_function(T,A_kinetic,Ea,R_gas):
    k = A_kinetic * np.exp(-Ea / (R_gas * T))
    return k 

def Viscosity_Correlation(Molar_mass,n,M_process,T,R_gas,p):
    T_r = T * 1.8
    Mr = Molar_mass
    mol_fraction = n / np.sum(n)
    M_r_individual = mol_fraction * Mr
    Mr_mix = np.sum(M_r_individual)
    V_flow = (np.sum(n) * (R_gas * (10 ** 3)) * T) / (p * (100))
    roe = M_process / V_flow 
    roe_lbft = roe * 0.06243
    Kv = ((9.4 + (0.02 * Mr_mix))* (T_r ** 1.5)) / (209 + (19 * Mr_mix) + T_r)
    x = 3.5 + (986 / T_r) + (0.01 * Mr_mix)
    y = 2.4 - (0.2 * x)
    viscosity = ((10 ** (-4)) * (Kv * np.exp(x * (((roe_lbft / 62.4)) ** y)))) / 1000
    return viscosity

viscosity = Viscosity_Correlation(Molar_mass,n,M_process,T,R_gas,p)

print(viscosity) 

def specific_heat_capacity(T,a,b,c,d):
    Cp = (a + ( b * T) + (c * (T ** 2)) + (d * (T ** 3)))
    return Cp
    
Cp = specific_heat_capacity(T,a,b,c,d)


def specific_enthalpy(Cp,n,T_ref,T, Molar_mass):
    delta_H,error = quad_vec(specific_heat_capacity,T_ref,T, args=(a,b,c,d))
    mol_fraction = calc_mol_fraction(n)
    mixture_H = delta_H * mol_fraction
    H_j_Kg = mixture_H / Molar_mass
    mixture_deltaH = np.sum(H_j_Kg)
    Cp_mix_i = Cp * mol_fraction
    Cp_mix = np.sum(Cp_mix_i)
    return mixture_deltaH, Cp_mix

mixture_deltaH,Cp_mix = specific_enthalpy(Cp,n,T_ref,T,Molar_mass)
print(mixture_deltaH)   
print(Cp_mix)

a_interpolation = interp1d(T_emissivity, a_emissivity, axis=0, kind='linear')
a_Tf = a_interpolation(T_flue)

print(a_Tf)

def flue_gas_constants(a_Tf,D_outer,viscosity_flue,n_flue,conductivity_flue,velocity_flue,T_flue,L):
    p_atm = 1 / 1.01
    x = np.log10(p_atm * L)
    roe_flue = 1
    log_emissivity = a_Tf[0] + ( a_Tf[1] * x) + (a_Tf[2] * (x ** 2)) + (a_Tf[3] * (x ** 3))
    emissivity = (10 ** log_emissivity ) / T_flue
    absorb = emissivity
    viscosity_mix_i = viscosity_flue * (n_flue / np.sum(n_flue))
    V_mix = np.sum(viscosity_mix_i)
    conductivity_mix_i = conductivity_flue * (n_flue / np.sum(n_flue))
    C_mix = np.sum(conductivity_mix_i)
    Re_flue = (D_outer * velocity_flue * roe_flue) / V_mix
    Pr_flue = 0.70
    Nu_flue = 0.911 * (Re_flue ** 0.385) * (Pr_flue ** (1/3))
    Convective_flue = (Nu_flue * C_mix) / D_outer
    return Convective_flue, Re_flue, Pr_flue, Nu_flue, absorb,emissivity


Convective_flue, Re_flue, Pr_flue, Nu_flue,absorb,emissivity = flue_gas_constants(a_Tf,D_outer,viscosity_flue,n_flue,conductivity_flue,velocity_flue,T_flue,L)
print(Convective_flue, "convective heat transfer coefficent of the flue gas")

def conductivity_viscosity(T_constants_process,thermal_conductivity, Molar_mass,n,M_process,T,R_gas,p):
    Conductivity_P = interp1d(T_constants_process, thermal_conductivity, axis=0, kind='linear')
    mol_fraction = calc_mol_fraction(n)
    conductivity_process_i = (Conductivity_P(T) * mol_fraction)
    conductivity_process_mix = np.sum(conductivity_process_i)
    V_process = Viscosity_Correlation(Molar_mass,n,M_process,T,R_gas,p)
    return V_process,conductivity_process_mix

V_process, conductivity_process_mix = conductivity_viscosity(T_constants_process,thermal_conductivity, Molar_mass,n,M_process,T,R_gas,p)

print(V_process,"viscosity of mixture")
print(conductivity_process_mix)

def mol_from_c_vector(c,M_process,Molar_mass):
    sol = c * Molar_mass
    correction_factor = M_process / np.sum(sol)
    mass_flow_sol = c * Molar_mass * correction_factor
    real_mol_flow = mass_flow_sol / Molar_mass
    return real_mol_flow

def mass_flow_vector(c,M_process,Molar_mass):
    sol = c * Molar_mass
    correction_factor = M_process / np.sum(sol)
    mass_flow_sol = sol * correction_factor

    return mass_flow_sol


def process_dimensionless(n,p,T,A_inner,V_process,conductivity_process_mix,R_gas,M_process, Cp_mix):
    total_molar_flow = np.sum(n)
    V_flow = ((total_molar_flow * R_gas * T) / (p * 100) * (10 ** 3))
    roe = M_process / V_flow
    velocity = V_flow / A_inner
    Re = (roe * velocity * D_inner) / V_process
    Pr = 0.70
    Nu = 0.023 * (Re ** (4/5))* (Pr ** 0.4)
    h_process = (Nu * conductivity_process_mix) / D_inner
    return h_process, Re, Pr, Nu, V_flow

h_process, Re, Pr, Nu, V_flow = process_dimensionless(n,p,T,A_inner,V_process,conductivity_process_mix,R_gas,M_process, Cp_mix)
print(h_process, Re, Pr, Nu)
print(V_flow,"volumetric flow")
print(h_process, "convective heat transfer of process fluid")
print(boltzmann)
print(emissivity)
print(absorb)
def q(vars,T, As_outer, As_inner, Convective_flue,boltzmann,T_flue,h_process,
      k_coil,D_outer,D_inner,emissivity,absorb,L_PFR):
    T_p0, T_pi = vars
    q_rad = As_outer * boltzmann * ((emissivity * (T_flue ** 4)) - (absorb * (T_p0 ** 4)))
    q_conv_flue = As_outer * Convective_flue * (T_flue - T_p0)
    q_cond_process = As_inner * (h_process) * (T_pi - T)
    q_conv_pipe = ((2 * np.pi * L_PFR * k_coil) / ((np.log(D_outer / D_inner)))) * (T_p0 - T_pi)
    f1 = q_rad + q_conv_flue - q_cond_process 
    f2 = q_cond_process - q_conv_pipe
    return [f1, f2] 

guess = [750, 650] #iniital guess temps for temperature sovler
T_p0 = guess[0]
T_pi = guess[1]
T_solution = fsolve(q, guess,
                    args=(T, As_outer, As_inner, Convective_flue,
                          boltzmann, T_flue, h_process,
                          k_coil, D_outer, D_inner,
                          emissivity, absorb, L_PFR))

Tpo_sol, Tpi_sol = T_solution
#check temperatuer + flux
print(Tpo_sol, "outside wall temperature")
print(Tpi_sol, "inside wall temperaurte")

q_cond_process = A_outer * h_process * (Tpi_sol - T)
q_rad = A_outer * boltzmann * ((emissivity * (T_flue ** 4)) - (absorb * (Tpo_sol ** 4))) 
q_cond_flue = np.pi * D_inner * Convective_flue * (T_flue - Tpo_sol)

print((q_cond_flue+q_rad), "heat flux")

#stoichometric matrix adapted from stack overflow:https://stackoverflow.com/questions/49896768/how-can-i-use-python-to-create-a-stoichiometric-matrix
def coeff_comp(s):
    # Separate stoichiometric coefficient and compound
    result = re.search(r'(?P<coeff>\d*)(?P<comp>.*)', s)
    coeff = result.group('coeff')
    comp = result.group('comp')
    if not coeff:
        coeff = '1'                          # coefficient=1 if it is missing
    return comp, int(coeff)

equations = ['R1 : EDC -> Cl + CH2ClCH2',
             'R2 : EDC + Cl -> CH2ClCHCl + HCl' ,
             'R3 : EDC + CHClCH -> VCM + CH2ClCHCl',
             'R4 : EDC + CH2ClCH2 -> EC + CH2ClCHCl',
             'R5 : EDC + CHCl2CH2 -> one_EDC + CH2ClCHCl',
             'R6 : EDC + CH2ClCCl2 -> TCE + CH2ClCHCl',
             'R7: VCM + Cl -> CHCl2CH2',
             'R8: VCM + Cl -> HCl + CHClCH',
             'R9: VCM + CHClCH -> CP + Cl',
             'R10: VCM + CHCl2CH2 -> CB + Cl',
             'R11: VCM + CH2ClCH2 -> EC + CHClCH',
             'R12: CHClCH -> AC + Cl',
             'R13: CH2ClCHCl -> VCM + Cl',
             'R14: CH2ClCCl2-> DC + Cl',
             'R15: EC + Cl -> HCl + CH2ClCH2',
             'R16: one_EDC + Cl -> HCl + CHCl2CH2',
             'R17: TCE + Cl -> HCl + CH2ClCCl2',
             'R18: 2AC + CHClCH -> C6H6 + Cl',
             'R19: CH2ClCH2+ Cl -> VCM + HCl',
             'R20: CH2ClCHCl + Cl -> DC + HCl',
             'R21: AC + 2Cl -> 2C + 2HCl']
reactions_dict={}
for equation in equations:
    compounds = {}                           # dict -> compound: coeff 
    eq = equation.replace(' ', '')  
    r_id, reaction = eq.split(':')           # separate id from chem reaction
    lhs, rhs = reaction.split('->')         # split left and right hand side
    reagents = lhs.split('+')                # get list of reagents
    products = rhs.split('+')                # get list of products
    for reagent in reagents:
        comp, coeff = coeff_comp(reagent)
        compounds[comp] = - coeff            # negative for reactants
    for product in products:
        comp, coeff = coeff_comp(product)
        compounds[comp] = coeff              # positive for products
    reactions_dict[r_id] = compounds         


df = pd.DataFrame(reactions_dict).fillna(value=0).astype(int)
S = df.to_numpy() # convert from pandas to a numpy array

print(df)


R = [] #reaction order matrix

for row in S:
    new_row = []
    for val in row:
        if val < 0:
            new_row.append(-val) # turn reactant from negative to positive for rates
        else :
            new_row.append(0)
    R.append(new_row)
R = np.array(R)
V_flow_inlet = (np.sum(n) * (R_gas * 1000) * T) / (p * 100)

c0 = n / V_flow_inlet # inlet conc of each species (kmol/m^3)

Molar_mass = np.array([99,35.5,63.5,98,36.5,61.5,62.5,64.5,98,99,132.5,133.5,88.5,125,26,97,78,12]) # Kg/Kmol

M_normalised = 1 # normalised to allow dc_dz to be in the unit of (kmol/kg)
M_real = 28.94897299 #actual inlet mass to scale mass after
z = (0,1) # axial position of reactor coil
z_points = np.linspace(0,1,500) #points along the axial position for the solver

C = np.pi * D_inner #circumfrence of the reactor 
T_po_profile = []
T_pi_profile = []
z_storage = []
Q_rad_profile = []
Q_conv_profile = []
def mass_energyODE(z, y,
                   S,n, c0,Ea, R,R_gas, M_process, A_kinetic, V_flow,
                   L_PFR, C, a, b, c, d, T_ref,
                   a_Tf, D_outer, viscosity_flue, n_flue, conductivity_flue,
                   velocity_flue, T_flue, T_constant_process,
                   thermal_conductivity, p,
                   A_inner, A_outer, k_coil, boltzmann, L,enthalpy_formation,Molar_mass,As_outer, As_inner):
    h_current = y[0] #speicifc enthalpy at currnet axial position 
    c_current = y[1:] #concentraion vecotr 9mol/kg) at the current axial position
    n = mol_from_c_vector(c_current,M_process,Molar_mass)
    mol_fraction = calc_mol_fraction(n)
  
    def enthalpy_objective(T_guess): #find temperature from the specific enthalpy minus the reaction energy
        Cp_guess = np.sum(mol_fraction * specific_heat_capacity(T_guess, a, b, c, d))
        h_guess, _ = specific_enthalpy(Cp_guess, n, T_ref, T_guess,Molar_mass)
        k_a = k_function(T_guess, A_kinetic, Ea, R_gas)
        r_guess = np.empty(len(k_a))
        for j in range(len(k_a)):
            r_guess[j] = k_a[j]
            for i in range(len(c_current)):
                r_guess[j] *= c_current[i] ** R[i, j]
        q_rxn_guess = -(np.sum(S.T @ (enthalpy_formation) * r_guess))
        return h_guess - h_current - (q_rxn_guess)
    T_new = fsolve(enthalpy_objective, x0=T)[0]
    T_new = np.clip(T_new, T_constant_process[0], T_constant_process[-1])
    
    Cp = specific_heat_capacity(T_new,a,b,c,d) #recacluate heat transfer coefficents e.g Re and Cp for new T
    mixture_deltaH,Cp_mix = specific_enthalpy(Cp,n,T_ref,T_new,Molar_mass)
    Convective_flue, Re_flue, Pr_flue, Nu_flue,absorb,emissivity = flue_gas_constants(a_Tf,D_outer,viscosity_flue,n_flue,
                                                                                      conductivity_flue,velocity_flue,T_flue,L)
    V_process, conductivity_process_mix = conductivity_viscosity(T_constants_process,thermal_conductivity, Molar_mass,n,M_process,T_new,R_gas,p)
    h_process, Re, Pr, Nu,V_flow = process_dimensionless(n,p,T_new,A_inner,V_process,conductivity_process_mix,R_gas,M_process, Cp_mix)
    guess = [700,600]
    T_p0 = guess[0] 
    T_pi = guess[1]
    T_solution = fsolve(q, guess,
                    args=(T_new, As_outer, As_inner, Convective_flue,
                          boltzmann, T_flue, h_process,
                          k_coil, D_outer, D_inner,
                          emissivity, absorb, L_PFR))

    Tpo_sol, Tpi_sol = T_solution #solver for the temperature of the outer coil and inner 
    T_po_profile.append(Tpo_sol)
    T_pi_profile.append(Tpi_sol)
    z_storage.append(z)
    Q_rad = As_outer * boltzmann * ((emissivity * (T_flue ** 4)) - (absorb * (Tpo_sol ** 4)))
    Q_rad_profile.append(Q_rad)
    Q_conv = As_outer * Convective_flue * (T_flue - Tpo_sol)
    Q_conv_profile.append(Q_conv)
    q_cond_process = (h_process * (Tpi_sol - T_new)) / (1000 * M_process) #convert heat flux to (kw/m^2Kg) of the mass inr the coil 
    k_a = k_function(T_new, A_kinetic, Ea, R_gas)

    def reaction_rates(c_current): #  function adapted from The Law of mass action: Mathematical modelling and python implementation for chemical kinetics
        r = np.empty(len(k_a))
        for j in range(len(k_a)):
            r[j] = k_a[j]
            for i in range(len(c_current)):
                r[j] *= c_current[i] ** R[i, j]
        return np.array(r)
    r = reaction_rates(c_current) 
    
    q_rxn = np.sum(S.T @ (enthalpy_formation) * r)
    
    dh_dz = ((C * L_PFR * (q_cond_process - q_rxn))) #ODE for the energy balance
    dc_dz = ((A_inner * L_PFR) * (S @ r)) #ODE for the material balance 
        
    
    return np.concatenate((([dh_dz]), dc_dz))



h_start, _ = specific_enthalpy(specific_heat_capacity(T, a,b,c,d), 
                               n, T_ref, T,Molar_mass)


y0 = np.concatenate(([h_start], c0))

solution = solve_ivp(
    mass_energyODE,
    z,
    y0,
    method='Radau',
    t_eval=z_points, atol = 1e-09, rtol = 1e-07,
    args=(S,n, c0,Ea, R,R_gas, M_process, A_kinetic, V_flow, L_PFR, C, a, b, c, d, T_ref,
          a_Tf, D_outer, viscosity_flue, n_flue, conductivity_flue,
          velocity_flue, T_flue, T_constants_process,
          thermal_conductivity, p,
          A_inner, A_outer, k_coil, boltzmann, L,enthalpy_formation,Molar_mass,As_outer, As_inner))
z_arr = np.array(z_storage)
T_po_profile = np.array(T_po_profile)
T_pi_profile = np.array(T_pi_profile)
print(T_po_profile)
print(T_pi_profile)
sort_idx = np.argsort(z_arr)
z_sorted = z_arr[sort_idx]
species_list = list(df.index) 
z_unique, unique_idx = np.unique(z_sorted, return_index=True)
# main compponent list
edc_idx = species_list.index("EDC") + 1
vcm_idx = species_list.index("VCM") + 1
hcl_idx = species_list.index("HCl") + 1


Main_products = [("EDC", edc_idx, 'o'), ("VCM", vcm_idx, 's'), ("HCl", hcl_idx, '^')]

plt.figure(figsize=(10, 6))

for name, idx, marker in Main_products:
    
    plt.plot(
        solution.t, 
        solution.y[idx], 
        label=name, 
        marker=marker, 
        markevery=10, 
        linewidth=2
    )

plt.xlabel('Axial Position (z)')
plt.ylabel('Concentration (kmol/m³)') 
plt.title('Pyrolysis Reactor Species Profile (Integrated)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)


plt.savefig("mainproducts_profile.jpg", dpi=300)
plt.show()
print(solution.y[edc_idx,0])
print(solution.y[edc_idx,-1])
print(solution.y[1:,-1])
print(np.sum((solution.y[1:,0])* Molar_mass))
print(np.sum((solution.y[1:,-1])* Molar_mass))
Q_rad_profile = np.array(Q_rad_profile) / (10 ** 6) #Mw
Q_conv_profile = np.array(Q_conv_profile) / (10 ** 6)

solution_mass_1 = mass_flow_vector(solution.y[1:,:], M_process, Molar_mass[:,np.newaxis]) * 100
solution_mass = (M_process / np.sum((solution_mass_1[:,0]))) * solution_mass_1
print(np.sum(solution_mass[:,0]))
print(solution_mass[:,0])
edc_idx = species_list.index("EDC")
vcm_idx = species_list.index("VCM")
hcl_idx = species_list.index("HCl")
Main_products = [("EDC", edc_idx, 'o'), ("VCM", vcm_idx, 's'), ("HCl", hcl_idx, '^')]


for name, idx, marker in Main_products:
    
    plt.plot(
        solution.t, 
        solution_mass[idx], 
        label=name,
        linewidth=2
    )

plt.xlabel('Axial Position (z)')
plt.ylabel('mass flow rate (Kg/s)') 
plt.title('Species of main products')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Main products.png",format = 'png')
plt.show 

m_EDC_in = solution_mass[0,0]
m_EDC_out = solution_mass[0,-1]
def percentage_conversion(m_in,m_out):
    conversion = (m_in - m_out) / m_in
    percentage_conversion = conversion * 100
    return percentage_conversion

percentage_conversion = percentage_conversion(m_EDC_in,m_EDC_out)

print(percentage_conversion, "percentage conversion of EDC by mass (%)")



T_profile = []

for i in range(len(solution.t)):
    h_val = solution.y[0, i]
    # use current conc for mol fractiions 
    c_current = solution.y[1:, i]
    mf_current = c_current / np.sum(c_current)
    
    def objective(T_guess):
        cp_g = specific_heat_capacity(T_guess, a, b, c, d)
        h_g, _ = specific_enthalpy(cp_g, mf_current, T_ref, T_guess,Molar_mass)
        return h_g - h_val
    
    # invert h
    T_found = fsolve(objective, x0=560)[0]
    T_profile.append(T_found)
z_internal = np.linspace(0, 1, len(T_po_profile))
T_po_arr = np.array(T_po_profile)
T_pi_arr = np.array(T_pi_profile)
T_po_sorted = T_po_arr[sort_idx]
T_pi_sorted = T_pi_arr[sort_idx]
interp_Tpo = interp1d(z_unique, T_po_sorted[unique_idx])
interp_Tpi = interp1d(z_unique, T_pi_sorted[unique_idx])

T_po_aligned = interp_Tpo(solution.t)
T_pi_aligned = interp_Tpi(solution.t)

Temp_profile = [("Outer pipe temperature (K)",T_po_aligned,'o','#DC143C'),("Inner pipe temperature (K)",T_pi_aligned,'^','#F4A742'),("Process temperature (k)",T_profile,'s','#2166AC')]

for name, data, marker,Colour in Temp_profile:
    
    plt.plot(
        solution.t, 
        data, 
        label=name, 
        color = Colour,
        linewidth=2
    )
plt.xlabel('Axial Position (z)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Profile along Reactor')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("Temperature.png", format='png')
plt.show()
# inlet and outlet temperature
print("inlet temperature" ,T_profile[0])
print("outlet temperature",T_profile[-1])

Q_rad = Q_rad_profile[sort_idx]
interp_Q_rad = interp1d(z_unique,Q_rad[unique_idx])
Q_rad_aligned = interp_Q_rad(solution.t)
plt.plot(
    solution.t,
    Q_rad_aligned,
    label = "MW from radiation",
    marker = 'x',
    markevery = 100,
    linewidth = 2
)
plt.xlabel("axial position")
plt.ylabel("Heat flow from radiation")
plt.title("Heat from radiation (Mw)")
plt.legend()
plt.grid(True,linestyle='--',alpha=0.6)
plt.savefig("Rad.png" , format='png')
plt.show

outlet_mass_flow_data = solution_mass[:,-1]
df = pd.DataFrame({
    "mass flow (kg/s)": outlet_mass_flow_data,
    "species": species_list
})
print(df)
#export outlet mass flows of all species
df.to_csv('Outlet_mass_flows.csv', index=False)
