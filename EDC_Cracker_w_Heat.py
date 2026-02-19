import scipy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import re
import pandas as pd
from collections import defaultdict
from scipy.integrate import quad
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d

R_gas = 8.314 * (10 ** -3) #Kj/molK
P = 25 # bar
T = 560
A_kinetic = np.array([5.90e15,1.3e10,1.20e10,1.20e9,5e8,2e8,9.10e7,1.20e11,5e8,
              2e7,3e8,5e14,2.10e14,2e13,1.70e10,1.2e10,1.70e10,1e11,1e10,1e10,1.60e11]) # (m^3(n-1)/kmol(n-1)s)
Ea = np.array([342,7,34,42,45,48,0,56,31,30,61,90,84,70,4,6,15,20,13,12,70]) #Kj/mol

def k_function(T,A_kinetic,Ea,R_gas):
    k = A_kinetic * np.exp(-Ea / (R_gas * T))
    return k 

k_pyrolysis = k_function(T,A_kinetic,Ea,R_gas)
print(k_pyrolysis)

n = np.array([0.292,0,0,0,0,0,0,0,0,4.79e-06,0,4.92e-07,0,0,0,0,1.55e-05,0])
mol_fraction = n / np.sum(n)
a = np.array([20.486,0,0,0,30.291,0,5.949,-0.553,0,12.472,0,6.322,0,-3.444,26.821,12.954,0,-33.917])
b = np.array([23.130e-02,0,0,0,-7.201e-03,0,20.193e-02,26.063e-02,0,26.959e-02,0,34.307e-02,0,45.594e-02,75.781e-03,16.232e-02,0,47.436e-02])
c = np.array([-1.438e-04,0,0,0,12.460e-06,0,-1.536e-04,-1.840e-04,0,-2.0505e-04,0,-2.958e-04,0,-2.981e-04,-5.007e-05,-1.302e-04,0,-3.017e-04])
d = np.array([33.888e-09,0,0,0,-3.898e-09,0,47.730e-09,55.475e-09,0,63.011e-09,0,97.929e-09,0,82.564e-09,14.122e-09,42.077e-09,0,71.301e-09])
M_total = 28.95
p = 25 #bar
R = 8.314
T_ref = 273
T_flue = 1287
enthalpy_formation = np.array([-130.45,121,0,0,-92.3,0,37.2,-135.7,0,-127.5,0,-185.5,120,-188.15,0,0,0,0])

M_process = 29.85
n_flue = (np.array([0,(12.5*0.21),12.5,25]) * 2)
Molar_mass = np.array([99,35.5,63.5,98,36.5,61.5,62.5,64.5,98,99,132.5,133.5,88.5,125,26,97,78,12]) 
viscosity_flue = np.array([0,(505.7 * (10 ** -7)),(337 * (10 ** -7)),(296.8 * (10 ** -7))])
D_inner= 0.30 # diameter of coil (m)
D_outer = 0.31
A_inner = np.pi * ((D_inner / 2) ** 2)
A_outer = np.pi * ((D_outer / 2) ** 2)
volumetric_flow_flue = 2.465828475
velocity_flue = 1.04
conductivity_flue = (np.array([0,75.8,55.1,63.7]) * (10 ** -3))

T_emissivity = np.array([1000,1500])
a_emissivity = np.array([[2.6367,0.2712,-0.0804,0.030],
                        [2.7178,0.3386,-0.0990,-0.0030]
                        ])
T_constant_process = np.array([273,298,373,473,573,673,773,873])

thermal_conductivity = np.array([[7.33,0,0,0,13.10,0,10.49,0,0,0,0,0,0,0,0,0,0,0],
                                 [8.58,0,0,0,14.41,0,12.11,0,0,0,0,0,0,0,0,0,0,0],
                                 [12.75,0,0,0,18.15,0,17.38,0,0,0,0,0,0,0,0,0,0,0],
                                 [19.21,0,0,0,22.76,0,25.18,0,0,0,0,0,0,0,0,0,0,0],
                                 [26.55,0,0,0,26.95,0,33.68,0,0,0,0,0,0,0,0,0,0,0],
                                 [34.64,0,0,0,30.78,0,42.69,0,0,0,0,0,0,0,0,0,0,0],
                                 [43.35,0,0,0,34.28,0,52.10,0,0,0,0,0,0,0,0,0,0,0],
                                 [52.59,0,0,0,37.49,0,61.81,0,0,0,0,0,0,0,0,0,0,0]
                                ]) * (10 ** -3)

viscosity_process = np.array([[8.46,0,0,0,9.35,0,13.40,0,0,0,0,0,0,0,0,0,0,0],
                              [9.20,0,0,0,10.24,0,14.64,0,0,0,0,0,0,0,0,0,0,0],
                              [11.38,0,0,0,12.80,0,18.30,0,0,0,0,0,0,0,0,0,0,0],
                              [14.21,0,0,0,16.01,0,23.02,0,0,0,0,0,0,0,0,0,0,0],
                              [16.95,0,0,0,19.01,0,27.51,0,0,0,0,0,0,0,0,0,0,0],
                              [19.64,0,0,0,21.83,0,31.73,0,0,0,0,0,0,0,0,0,0,0],
                              [22.27,0,0,0,24.50,0,35.63,0,0,0,0,0,0,0,0,0,0,0],
                              [24.85,0,0,0,27.03,0,39.17,0,0,0,0,0,0,0,0,0,0,0]
                             ]) * (10 ** -7)


T_constants_coil = np.array([20,100,200,300,400,500,600,700,800,900,1000]) + 273
thermal_conductivity_coil = np.array([11.5,1.035,1.089,1.127,1.157,1.191,1.122,1.251,1.266,1.283,1.291])

interp_thermal = interp1d(T_constants_coil, thermal_conductivity_coil,axis=0,kind='linear')
k_coil = interp_thermal(1100)

boltzmann = 5.67 * (10 ** -8)
L = 1.5

def specific_heat_capacity(T,a,b,c,d):
    Cp = a + ( b * T) + (c * (T ** 2)) + (d * (T ** 3))
    return Cp
    
Cp = specific_heat_capacity(T,a,b,c,d)


def specific_enthalpy(Cp,mol_fraction,T_ref,T):
    delta_H,error = quad_vec(specific_heat_capacity,T_ref,T, args=(a,b,c,d))
    mixture_H = delta_H * mol_fraction
    mixture_deltaH = np.sum(mixture_H)
    Cp_mix_i = Cp * mol_fraction
    Cp_mix = np.sum(Cp_mix_i)
    return mixture_deltaH, Cp_mix
    
mixture_deltaH,Cp_mix = specific_enthalpy(Cp,mol_fraction,T_ref,T)
print(mixture_deltaH)   
print(Cp_mix)

a_interpolation = interp1d(T_emissivity, a_emissivity, axis=0, kind='linear')
a_Tf = a_interpolation(T_flue)
print(a_Tf)



def flue_gas_constants(a_Tf,D_outer,viscosity_flue,n_flue,conductivity_flue,velocity_flue,T_flue):
    p_atm = 1 / 1.01
    x = np.log10(p_atm * L)
    log_emissivity = a_Tf[0] + ( a_Tf[1] * x) + (a_Tf[2] * (x ** 2)) + (a_Tf[3] * (x ** 3))
    emissivity = (10 ** log_emissivity ) / T_flue
    absorb = emissivity
    viscosity_mix_i = viscosity_flue * (n_flue / np.sum(n_flue))
    V_mix = np.sum(viscosity_mix_i)
    conductivity_mix_i = conductivity_flue * (n_flue / np.sum(n_flue))
    C_mix = np.sum(conductivity_mix_i)
    Re_flue = (D_outer * velocity_flue) / V_mix
    Pr_flue = 0.7
    Nu_flue = 0.683 * (Re_flue ** 0.466) * (Pr_flue ** (1/3))
    Convective_flue = (Nu_flue * C_mix) / D_outer
    return Convective_flue, Re_flue, Pr_flue, Nu_flue, absorb,emissivity

Convective_flue, Re_flue, Pr_flue, Nu_flue,absorb,emissivity = flue_gas_constants(a_Tf,D_outer,viscosity_flue,n_flue,conductivity_flue,velocity_flue,T_flue)
print(Convective_flue, "convective heat transfer coefficent of the flue gas")

def conductivity_viscosity(T_constant_process,thermal_conductivity,T,viscosity_process,mol_fraction):
    Conductivity_P = interp1d(T_constant_process, thermal_conductivity, axis=0, kind='linear')
    conductivity_process_i = (Conductivity_P(T) * mol_fraction)
    conductivity_process_mix = np.sum(conductivity_process_i) 
    Viscosity_P = interp1d(T_constant_process, viscosity_process, axis=0, kind='linear')
    V_process_i = Viscosity_P(T) * mol_fraction
    V_process = np.sum(V_process_i)
    return V_process,conductivity_process_mix

V_process, conductivity_process_mix = conductivity_viscosity(T_constant_process,thermal_conductivity,T,viscosity_process,n)

print(V_process)
print(conductivity_process_mix)

def process_dimensionless(p,T,n,A_inner,V_process,conductivity_process_mix,R_gas,M_process, Cp_mix):
    V_flow = 0.753657999
    roe = M_process / V_flow
    velocity = V_flow / A_inner
    Re = (roe * velocity * D_inner) / V_process
    Pr = (V_process * Cp_mix) / conductivity_process_mix
    Nu = 0.023 * (Re ** (4/5))* (Pr ** 0.4)
    h_process = (Nu * conductivity_process_mix) / D_inner
    return h_process, Re, Pr, Nu

h_process, Re, Pr, Nu = process_dimensionless(p,T,n,A_inner,V_process,conductivity_process_mix,R,M_process, Cp_mix)
print(h_process, Re, Pr, Nu)

print(boltzmann)
print(emissivity)
print(absorb)
def q(vars,T, A_outer, A_inner, Convective_flue,boltzmann,T_flue,h_process,
      k_coil,D_outer,D_inner,emissivity,absorb,L):
    T_p0, T_pi = vars
    q_rad = A_outer * boltzmann * ((emissivity * (T_flue ** 4)) - (absorb * (T_p0 ** 4)))
    q_cond_flue = A_outer * Convective_flue * (T_flue - T_p0)
    q_cond_process = A_inner * h_process * (T_pi - T)
    q_conv = ((2 * np.pi * L * k_coil) / (D_outer * np.log(D_outer / D_inner))) * (T_p0 - T_pi)
    f1 = q_rad + q_cond_flue - q_conv 
    f2 = q_conv - q_cond_process
    return [f1, f2]

guess = [700, 600] 
T_p0 = guess[0]
T_pi = guess[1]
T_solution = fsolve(q, guess,
                    args=(T, A_outer, A_inner, Convective_flue,
                          boltzmann, T_flue, h_process,
                          k_coil, D_outer, D_inner,
                          emissivity, absorb, L))

Tpo_sol, Tpi_sol = T_solution

print(Tpo_sol, "outside wall temperature")
print(Tpi_sol, "inside wall temperaurte")

q_cond_process = A_inner * h_process * (Tpi_sol - T)
q_rad = A_outer * boltzmann * ((emissivity * (T_flue ** 4)) - (absorb * (Tpo_sol ** 4)))
q_cond_flue = A_outer * Convective_flue * (T_flue - Tpo_sol)

print(q_cond_process)
dh_dz = 20000
heat_rxn = -2000
def h_DeltaZ(dh_dz, T, mol_fraction, T_ref, a, b, c, d,heat_rxn):
    # find specific heat from current temp 
    Cp_T = specific_heat_capacity(T, a, b, c, d)
    h_z, _ = specific_enthalpy(Cp_T, mol_fraction, T_ref, T)
    
    #find the target enthalpy from current and integral result 
    h_target = h_z + dh_dz - heat_rxn

   
    
    def objective(T_guess):
        Cp_guess = specific_heat_capacity(T_guess, a, b, c, d)
        h_guess, _ = specific_enthalpy(Cp_guess, mol_fraction, T_ref, T_guess)
        return h_guess - h_target
    # 4. Solve for T_unknown
    # We use the current T as the starting guess because T_unknown will be nearby
    T_unknown_solution = fsolve(objective, x0=T)
    
    return T_unknown_solution[0]



T_unkown = h_DeltaZ(dh_dz,T,mol_fraction,T_ref,a,b,c,d,heat_rxn)

print(T_unkown, "temperature of next time step (K)")

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

c0 = np.array([0.38,0,0,0,0,0,0,0,0,6.49796E-06,0,6.54087E-7,0,0,0,0,2.05862E-5,0]) # concentration of each species (kmol/m^3)
Molar_mass = np.array([99,35.5,63.5,98,36.5,61.5,62.5,64.5,98,99,132.5,133.5,88.5,125,26,97,78,12]) # Kg/Kmol

M_normalised = 1 # normalised to allow dc_dz to be in the unit of (kmol/kg)
M_real = 28.94897299 #actual inlet mass to scale mass after
z = (0,1) # axial position of reactor coil
z_points = np.linspace(0,1,100) #points along the axial position for the ODE 

D = 0.19 # diameter of coil (m)

PFR_length = 400 # reacotr length (m)
Volume = A_inner * PFR_length
volumetric_flow = 0.753657999 # inlet volumetric flowrate (m^3/s)
residence_time = Volume / volumetric_flow #residence time of the reactor (s)
print(residence_time)


def mass_from_c_vector(c,M_total,Molar_mass):
    sol = c * Molar_mass
    correction_factor = M_total / np.sum(sol)
    mass_flow_sol = c * Molar_mass * correction_factor
    real_mol_flow = mass_flow_sol / Molar_mass
    return real_mol_flow

def heat_rxn(enthalpy_formation, mol_in, mol_out):
    heat_rxn_p = np.sum(mol_in * enthalpy_formation)
    heat_rxn_r = np.sum(mol_out * enthalpy_formation)
    heat_rxn = heat_rxn_p - heat_rxn_r
    return heat_rxn

def mass_flow_vector(c,M_total,Molar_mass):
    sol = c * Molar_mass
    correction_factor = M_total / np.sum(sol)
    mass_flow_sol = c * Molar_mass * correction_factor

    return mass_flow_sol


print(Volume)

C = np.pi * D_inner
L_PFR = PFR_length
def mass_energyODE(z, y,
                   S,n, c0,Ea, R,R_gas, M_total, A_kinetic, volumetric_flow,
                   L_PFR, C, a, b, c, d, T_ref,
                   a_Tf, D_outer, viscosity_flue, n_flue, conductivity_flue,
                   velocity_flue, T_flue, T_constant_process,
                   thermal_conductivity, viscosity_process, p,
                   M_process, A_inner, A_outer, k_coil, boltzmann, L,enthalpy_formation,Molar_mass):
    h_current = y[0] #speicifc enthalpy at currnet axial position 
    c_current = y[1:] #concentraion vecotr 9mol/kg) at the current axial position
    n = mass_from_c_vector(c_current,M_total,Molar_mass)
    mol_fraction = n / (np.sum(n))
    
    def enthalpy_objective(T_guess): #find temperature from the specific enthalpy minus the reaction energy
        Cp_guess = np.sum(mol_fraction * specific_heat_capacity(T_guess, a, b, c, d))
        h_guess, _ = specific_enthalpy(Cp_guess, mol_fraction, T_ref, T_guess)
        k_a = k_function(T_guess, A_kinetic, Ea, R_gas)
        r_guess = np.empty(len(k_a))
        for j in range(len(k_a)):
            r_guess[j] = k_a[j]
            for i in range(len(c_current)):
                r_guess[j] *= c_current[i] ** R[i,j]
                
        q_rxn_guess = -np.sum(S.T @ enthalpy_formation * r_guess)
        return h_guess - h_current - q_rxn_guess
    T_new = fsolve(enthalpy_objective, x0=T)[0]
    
    Cp = specific_heat_capacity(T_new,a,b,c,d) #recacluate heat transfer coefficents e.g Re and Cp for new T
    mixture_deltaH,Cp_mix = specific_enthalpy(Cp,mol_fraction,T_ref,T_new)
    Convective_flue, Re_flue, Pr_flue, Nu_flue,absorb,emissivity = flue_gas_constants(a_Tf,D_outer,viscosity_flue,n_flue,
                                                                                      conductivity_flue,velocity_flue,T_flue)
    V_process, conductivity_process_mix = conductivity_viscosity(T_constant_process,thermal_conductivity,T,viscosity_process,mol_fraction)
    h_process, Re, Pr, Nu = process_dimensionless(p,T_new,n,A_inner,V_process,conductivity_process_mix,R_gas,M_process, Cp_mix)
    guess = [700,600]
    T_p0 = guess[0] 
    T_pi = guess[1]
    T_solution = fsolve(q, guess,
                    args=(T, A_outer, A_inner, Convective_flue,
                          boltzmann, T_flue, h_process,
                          k_coil, D_outer, D_inner,
                          emissivity, absorb, L))

    Tpo_sol, Tpi_sol = T_solution #solver for the temperature of the outer coil and inner 
    q_cond_process = A_inner * h_process * (Tpi_sol - T_new)
      
    dh_dz = (C * L_PFR * q_cond_process) / 28.95
    k_new = k_function(T_new,A_kinetic,Ea,R_gas)
    k_a = k_new
    def reaction_rates(c_current):
        r = np.empty(len(k_a))
        for j in range(len(k_a)): # For each reaction
            r[j] = k_a[j] # Start with k_j
            for i in range(len(c_current)): # For each species
                r[j] *= c_current[i] ** R[i, j]
        return np.array(r)      
    
    r = (reaction_rates(c_current) / 1)
    dc_dz = ((A_inner * L_PFR) * (S @ r))
        
    
    return np.concatenate((([dh_dz]), dc_dz))



h_start, _ = specific_enthalpy(specific_heat_capacity(T, a,b,c,d), 
                               mol_fraction, T_ref, T)


y0 = np.concatenate(([h_start], c0))

solution = solve_ivp(
    mass_energyODE,
    z,
    y0,
    method='Radau',
    t_eval=z_points, atol = 1e-07, rtol = 1e-05,
    args=(S,n, c0,Ea, R,R_gas, M_total, A_kinetic, volumetric_flow, L_PFR, C, a, b, c, d, T_ref,
          a_Tf, D_outer, viscosity_flue, n_flue, conductivity_flue,
          velocity_flue, T_flue, T_constant_process,
          thermal_conductivity, viscosity_process, p,
          M_process, A_inner, A_outer, k_coil, boltzmann, L,enthalpy_formation,Molar_mass))

