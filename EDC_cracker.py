import numpy as np
import scipy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import re
import pandas as pd

T = 730 #kelvin
R = 8.314 * (10 ** -3) #Kj/molK
P = 25 # bar
A = np.array([5.90e15,1.3e10,1.20e10,1.20e9,5e8,2e8,9.10e7,1.20e11,5e8,
              2e7,3e8,5e14,2.10e14,2e13,1.70e10,1.2e10,1.70e10,1e11,1e10,1e10,1.60e11]) # (m^3(n-1)/kmol(n-1)s)
Ea = np.array([342,7,34,42,45,48,0,56,31,30,61,90,84,70,4,6,15,20,13,12,70]) #Kj/mol

def k(A,Ea):
    k = A * np.exp(-Ea / (R * T))
    return k 

k = k(A,Ea)

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

D = 0.18 # diameter of coil (m)
A = np.pi * ((D / 2) ** 2) # cross sectional area of coiled reactor (m^2)
PFR_length = 450 # reacotr length (m)
Volume = A * PFR_length
Volumetric_flow = 0.753657999 # inlet volumetric flowrate (m^3/s)
residence_time = Volume / Volumetric_flow #residence time of the reactor (s)
print(residence_time)

def ODE_pyrolysis_solver(S, k, c0,R, z,z_points, M_normalised, A, reactor_length):
    def reaction_rates(c):
        r = np.empty(len(k))
        for j in range(len(k)):
            r[j] = k[j]
            for i in range(len(c)):
                r[j] *= (c[i] ** R[i,j])
        return np.array(r)
    def ODEs(t,c):
        r = (reaction_rates(c) / M_normalised) #turn the reaction yield matrix into a mass basis (kmol/m^3kg)
        dc_dz = (S @ r) * A * reactor_length
        return dc_dz
    soln = solve_ivp(ODEs, z, c0,t_eval=z_points, method='Radau', atol = 1e-10, rtol = 1e-8)
    return soln

soln = ODE_pyrolysis_solver(S,k,c0,R,z,z_points,M_normalised,A,reactor_length)



solution_inlet = soln.y[:,0] * MW
correction_factor = M_real / np.sum(solution_inlet)

soln_mass = soln.y * Molar_mass[:, np.newaxis] * correction_factor #ODE solver scaled to real mass flowrate

species_list = list(df.index) 

edc_idx = species_list.index("EDC")
vcm_idx = species_list.index("VCM")
hcl_idx = species_list.index("HCl")

Main_products = [("EDC", edc_idx, 'o'), ("VCM", vcm_idx, 's'), ("HCl", hcl_idx, '^')]

for name, idx, marker in Main_products:
    plt.plot(
        soln.t, 
        soln_mass[idx], 
        label=name, 
        marker=marker, 
        markevery=10, 
        linewidth=2
    )

plt.xlabel('Axial Position (z)')
plt.ylabel('mass flowrate (kg/s)')
plt.title('Pyrolysis reactor mass flow rate profile')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

EC_idx = species_list.index("EC")
one_EDC_idx = species_list.index("one_EDC")
AC_idx = species_list.index("AC")
CB_idx = species_list.index("CB")
C6H6_idx = species_list.index("C6H6")

CP_idx = species_list.index("CP")
DC_idx = species_list.index("DC")
TCE_idx = species_list.index("TCE")

C_idx = species_list.index("C")

main_byproducts = [("1:1-Dichloro", one_EDC_idx, 's'),("CB",CB_idx,"x"),("C6H6",C6H6_idx,"x")]

for name, idx, marker in main_byproducts:
    plt.plot(
        soln.t, 
        soln_mass[idx], 
        label=name, 
        marker=marker, 
        markevery=10, 
        linewidth=2
    )

plt.xlabel('Axial Position (z)')
plt.ylabel('mass flow (kg/s)')
plt.title('mass flow of main byproducts (kg/s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

other_byproducts = [("CP",CP_idx,'o'),("DC",DC_idx,'x'),("TCE",TCE_idx,'^'),("EC", EC_idx, 'o')]

for name, idx, marker in other_byproducts:
    plt.plot(
        soln.t,
        soln_mass[idx],
        label = name,
        marker = marker,
        markevery=10,
        linewidth=2,
    )

plt.xlabel("axial position")
plt.ylabel("mass flow (mol/kg)")
plt.title("mass flow of other byproducts")
plt.legend()
plt.grid(True,linestyle='--',alpha=0.6)
plt.show

plt.plot(
    soln.t,
    soln_mass[C_idx],
    label = "Coke",
    marker = 'x',
    markevery = 10,
    linewidth = 2
)
plt.xlabel("axial position")
plt.ylabel("mass flow (kg/s)")
plt.title("mass flow of coke (carbon  soot)")
plt.legend()
plt.grid(True,linestyle='--',alpha=0.6)
plt.show

plt.plot(
    soln.t,
    soln_mass[C6H6_idx],
    label = "benzene",
    marker = 'x',
    markevery = 10,
    linewidth = 2
)
plt.xlabel("axial position")
plt.ylabel("mass flow (kg/s)")
plt.title("mass flow of benzene")
plt.legend()
plt.grid(True,linestyle='--',alpha=0.6)
plt.show

mass_flow_in = soln_mass[:,0]
mass_flow_out = soln_mass[:,-1]
percentage_differnce = ((np.sum(mass_flow_in) - np.sum(mass_flow_out)) / np.sum(mass_flow_in)) * 100
print(percentage_differnce, "Percentage difference between inlet and outlet (%)")

outlet_mass_flow_data = soln_mass[:,-1]
df = pd.DataFrame({
    "mass flow (kg/s)": outlet_mass_flow_data,
    "species": species_list
})
print(df)

df.to_csv('Outlet_mass_flows.csv', index=False)
