The overall goal of this suite of PySCF calculations is to produce a set of forces between methanol (CH3OH, here designated MEOH) and sodium methyl monophosphate (CH3NaO4P-, designated MMP), in the presence of water (treated as MM objects). The forces will based on coordinates that are given (based, e.g., on a gromacs/MD simulation), a desired constraint distance between methanol's Oxygen and MMP's Phosphorus.

Steps along the way:

In the folder QM_MEOH, write a minimizer (say minimize.ipynb) for a geometry-optimized MEOH molecule. This would include machinery for generating the .cube orbitals, and visualizing them. Also write an analyzer (analyze.ipynb) that graphs or otherwise reports on the results. Hopefully, the contents of visualize_QMMM.py will help in this regard. -Done


In the folder QM_MMP, do the same thing for MMP (minimize and analyze).


In the folder QM_MEOH_MMP, repeat with both molecules, but impose a constrained distance between the MEOH's Oxygen and the MMP's Phosphorus. A worked example is at ~/QMMM/CH3Cl_hydroxide/7_atoms_10_waters/calculate.ipynb


In the folder QM_MEOH_MMP_water, repeat QM_MEOH_MMP, but including the effects of MM water molecules. See an example at ~/QMMM/H2O_hydroxide/1_H2O_optimization/minimize.ipynb