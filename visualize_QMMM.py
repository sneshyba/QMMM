import plotly.graph_objects as go
import numpy as np
import MDAnalysis as mda
from scipy.spatial.transform import Rotation
from pyscf import scf, qmmm, gto
import py3Dmol
import pickle
HARTREE_TO_KJMOL = 2625.49962
default_radius = 0.2


def show_orbital(view,orbital_filenames,orbital_to_display,mol,autoscale=False):

    if isinstance(orbital_filenames,list):
        cube_file = orbital_filenames[orbital_to_display]
    else:
        cube_file = orbital_filenames

    print()
    print("Displaying orbital:", orbital_to_display)
    print("Cube file:", cube_file)
    
    
    # ============================================================
    # Read cube file
    # ============================================================
    
    with open(cube_file, "r") as f:
        cube_data = f.read()
    
    
    # ============================================================
    # Read molecular geometry
    # ============================================================
    
    xyz = mol.atom_coords(unit="Angstrom")
    
    symbols = [
        mol.atom_symbol(i)
        for i in range(mol.natm)
    ]
    
    xyz_string = ""
    
    for symbol, coord in zip(symbols, xyz):
        xyz_string += (
            f"{symbol} "
            f"{coord[0]:.6f} "
            f"{coord[1]:.6f} "
            f"{coord[2]:.6f}\n"
        )
    
    
    # ============================================================
    # Display with Py3Dmol
    # ============================================================
    
    # Add molecular structure
    view.addModel(
        xyz_string,
        "xyz"
    )
    
    view.setStyle(
        {},
        {
            "stick": {},
            "sphere": {"scale": 0.3}
        }
    )

    # Add orbital isosurface
    if autoscale:
        view.addVolumetricData(
            cube_data,
            "cube",
            {
                "color": "blue",
                "opacity": 0.9
            }
        )
        
        view.addVolumetricData(
            cube_data,
            "cube",
            {
                "color": "red",
                "opacity": 0.6
            }
        )
    else:
        view.addVolumetricData(
            cube_data,
            "cube",
            {
                "isoval": 0.05,
                "color": "blue",
                "opacity": 0.9
            }
        )
        
        view.addVolumetricData(
            cube_data,
            "cube",
            {
                "isoval": -0.05,
                "color": "red",
                "opacity": 0.6
            }
        )
        
    
    view.zoomTo()

    return



def show_MM_atoms(view,water_optimization_file,iframe):

    # ============================================================
    # Read saved water optimization
    # ============================================================
    
    with open(water_optimization_file, "rb") as f:
        water_data = pickle.load(f)
    
    x_initial = water_data["x_initial"]
    x_optimized = water_data["x_optimized"]
    
    print("Initial variables:")
    print(x_initial)
    
    print()
    print("Optimized variables:")
    print(x_optimized)
    
    
    # ============================================================
    # Get TIP4P-D reference geometry
    # ============================================================
    
    water_H1_ref, water_H2_ref, water_M_ref = get_water_ref()
    
    
    # ============================================================
    # Choose which state to display
    # ============================================================
    
    
    if iframe == 0:
        x_water = x_initial
    else:
        x_water = x_optimized
    
    
    # ============================================================
    # Reconstruct MM water coordinates
    # ============================================================
    
    O, H1, H2, M = water_from_6dof(
        x_water,
        water_H1_ref,
        water_H2_ref,
        water_M_ref
    )
    
    print()
    print("MM water coordinates:")
    print("O :", O)
    print("H1:", H1)
    print("H2:", H2)
    print("M :", M)
    
    
    # ============================================================
    # Make XYZ for the three visible MM atoms
    # ============================================================
    
    xyz_string = f"""3
    TIP4P-D water
    O  {O[0]:12.6f} {O[1]:12.6f} {O[2]:12.6f}
    H  {H1[0]:12.6f} {H1[1]:12.6f} {H1[2]:12.6f}
    H  {H2[0]:12.6f} {H2[1]:12.6f} {H2[2]:12.6f}
    """
    
    
    # ============================================================
    # Create viewer
    # ============================================================
    
    # view = py3Dmol.view(
    #     width=500,
    #     height=500
    # )
    
    
    # ============================================================
    # Add MM water
    # ============================================================
    
    view.addModel(
        xyz_string,
        "xyz"
    )
    
    view.setStyle(
        {},
        {
            "stick": {
                "radius": 0.12
            },
            "sphere": {
                "scale": default_radius
            }
        }
    )
    
    
    # ============================================================
    # Display
    # ============================================================
    
    view.zoomTo()
    return



def show_QM_atoms(view, mol):

    for i in range(mol.natm):

        X = mol.atom_coords(unit="Angstrom")[i]
        symbol = mol.atom_symbol(i)

        if symbol == "O":
            radius = 1.52 * default_radius
            color = "red"

        elif symbol == "H":
            radius = 1.20 * default_radius
            color = "white"

        else:
            radius = 1.70 * default_radius
            color = "gray"

        view.addSphere({
            "center": {
                "x": float(X[0]),
                "y": float(X[1]),
                "z": float(X[2])
            },
            "radius": radius,
            "color": color
        })

