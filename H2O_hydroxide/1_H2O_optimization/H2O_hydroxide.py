import plotly.graph_objects as go
import numpy as np
import MDAnalysis as mda
from scipy.spatial.transform import Rotation
from pyscf import scf, qmmm, gto
import py3Dmol
import pickle
HARTREE_TO_KJMOL = 2625.49962
default_radius = 0.2
# default_radius = 0.1

water_H1_ref = np.array([
     0.757,
     0.586,
     0.000
])

water_H2_ref = np.array([
    -0.757,
     0.586,
     0.000
])

water_M_ref = np.array([
     0.000,
     0.151,
     0.000
])

def get_water_ref():
    
    # water_H1_ref = np.array([
    #      0.757,
    #      0.586,
    #      0.000
    # ])
    
    # water_H2_ref = np.array([
    #     -0.757,
    #      0.586,
    #      0.000
    # ])
    
    # water_M_ref = np.array([
    #      0.000,
    #      0.151,
    #      0.000
    # ])
    return water_H1_ref,water_H2_ref, water_M_ref


# ============================================================
# Lennard-Jones energy
# ============================================================

def lj_energy(
    qm_coords,
    qm_atom_types,
    mm_coords,
    mm_atom_types,
    lj_params
):
    """
    Calculate QM-MM Lennard-Jones energy.

    Parameters
    ----------
    qm_coords : (N,3) array
        QM coordinates in Angstrom.

    qm_atom_types : list
        LJ atom type for each QM atom.

    mm_coords : (M,3) array
        MM LJ-site coordinates in Angstrom.

    mm_atom_types : list
        LJ atom type for each MM particle.

    lj_params : dict
        Dictionary containing sigma (Angstrom) and
        epsilon (kJ/mol) for each atom type.

    Returns
    -------
    total : float
        Total QM-MM LJ energy in kJ/mol.
    """

    total = 0.0

    # print()
    # print("Individual QM-MM LJ interactions")
    # print("----------------------------------------")

    for i, r_qm in enumerate(qm_coords):

        for j, r_mm in enumerate(mm_coords):

            qm_type = qm_atom_types[i]
            mm_type = mm_atom_types[j]

            r = np.linalg.norm(r_qm - r_mm)

            if r == 0.0:
                raise ValueError(
                    f"Zero distance between QM atom {i} "
                    f"and MM atom {j}"
                )

            sigma_qm = lj_params[qm_type]["sigma_A"]
            epsilon_qm = lj_params[qm_type]["epsilon_kJmol"]

            sigma_mm = lj_params[mm_type]["sigma_A"]
            epsilon_mm = lj_params[mm_type]["epsilon_kJmol"]

            # Lorentz-Berthelot mixing rules
            sigma = 0.5 * (sigma_qm + sigma_mm)
            epsilon = np.sqrt(epsilon_qm * epsilon_mm)

            sr6 = (sigma / r) ** 6
            sr12 = sr6 ** 2

            e = 4.0 * epsilon * (sr12 - sr6)

            total += e

            # print(
            #     f"QM {i:2d} ({qm_type:6s})  "
            #     f"MM {j:2d} ({mm_type:6s})  "
            #     f"r = {r:8.4f} Å   "
            #     f"E = {e:14.6f} kJ/mol"
            # )

    return total


# ============================================================
# Total QM/MM + LJ interaction energy
# ============================================================

def total_qm_water_energy(x, mol, energy_qm_kjmol, qm_coords, qm_atom_types, lj_params, verbose=False):

    # --------------------------------------------------------
    # Build TIP4P-D water
    # --------------------------------------------------------
    water_O, water_H1, water_H2, water_M = \
        water_from_6dof(
            x,
            water_H1_ref,
            water_H2_ref,
            water_M_ref
        )
    
    # --------------------------------------------------------
    # TIP4P-D electrostatic sites
    # --------------------------------------------------------

    mm_charge_coords = np.array([
        water_H1,
        water_H2,
        water_M
    ])

    mm_charges = np.array([
        +0.58,
        +0.58,
        -1.16
    ])

    # --------------------------------------------------------
    # PySCF QM/MM electrostatics
    # --------------------------------------------------------

    mf_qm = scf.RHF(mol)
    mf_qm.verbose = 0
    
    mf_qmmm = qmmm.mm_charge(
        mf_qm,
        mm_charge_coords,
        mm_charges,
        unit='Angstrom'
    )

    mf_qmmm.verbose = 0
    energy_qmmm_hartree = mf_qmmm.kernel()    
    energy_qmmm_kjmol = (
        energy_qmmm_hartree * HARTREE_TO_KJMOL
    )

    # --------------------------------------------------------
    # QM/MM electrostatic interaction
    # --------------------------------------------------------

    energy_electrostatic = (
        energy_qmmm_kjmol
        - energy_qm_kjmol
    )

    # --------------------------------------------------------
    # LJ
    #
    # Only TIP4P oxygen is an LJ site.
    # --------------------------------------------------------

    mm_lj_coords = np.array([
        water_O
    ])

    mm_lj_types = [
        "TIP4P_O"
    ]

    energy_lj = lj_energy(
        qm_coords,
        qm_atom_types,
        mm_lj_coords,
        mm_lj_types,
        lj_params
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    energy_total = (
        energy_electrostatic
        + energy_lj
    )

    if verbose:

        print()
        print("--------------------------------------------")
        print("QM/MM water energy")
        print("--------------------------------------------")
        print(f"Electrostatic = {energy_electrostatic:12.6f} kJ/mol")
        print(f"LJ            = {energy_lj:12.6f} kJ/mol")
        print(f"Total         = {energy_total:12.6f} kJ/mol")
        print("--------------------------------------------")

    return energy_total
    
def savexyz(xyzfile, x0_xyz, mol, result, water_H1_ref, water_H2_ref, water_M_ref):

    with open(xyzfile, "w") as f:
    
        for label, water_xyz in [
            ("Initial", x0_xyz),
            ("Optimized", result.x)
        ]:
    
            qm_coords_A = mol.atom_coords(unit="Angstrom")
    
            O, H1, H2, M = water_from_6dof(
                water_xyz,
                water_H1_ref,
                water_H2_ref,
                water_M_ref
            )
    
            qm_symbols = [
                mol.atom_symbol(i)
                for i in range(mol.natm)
            ]
    
            symbols = qm_symbols + ["O", "H", "H"]
    
            coords = np.vstack([
                qm_coords_A,
                O,
                H1,
                H2
            ])
    
            f.write(f"{len(symbols)}\n")
            f.write(f"{label} QM/MM geometry\n")
    
            for symbol, xyz in zip(symbols, coords):
                f.write(
                    f"{symbol:2s} "
                    f"{xyz[0]:12.6f} "
                    f"{xyz[1]:12.6f} "
                    f"{xyz[2]:12.6f}\n"
                )
            
            
def visualize_H2O_hydroxide(u):

    # ------------------------------------------------------------
    # Atom positions
    # ------------------------------------------------------------
    
    u.trajectory[0]
    positions = u.atoms.positions.copy()
    
    
    # ============================================================
    # Bonds
    # ============================================================
    
    bonds = [
        (0, 1),   # QM O-H
        (2, 3),   # water O-H1
        (2, 4)    # water O-H2
    ]
    
    
    def make_bonds(positions):
    
        x = []
        y = []
        z = []
    
        for i, j in bonds:
            x += [positions[i, 0], positions[j, 0], None]
            y += [positions[i, 1], positions[j, 1], None]
            z += [positions[i, 2], positions[j, 2], None]
    
        return x, y, z
    
    
    # ============================================================
    # Save coordinates for all frames
    # ============================================================
    
    coordinates = []
    
    for ts in u.trajectory:
        coordinates.append(u.atoms.positions.copy())
    
    
    # ============================================================
    # Initial frame
    # ============================================================
    
    pos = coordinates[0]
    
    bond_x, bond_y, bond_z = make_bonds(pos)
    
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="markers+text",
                text=u.atoms.names,
                marker=dict(size=8),
                name="Atoms"
            ),
    
            go.Scatter3d(
                x=bond_x,
                y=bond_y,
                z=bond_z,
                mode="lines",
                line=dict(width=6),
                name="Bonds"
            )
        ]
    )
    
    fig.frames = [
        go.Frame(
            name=str(i),
            data=[
                go.Scatter3d(
                    x=pos[:, 0],
                    y=pos[:, 1],
                    z=pos[:, 2],
                    mode="markers+text",
                    text=u.atoms.names,
                    marker=dict(size=8)
                ),
    
                go.Scatter3d(
                    x=make_bonds(pos)[0],
                    y=make_bonds(pos)[1],
                    z=make_bonds(pos)[2],
                    mode="lines",
                    line=dict(width=6)
                )
            ]
        )
        for i, pos in enumerate(coordinates)
    ]
    
    
    # ============================================================
    # Calculate coordinate ranges over ALL frames
    # ============================================================
    
    all_xyz = np.concatenate(coordinates)
    
    xmin, xmax = all_xyz[:, 0].min(), all_xyz[:, 0].max()
    ymin, ymax = all_xyz[:, 1].min(), all_xyz[:, 1].max()
    zmin, zmax = all_xyz[:, 2].min(), all_xyz[:, 2].max()
    
    # ------------------------------------------------------------
    # Add buffer
    # ------------------------------------------------------------
    
    pad_fraction = 0.20
    
    xcenter = 0.5 * (xmin + xmax)
    ycenter = 0.5 * (ymin + ymax)
    zcenter = 0.5 * (zmin + zmax)
    
    xspan = xmax - xmin
    yspan = ymax - ymin
    zspan = zmax - zmin
    
    # Use the largest dimension for ALL three axes
    span = max(xspan, yspan, zspan)
    
    # Add 20% buffer on each side
    span *= (1 + 2 * pad_fraction)
    
    # ============================================================
    # Layout
    # ============================================================
    
    fig.update_layout(
    
        width=900,
        height=900,
        sliders=[
            {
                "active": 0,
    
                "currentvalue": {
                    "prefix": "Geometry: "
                },
    
                "steps": [
                    {
                        "label": "Initial",
                        "method": "animate",
                        "args": [
                            ["0"],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": True
                                },
                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    },
    
                    {
                        "label": "Optimized",
                        "method": "animate",
                        "args": [
                            ["1"],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": True
                                },
                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    }
                ]
            }
        ],
    
        scene=dict(
    
            # This makes the physical x/y/z scales equal
            aspectmode="cube",
    
            camera=dict(
                projection=dict(
                    type="orthographic"
                )
            ),
    
            xaxis=dict(
                title="X (Å)",
                range=[
                    xcenter - span / 2,
                    xcenter + span / 2
                ]
            ),
    
            yaxis=dict(
                title="Y (Å)",
                range=[
                    ycenter - span / 2,
                    ycenter + span / 2
                ]
            ),
    
            zaxis=dict(
                title="Z (Å)",
                range=[
                    zcenter - span / 2,
                    zcenter + span / 2
                ]
            )
        )
    )
    
    fig.show()

def water_from_6dof(
    x,
    water_H1_ref,
    water_H2_ref,
    water_M_ref
):
    """
    Convert 6 rigid-body variables into TIP4P-D coordinates.

    x[0:3] = oxygen position in Angstrom
    x[3:6] = rotation vector in radians
    """

    O = np.asarray(x[:3])

    rotvec = np.asarray(x[3:6])
    R = Rotation.from_rotvec(rotvec)

    H1 = O + R.apply(water_H1_ref)
    H2 = O + R.apply(water_H2_ref)
    M  = O + R.apply(water_M_ref)

    return O, H1, H2, M


def finite_difference_gradient_check(result,mol,energy_qm_kjmol,qm_coords,qm_atom_types,lj_params):
    print()
    print("======================================================")
    print("       FINITE-DIFFERENCE GRADIENT CHECK")
    print("======================================================")
    
    x_opt = result.x.copy()
    
    delta = 1.0e-2   # Å for translation, radians for rotation
    
    for i in range(6):
    
        xp = x_opt.copy()
        xm = x_opt.copy()
    
        xp[i] += delta
        xm[i] -= delta
    
        Ep = total_qm_water_energy(
            xp,
            mol,
            energy_qm_kjmol,
            qm_coords,
            qm_atom_types,
            lj_params
        )
    
        Em = total_qm_water_energy(
            xm,
            mol,
            energy_qm_kjmol,
            qm_coords,
            qm_atom_types,
            lj_params
        )
    
        numerical_gradient = (Ep - Em) / (2.0 * delta)
    
        print(
            f"{i}:  "
            f"finite difference = {numerical_gradient: .8f}   "
            f"L-BFGS-B = {result.jac[i]: .8f}"
        )

def energy_breakdown_at_optimized_geometry(result,mol,energy_qm_kjmol,qm_coords,qm_atom_types,lj_params):
    # ============================================================
    # Energy breakdown at optimized geometry
    # ============================================================
    O_opt, H1_opt, H2_opt, M_opt = \
        water_from_6dof(
            result.x,
            water_H1_ref,
            water_H2_ref,
            water_M_ref
        )
    print("Optimized water:")
    print("O :", O_opt)
    print("H1:", H1_opt)
    print("H2:", H2_opt)
    print("M :", M_opt)
    
    # ------------------------------------------------------------
    # Electrostatic interaction
    # ------------------------------------------------------------
    
    mm_charge_coords = np.array([
        H1_opt,
        H2_opt,
        M_opt
    ])
    
    mm_charges = np.array([
        +0.58,
        +0.58,
        -1.16
    ])
    
    mf_qm = scf.RHF(mol)
    mf_qm.verbose = 0
    
    mf_qmmm = qmmm.mm_charge(
        mf_qm,
        mm_charge_coords,
        mm_charges,
        unit='Angstrom'
    )
    
    mf_qmmm.verbose = 0
    
    energy_qmmm_hartree = mf_qmmm.kernel()
    
    energy_qmmm_kjmol = (
        energy_qmmm_hartree * HARTREE_TO_KJMOL
    )
    
    E_elec = (
        energy_qmmm_kjmol
        - energy_qm_kjmol
    )
    
    # ------------------------------------------------------------
    # LJ interaction
    # ------------------------------------------------------------
    
    E_LJ = lj_energy(
        qm_coords,
        qm_atom_types,
        np.array([O_opt]),
        ["TIP4P_O"],
        lj_params
    )
    
    # ------------------------------------------------------------
    # Total
    # ------------------------------------------------------------
    
    E_total = E_elec + E_LJ
    
    print()
    print("Energy breakdown:")
    print(f"Electrostatic = {E_elec:12.6f} kJ/mol")
    print(f"LJ            = {E_LJ:12.6f} kJ/mol")
    print(f"Total         = {E_total:12.6f} kJ/mol")

    return O_opt, H1_opt, H2_opt, M_opt

def run_qmmm_scf(H1, H2, M, mol):

    mm_charge_coords = np.array([
        H1,
        H2,
        M
    ])

    mm_charges = np.array([
        +0.58,
        +0.58,
        -1.16
    ])

    mf = scf.RHF(mol)
    mf.verbose = 0

    mf_qmmm = qmmm.mm_charge(
        mf,
        mm_charge_coords,
        mm_charges,
        unit='Angstrom'
    )

    mf_qmmm.verbose = 0

    energy = mf_qmmm.kernel()

    return mf_qmmm, energy

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

def subtract_cube_files(
    cube_initial,
    cube_optimized,
    cube_difference
):
    """
    Calculate:

        difference = optimized - initial

    for two Gaussian cube files.

    Assumes the two files have identical grids.
    """

    # --------------------------------------------------------
    # Read the cube files
    # --------------------------------------------------------

    with open(cube_initial, "r") as f:
        lines_initial = f.readlines()

    with open(cube_optimized, "r") as f:
        lines_optimized = f.readlines()

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if len(lines_initial) != len(lines_optimized):
        raise ValueError(
            "Cube files have different numbers of lines."
        )

    # Cube header:
    #   lines 0-1 = comments
    #   line 2    = number of atoms + origin
    #   lines 3-5 = grid information
    #   following lines = atom records
    #
    # We need to determine where the volumetric data starts.

    natoms_initial = abs(int(lines_initial[2].split()[0]))
    natoms_optimized = abs(int(lines_optimized[2].split()[0]))

    if natoms_initial != natoms_optimized:
        raise ValueError(
            "Cube files have different numbers of atoms."
        )

    data_start = 6 + natoms_initial

    # --------------------------------------------------------
    # Check that the headers/grids are identical
    # --------------------------------------------------------

    # Origin + grid definitions
    for i in range(2, 6):

        a = np.array(
            [float(x) for x in lines_initial[i].split()]
        )

        b = np.array(
            [float(x) for x in lines_optimized[i].split()]
        )

        if not np.allclose(a, b, atol=1e-10):
            raise ValueError(
                f"Cube grids differ on header line {i}.\n"
                f"Initial:   {lines_initial[i]}"
                f"Optimized: {lines_optimized[i]}"
            )

    print("Cube grids match.")

    # --------------------------------------------------------
    # Check atom records
    # --------------------------------------------------------

    for i in range(6, data_start):

        if lines_initial[i].strip() != lines_optimized[i].strip():

            print(
                f"WARNING: atom records differ on line {i}"
            )

    # --------------------------------------------------------
    # Extract volumetric data
    # --------------------------------------------------------

    initial_values = np.array(
        [
            float(value)
            for line in lines_initial[data_start:]
            for value in line.split()
        ]
    )

    optimized_values = np.array(
        [
            float(value)
            for line in lines_optimized[data_start:]
            for value in line.split()
        ]
    )

    if len(initial_values) != len(optimized_values):
        raise ValueError(
            "Cube files contain different numbers of "
            "volumetric data points."
        )

    print(
        "Number of grid points:",
        len(initial_values)
    )

    # --------------------------------------------------------
    # Calculate difference
    # --------------------------------------------------------

    difference_values = (
        optimized_values - initial_values
    )

    print(
        "Maximum |difference|:",
        np.max(np.abs(difference_values))
    )

    # --------------------------------------------------------
    # Reconstruct cube file
    # --------------------------------------------------------

    with open(cube_difference, "w") as f:

        # Copy everything before the volumetric data
        for line in lines_initial[:data_start]:
            f.write(line)

        # Cube files conventionally contain up to 6
        # numbers per line.

        for i in range(0, len(difference_values), 6):

            chunk = difference_values[i:i+6]

            f.write(
                " ".join(
                    f"{value:13.5E}"
                    for value in chunk
                )
                + "\n"
            )

    print()
    print("Wrote:")
    print(cube_difference)

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

def cube_minmax(filename):

    with open(filename, "r") as f:
        lines = f.readlines()

    natoms = abs(int(lines[2].split()[0]))
    data_start = 6 + natoms

    values = np.array([
        float(value)
        for line in lines[data_start:]
        for value in line.split()
    ])

    print(filename)
    print("min =", values.min())
    print("max =", values.max())
    print("max |value| =", np.max(np.abs(values)))