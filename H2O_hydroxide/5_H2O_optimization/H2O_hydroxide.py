import plotly.graph_objects as go
import numpy as np
import MDAnalysis as mda
from scipy.spatial.transform import Rotation
from pyscf import scf, qmmm, gto
HARTREE_TO_KJMOL = 2625.49962

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

def total_qm_water_energy(x, water_H1_ref, water_H2_ref, water_M_ref, mol, energy_qm_kjmol, qm_coords, qm_atom_types, lj_params, verbose=False):

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

    # with contextlib.redirect_stdout(io.StringIO()):
    #     energy_qmmm_hartree = mf_qmmm.kernel()
    
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

def get_water_ref():
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
    return water_H1_ref,water_H2_ref, water_M_ref

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