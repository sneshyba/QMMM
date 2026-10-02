import numpy as np
from pyscf import gto
# ============================================================
# GROMACS atom-name -> chemical element
# ============================================================

ELEMENT_MAP = {

    # --------------------------------------------------------
    # MMP
    # --------------------------------------------------------
    "P":   "P",
    "OME": "O",
    "OOH": "O",
    "O1":  "O",
    "O2":  "O",
    "O3":  "O",
    "C":   "C",
    "H":   "H",

    # --------------------------------------------------------
    # MEOH
    # --------------------------------------------------------
    "CM":  "C",
    "OM":  "O",
    "HO":  "H",
    "HM1": "H",
    "HM2": "H",
    "HM3": "H",

    # --------------------------------------------------------
    # TIP4P-D water
    # --------------------------------------------------------
    "OW":  "O",
    "HW2": "H",
    "HW3": "H",

    # MW4 deliberately NOT included.
    #
    # MW4 is a virtual site, not a real atom.
    # It is handled through MM_CHARGES below.
}


# ============================================================
# MM point charges
# ============================================================

MM_CHARGES = {

    # --------------------------------------------------------
    # TIP4P-D water
    # --------------------------------------------------------
    "OW":  0.00,
    "HW2": +0.58,
    "HW3": +0.58,
    "MW4": -1.16,

    # --------------------------------------------------------
    # Na
    #
    # Add the actual charge used in your GROMACS topology.
    # For Na+ this is normally +1.0.
    # --------------------------------------------------------
    "NA": +1.00,
}


# ============================================================
# Read GROMACS .gro file
# ============================================================

def read_gro(
    gro_file,
    qm_residues=("MMP", "MEOH"),
    mm_residues=("NA", "SOL"),
):
    """
    Read a GROMACS .gro file and separate the system into
    QM atoms and MM point-charge sites.

    Parameters
    ----------
    gro_file : str
        Path to the GROMACS .gro file.

    qm_residues : tuple/list
        Residue names to treat as QM.

        Example:
            ("MMP", "MEOH")

    mm_residues : tuple/list
        Residue names to treat as MM.

        Example:
            ("NA", "SOL")

    Returns
    -------
    qm_atoms : str
        PySCF atom specification.

    mm_coords : list
        MM coordinates in Angstrom.

    mm_charges : list
        MM point charges.

    mm_names : list
        GROMACS atom names corresponding to MM sites.

    mm_residue_names : list
        GROMACS residue names corresponding to MM sites.

    Notes
    -----
    GROMACS .gro coordinates are in nm.
    They are converted to Angstrom for PySCF.

    TIP4P-D MW4 is a virtual site. It is not included
    in qm_atoms, but its point charge is included in
    the MM environment.
    """

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------

    qm_atoms = []

    mm_coords = []
    mm_charges = []
    mm_names = []
    mm_residue_names = []

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    with open(gro_file, "r") as f:
        lines = f.readlines()

    # --------------------------------------------------------
    # Number of atoms
    # --------------------------------------------------------

    natoms = int(lines[1].strip())

    # --------------------------------------------------------
    # Basic sanity check
    # --------------------------------------------------------

    if len(lines) < natoms + 3:
        raise ValueError(
            f"{gro_file} appears to contain fewer atom lines "
            f"than specified by natoms = {natoms}"
        )

    # --------------------------------------------------------
    # Loop over atoms
    # --------------------------------------------------------

    for line in lines[2:2 + natoms]:

        # ====================================================
        # GROMACS .gro fixed-width fields
        # ====================================================

        resid = line[0:5].strip()
        resname = line[5:10].strip()
        atom_name = line[10:15].strip()

        # Coordinates in nm
        x_nm = float(line[20:28])
        y_nm = float(line[28:36])
        z_nm = float(line[36:44])

        # Convert nm -> Angstrom
        x = 10.0 * x_nm
        y = 10.0 * y_nm
        z = 10.0 * z_nm

        coord = [x, y, z]

        # ====================================================
        # QM REGION
        # ====================================================

        if resname in qm_residues:

            # ------------------------------------------------
            # Virtual sites cannot be QM atoms
            # ------------------------------------------------

            if atom_name == "MW4":
                raise ValueError(
                    f"Virtual site MW4 encountered in QM residue "
                    f"{resname}, residue {resid}."
                )

            # ------------------------------------------------
            # Check atom name
            # ------------------------------------------------

            if atom_name not in ELEMENT_MAP:
                raise ValueError(
                    f"Unknown QM atom name '{atom_name}' "
                    f"in residue {resname}, residue {resid}."
                )

            element = ELEMENT_MAP[atom_name]

            # ------------------------------------------------
            # PySCF atom specification
            # ------------------------------------------------

            qm_atoms.append(
                f"{element} "
                f"{x:.8f} "
                f"{y:.8f} "
                f"{z:.8f}"
            )

        # ====================================================
        # MM REGION
        # ====================================================

        elif resname in mm_residues:

            # ------------------------------------------------
            # MM point charge
            # ------------------------------------------------

            if atom_name in MM_CHARGES:

                mm_coords.append(coord)
                mm_charges.append(MM_CHARGES[atom_name])
                mm_names.append(atom_name)
                mm_residue_names.append(resname)

            # ------------------------------------------------
            # Known real atom but no MM charge
            # ------------------------------------------------

            elif atom_name in ELEMENT_MAP:

                raise ValueError(
                    f"No MM charge defined for atom "
                    f"'{atom_name}' in residue {resname}, "
                    f"residue {resid}."
                )

            # ------------------------------------------------
            # Unknown atom
            # ------------------------------------------------

            else:

                raise ValueError(
                    f"Unknown MM atom name '{atom_name}' "
                    f"in residue {resname}, residue {resid}."
                )

        # ====================================================
        # IGNORE OTHER RESIDUES
        # ====================================================

        else:

            continue

    # ========================================================
    # Return
    # ========================================================

    return (
        "\n".join(qm_atoms),
        mm_coords,
        mm_charges,
        mm_names,
        mm_residue_names,
    )

def pdb_to_gro(pdb_file, gro_file, residue_name="MOL", box_padding=1.0):
    """
    Convert a PDB file to a GROMACS .gro file.

    Parameters
    ----------
    pdb_file : str
        Input PDB filename.
    gro_file : str
        Output GRO filename.
    residue_name : str
        Residue name to use in the GRO file.
    box_padding : float
        Padding in nm added around the molecule in each direction.
    """

    atoms = []

    with open(pdb_file) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom_number = int(line[6:11])
            atom_name = line[12:16].strip()

            # PDB coordinates are in Å
            x = float(line[30:38]) / 10.0
            y = float(line[38:46]) / 10.0
            z = float(line[46:54]) / 10.0

            atoms.append((atom_number, atom_name, x, y, z))

    # Determine molecular bounding box
    xs = [a[2] for a in atoms]
    ys = [a[3] for a in atoms]
    zs = [a[4] for a in atoms]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    # Shift molecule so its minimum coordinate is box_padding
    shift_x = box_padding - xmin
    shift_y = box_padding - ymin
    shift_z = box_padding - zmin

    box_x = (xmax - xmin) + 2 * box_padding
    box_y = (ymax - ymin) + 2 * box_padding
    box_z = (zmax - zmin) + 2 * box_padding

    with open(gro_file, "w") as f:

        f.write("Converted from PDB\n")
        f.write(f"{len(atoms):5d}\n")

        for i, (atom_number, atom_name, x, y, z) in enumerate(atoms, start=1):

            x += shift_x
            y += shift_y
            z += shift_z

            # GRO fixed-width format
            f.write(
                f"{1:5d}"
                f"{residue_name:<5s}"
                f"{atom_name:>5s}"
                f"{i:5d}"
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}\n"
            )

        f.write(f"{box_x:10.5f}{box_y:10.5f}{box_z:10.5f}\n")