def read_cube_grid_info(path):
    with open(path, "r") as f:
        line1 = f.readline()  # comment
        line2 = f.readline()  # comment
        origin_line = f.readline().split()
        # line format: natoms origin_x origin_y origin_z
        natoms = int(origin_line[0])
        origin = tuple(map(float, origin_line[1:4])) # In atomic units (bohr)

        nx_line = f.readline().split()  # nx ax ay az
        ny_line = f.readline().split()  # ny bx by bz
        nz_line = f.readline().split()  # nz cx cy cz

    nx = int(nx_line[0]); ax = tuple(map(float, nx_line[1:4]))
    ny = int(ny_line[0]); by = tuple(map(float, ny_line[1:4]))
    nz = int(nz_line[0]); cz = tuple(map(float, nz_line[1:4]))

    return {
        "natoms": natoms,
        "origin": origin,
        "nx": nx, "ny": ny, "nz": nz,
        "ax": ax, "by": by, "cz": cz,
    }

def surf_to_mesh3d(surf):
    pts = surf.points
    faces = surf.faces.reshape(-1, 4)  # [3, a, b, c]
    i = faces[:, 1]
    j = faces[:, 2]
    k = faces[:, 3]
    return pts, i, j, k

def add_mesh_with_edges(fig, pts, i, j, k, color, opacity=0.35, name=""):
    # filled surface
    fig.add_trace(go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
        name=name
    ))

    # triangle edges as line segments
    edges = np.vstack([
        np.stack([i, j], axis=1),
        np.stack([j, k], axis=1),
        np.stack([k, i], axis=1),
    ])
    # unique undirected edges
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # build line coordinates with None separators
    x = []
    y = []
    z = []
    for a, b in edges:
        x += [pts[a, 0], pts[b, 0], None]
        y += [pts[a, 1], pts[b, 1], None]
        z += [pts[a, 2], pts[b, 2], None]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color="black", width=1),
        opacity=0.9,
        hoverinfo="skip",
        showlegend=False
    ))

def add_an_orbital(orbital_filename,fig,info,isofactor=0.15,opacity_plus=.95,opacity_minus=0.05):
    print('Trying to add '+orbital_filename)
    grid = pv.read(orbital_filename)
    name = grid.array_names[0]
    vals = grid[name]
    
    vmin, vmax = float(vals.min()), float(vals.max())
    m = max(abs(vmin), abs(vmax))
    print('from add_an_orbital ... m = ', m)

    iso = isofactor * m
    iso = .1
    
    surf_pos = grid.contour([+iso], scalars=name)
    surf_neg = grid.contour([-iso], scalars=name)
    
    if surf_pos.n_points > 0:
        pts, i, j, k = surf_to_mesh3d(surf_pos)
        pts = adjust_pts(pts,info)
        add_mesh_with_edges(fig, pts, i, j, k, color="red", opacity=opacity_plus, name="+")
    if surf_neg.n_points > 0:
        pts, i, j, k = surf_to_mesh3d(surf_neg)
        pts = adjust_pts(pts,info)
        add_mesh_with_edges(fig, pts, i, j, k, color="blue", opacity=opacity_minus, name="−")
    
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return

def adjust_pts(pts,info): # works for frame = 0
    au_conversion = 0.5292
    
    pts[:,0] *= info["ax"][0]*au_conversion
    pts[:,1] *= info["by"][1]*au_conversion
    pts[:,2] *= info["cz"][2]*au_conversion

    pts[:,0] += info["origin"][0]*au_conversion
    pts[:,1] += info["origin"][1]*au_conversion
    pts[:,2] += info["origin"][2]*au_conversion
    return pts

def add_atoms(mol,fig):
    au_conversion = 0.5292
    atom_xyz = []
    for a in mol.atom:  # mol.atom is a list like [(Z, (x,y,z)), ...] depending on PySCF version
        if isinstance(a[0], str):
            sym = a[0]
            x, y, z = a[1]
        else:
            # If entry is (Z, (x,y,z))
            sym = str(a[0])
            x, y, z = a[1]
        atom_xyz.append((sym, x, y, z))
    
    # Separate coordinates
    xs = [t[1] for t in atom_xyz]
    ys = [t[2] for t in atom_xyz]
    zs = [t[3] for t in atom_xyz]
    labels = [t[0] for t in atom_xyz]
    
    xs = np.array(xs)*au_conversion
    ys = np.array(ys)*au_conversion
    zs = np.array(zs)*au_conversion
    
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers+text",
        marker=dict(size=8, color="black", opacity=.9),
        text=labels,
        textposition="top center",
        name="atoms",
        showlegend=False,
    ))


import numpy as np


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