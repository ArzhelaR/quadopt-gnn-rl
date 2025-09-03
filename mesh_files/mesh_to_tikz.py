def msh_to_tikz(msh_file, tikz_file):
    nodes = {}
    edges = []

    with open(msh_file, "r") as f:
        lines = [line.strip() for line in f]

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Noeuds ---
        if line == "$Nodes":
            next_line = lines[i + 1]
            if " " in next_line:  # Format 4.x
                header = list(map(int, next_line.split()))
                numEntityBlocks, totalNumNodes = header[0], header[1]
                k = i + 2
                for _ in range(numEntityBlocks):
                    entType, entTag, param, numNodesBlock = map(int, lines[k].split())
                    k += 1
                    node_ids = [int(lines[k + j]) for j in range(numNodesBlock)]
                    k += numNodesBlock
                    for j in range(numNodesBlock):
                        x, y, z = map(float, lines[k].split())
                        nodes[node_ids[j]] = (x, y)
                        k += 1
                i = k
            else:  # Format 2.x
                n = int(next_line)
                for j in range(n):
                    parts = lines[i + 2 + j].split()
                    nid, x, y, z = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[nid] = (x, y)
                i += n + 2

        # --- Éléments ---
        elif line == "$Elements":
            header = list(map(int, lines[i + 1].split()))
            numEntityBlocks, totalNumElements = header[0], header[1]
            k = i + 2
            for _ in range(numEntityBlocks):
                entType, entTag, param, numElemBlock = map(int, lines[k].split())
                k += 1
                for _ in range(numElemBlock):
                    parts = list(map(int, lines[k].split()))
                    elemId, conn = parts[0], parts[1:]

                    if entType == 1:  # arêtes
                        edges.append((conn[0], conn[1]))

                    elif entType == 2:  # triangle
                        edges.extend([
                            (conn[0], conn[1]),
                            (conn[1], conn[2]),
                            (conn[2], conn[0])
                        ])

                    elif entType == 3:  # quadrangle
                        edges.extend([
                            (conn[0], conn[1]),
                            (conn[1], conn[2]),
                            (conn[2], conn[3]),
                            (conn[3], conn[0])
                        ])

                    k += 1
            i = k

        else:
            i += 1

    # --- Génération TikZ ---
    tikz = ["\\begin{tikzpicture}[scale=1]"]

    # Noeuds avec nom N1, N2, ...
    for nid, (x, y) in nodes.items():
        tikz.append(f"  \\fill ({x:.3f},{y:.3f}) circle (0.02);")
        tikz.append(f"  \\node[above right] at ({x:.3f},{y:.3f}) {{N{nid}}};")

    # Arêtes (éviter les doublons)
    seen = set()
    for n1, n2 in edges:
        if n1 in nodes and n2 in nodes:
            edge = tuple(sorted((n1, n2)))
            if edge not in seen:
                seen.add(edge)
                x1, y1 = nodes[n1]
                x2, y2 = nodes[n2]
                tikz.append(f"  \\draw ({x1:.3f},{y1:.3f}) -- ({x2:.3f},{y2:.3f});")

    tikz.append("\\end{tikzpicture}")

    with open(tikz_file, "w") as f:
        f.write("\n".join(tikz))


if __name__ == "__main__":

    # Exemple d’utilisation :
    msh_to_tikz("test_fig.msh", "mesh_output.tex")
