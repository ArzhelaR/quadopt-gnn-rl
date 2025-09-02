# Actions validity
By performing a large amount of random actions, it can result in degenerated mesh structures. To avoid it, we defined some topological properties the mesh must preserve :

Let $d$ be a dart of the mesh and $d_2= \beta_2(d)$ denote its twin dart, if it exists. 
We say the mesh is **valid** if the following conditions hold :

* General constraint : $\beta_1^{4}(d)=d$ & $\beta_1(d)\neq\beta_2(d)$.
* Interior darts : If $d$ is an inner dart and $d_2$ exists, 
  * $ \beta_2(d)=d_2$ & $\beta_2(d_2)=d$
  * $d$ and $d_2$ share the same extremities nodes.
* Boundary darts
    If $d$ is a boundary dart and $d_2$ is none, extremity nodes should be boundary nodes.
* Faces Each face must be composed of 4 differents nodes, by other means, no plane face.

To respect every time these conditions, we restricted each actions on differents aspects.

### FLIP restriction

* A flip operation on a boundary edge is always invalid.
* If its adjacent faces share more than one edge, as shown in the figure, it breaks $\beta_1(d) \neq \beta_2(d)$ property.
* Nodes $n3$ and $n5$ (or $n6/n4$ in the clockwise direction) cannot have more than 10 incident edges.


### SPLIT restriction

* A split operation on a boundary edge is always invalid.
* If $\beta_1^{3}(d) = \beta_1(\beta_2(d))$, split action on $d$ creates a plane face.
*  Nodes $n4$ and $n2$ cannot have more than 10 incident edges.

### COLLAPSE restriction

* A collapse operation on a boundary edge is always invalid.
* $n1$ node must not be on boundary.
* The final merged node $n1$ cannot have more than 10 incident edges.
* Face $f1$ must be different from $f2$, and $f3$ from $f4$.