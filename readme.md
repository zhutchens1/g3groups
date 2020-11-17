# RESOLVE-G3 (Gas in Galaxy Groups) Group Finding Algorithm and Catalogs

Intro first paragraph...


## Using the Group Catalogs
<details><summary>Click for details...</summary>

</details>

## Finding Giant-Only Cores of Groups
<details><summary>Click for details...</summary>

Some details on this...


</details>

## Associating Dwarf Galaxies to Giant-Only Groups
<details><summary>Click for details...</summary>

Some details on this...


</details>

## Finding Dwarf-Only Groups
<details><summary>Click for details...</summary>

With dwarf galaxies now associated to giant-only groups, we have a catalog of "giant+dwarf" groups, and the remaining step in the group finder is to search for dwarf-only groups -- groups that would have been missed because they do not contain a giant galaxy to be associated with. We have written an algorithm called "iterative combination" to perform this step, which is contained in the `iterativecombination.py` file. This algorithm uses an iterative approach, trying to merge nearest-neighbor pairs of "potential groups" based on the sizes of similarly-luminous giant+dwarf groups. The steps of this algorithm are:

1. Assign all ungrouped dwarfs (following step 2: association) to N=1 "potential" groups.
2. Use a k-d tree to identify pairs of nearest-neighbor potential groups (i.e., a pair of potential groups where each group is a NN to the other).
3. For every nearest-neighbor pair, check if the pair should be merged into a single group:
 -Ccmpute the integrated r-band absolute magnitude of all member galaxies belonging to the pair.
 -Determine the ~98th percentile of individual galaxy projected radii and peculiar velociies, `r_proj` and `dv_proj`, observed in giant+dwarf groups (identified in step 2) of the same group-integrated luminosity.
 -If all individual galaxies shared between the nearest-neighbor of potential groups can fit within the boundaries `r_proj` and `dv_proj`, computed from the center of the two potential groups, then we merge them into a single group. Else, we leave them alone.
4. Repeat from (2) until the dwarf-only group catalog has converged, when the potential groups are no longer merging between interations.







</details>
