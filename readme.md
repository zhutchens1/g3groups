# RESOLVE-G3 (Gas in Galaxy Groups) Group Finding Algorithm and Catalogs

Intro first paragraph...


## Using the Group Catalogs
<details><summary>Click for details...</summary>

</details>

## Step 1: Finding Giant-Only Cores of Groups
<details><summary>Click for details...</summary>

In the first step of the group finder, we use friends-of-friends (FoF)  to locate groups of giant galaxies. We define giants as galaxies that are  more massive than the gas-richness threshold scale from Kannappan et al. (2013). Therefore, in the stellar mass-selected group catalog, we select giants as `logMstar >= 9.5`; in the luminosity-selected group catalog, we select giants as absolute `M_r <= -19.4`. Figure 1 shows how this mass scale divides the galaxy population between dwarfs and giants.
We also require that giant galaxies are confined to the group-finding volume, which surrounds the survey with a buffer region to mitigate errors in group-finding near the survey redshift boundaries. This constraint is `2530 < cz [km/s] < 7470` for ECO, which includes RESOLVE-A, and `4250 < cz [km/s] < 7250` for RESOLVE-B. 

[FIGURE 1]

We employ an adaptive linking strategy during this giant-only FoF procedure, inspired by Robotham et al. (2011) and its volume-limited application in Mummery (2018). We use line-of-sight `b_LOS` and transverse `b_perp` linking multipliers of 1.1 and 0.07, respectively, as these are optimized for the study of galaxy environment (Duarte & Mamon, 2014). In a standard FoF approach, these values are multiplied by the mean separation of galaxies, `s_0=(V/N)^1/3`, and are used as linking lengths. Here we assign a different value of `s` to every galaxy, measured instead by the number density of galaxies which are greater than or equal to their luminosity or mass. We then look at the median value of `s` over all galaxies and scale all `s` values such that the median is retained at the original `(V/N)^1/3`. The figure below shows how the value of `s` varies with absolute magnitude. We apply these ECO `s` values to RESOLVE-B using a model fit, since the B semester volume is subject to cosmic variance. This approach ensures that the linking length rises with galaxy luminosity/stellar mass and therefore reduces fragmentation of identified groups.

[FIGURE 2]

 




</details>

## Step 2: Associating Dwarf Galaxies to Giant-Only Groups
<details><summary>Click for details...</summary>

Some details on this...


</details>

## Step 3: Finding Dwarf-Only Groups
<details>

With dwarf galaxies now associated to giant-only groups, we have a catalog of "giant+dwarf" groups, and the remaining step in the group finder is to search for dwarf-only groups -- groups that would have been missed because they do not contain a giant galaxy to be associated with. We have written an algorithm called "iterative combination" to perform this step, which is contained in the `iterativecombination.py` file. This algorithm uses an iterative approach, trying to merge nearest-neighbor pairs of "potential groups" based on the sizes of similarly-luminous giant+dwarf groups. The steps of this algorithm are:

 1. Assign all ungrouped dwarfs (following step 2: association) to N=1 "potential" groups.
 2. Use a k-d tree to identify pairs of nearest-neighbor potential groups (i.e., a pair of potential groups where each group is a NN to the other).
 3. For every nearest-neighbor pair, check if the pair should be merged into a single group:
  * a. Compute the integrated r-band absolute magnitude of all member galaxies belonging to the pair.
  * b. Determine the ~98th percentile of individual galaxy projected radii and peculiar velociies, `r_proj` and `dv_proj`, observed in giant+dwarf groups (identified in step 2) of the same group-integrated luminosity.
  * c. If all individual galaxies shared between the nearest-neighbor of potential groups can fit within the boundaries `r_proj` and `dv_proj`, computed from the center of the two potential groups, then we merge them into a single group. Else, we leave them alone.
 4. Repeat from (2) until the dwarf-only group catalog has converged, when the potential groups are no longer merging between interations.







</details>
