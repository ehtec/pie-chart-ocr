import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch

points = [(17, 158),(15, 135),(38, 183),(43, 19),(93, 88),(96, 140),(149, 163),(128, 248),(216, 265),(248, 210),(223, 167),(256, 151),(331, 214),(340, 187),(316, 53),(298, 35),(182, 0),(121, 42)]

alpha = 0.95 * alphashape.optimizealpha(points)
hull = alphashape.alphashape(points, alpha)
hull_pts = hull.exterior.coords.xy

fig, ax = plt.subplots()
ax.scatter(hull_pts[0], hull_pts[1], color='red')
ax.add_patch(PolygonPatch(hull, fill=False, color='green'))
plt.show()
