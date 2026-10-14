use super::*;
use array_macro::array;

/// Find collisions between two polygon meshes and extract interference lines.
///
/// # Details
///
/// Candidate triangle pairs come from a bounding volume hierarchy over the
/// triangles of `other`, queried with the bounding box of each triangle of `self`.
///
/// # Remarks
///
/// We see that the surfaces that are in contact are not interfering with each other.
/// Therefore, polygon meshes included in one plane are not interfering.
pub trait Collision {
    /// If `self` and `other` collide, then returns only one interference line.
    /// Otherwise, returns `None`.
    fn collide_with(&self, other: &PolygonMesh) -> Option<(Point3, Point3)>;
    /// Extract all interference lines between `self` and `other`.
    /// # Remarks
    /// The results is not arranged so that included lines make continuous maximal polyline curve.
    fn extract_interference(&self, other: &PolygonMesh) -> Vec<(Point3, Point3)>;
}

impl Collision for PolygonMesh {
    #[inline(always)]
    fn collide_with(&self, other: &PolygonMesh) -> Option<(Point3, Point3)> {
        are_colliding(self, other)
    }
    #[inline(always)]
    fn extract_interference(&self, other: &PolygonMesh) -> Vec<(Point3, Point3)> {
        collision(self, other)
    }
}

struct Node {
    bounds: BoundingBox<Point3>,
    start: usize,
    end: usize,
    children: Option<(usize, usize)>,
}

struct Bvh {
    boxes: Vec<BoundingBox<Point3>>,
    order: Vec<usize>,
    nodes: Vec<Node>,
}

const LEAF: usize = 4;

impl Bvh {
    fn new(tris: &[[Point3; 3]], within: &BoundingBox<Point3>) -> Bvh {
        let boxes: Vec<BoundingBox<Point3>> = tris.iter().map(|tri| tri.iter().collect()).collect();
        let order: Vec<usize> = (0..tris.len())
            .filter(|&i| !degenerate(tris[i]) && !apart(&boxes[i], within))
            .collect();
        let mut bvh = Bvh {
            boxes,
            order,
            nodes: Vec::new(),
        };
        if !bvh.order.is_empty() {
            bvh.split(0, bvh.order.len());
        }
        bvh
    }

    fn split(&mut self, start: usize, end: usize) -> usize {
        let bounds: BoundingBox<Point3> = self.order[start..end]
            .iter()
            .flat_map(|&i| [self.boxes[i].min(), self.boxes[i].max()])
            .collect();
        let index = self.nodes.len();
        self.nodes.push(Node {
            bounds,
            start,
            end,
            children: None,
        });
        if end - start > LEAF {
            let size = bounds.max() - bounds.min();
            let axis = (0..3)
                .max_by(|&a, &b| size[a].total_cmp(&size[b]))
                .unwrap_or(0);
            let boxes = &self.boxes;
            let centre = |i: &usize| boxes[*i].min()[axis] + boxes[*i].max()[axis];
            let middle = (end - start) / 2;
            self.order[start..end]
                .select_nth_unstable_by(middle, |a, b| centre(a).total_cmp(&centre(b)));
            let left = self.split(start, start + middle);
            let right = self.split(start + middle, end);
            self.nodes[index].children = Some((left, right));
        }
        index
    }

    fn overlapping(&self, bounds: &BoundingBox<Point3>, found: &mut Vec<usize>) {
        found.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0];
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            if apart(&node.bounds, bounds) {
                continue;
            }
            match node.children {
                Some((left, right)) => stack.extend([left, right]),
                None => found.extend(
                    self.order[node.start..node.end]
                        .iter()
                        .filter(|&&i| !apart(&self.boxes[i], bounds)),
                ),
            }
        }
        found.sort_unstable();
    }
}

fn apart(a: &BoundingBox<Point3>, b: &BoundingBox<Point3>) -> bool {
    let gap = 1.0e-9 * f64::max(a.diameter(), b.diameter());
    (0..3).any(|k| a.max()[k] + gap < b.min()[k] || b.max()[k] + gap < a.min()[k])
}

fn degenerate(tri: [Point3; 3]) -> bool { (tri[1] - tri[0]).cross(tri[2] - tri[0]).so_small() }

fn candidate_pairs<'a>(
    tris0: &'a [[Point3; 3]],
    tris1: &'a [[Point3; 3]],
) -> impl Iterator<Item = (usize, usize)> + 'a {
    let within: BoundingBox<Point3> = tris0.iter().flatten().collect();
    let bvh = Bvh::new(tris1, &within);
    let mut found = Vec::new();
    (0..tris0.len())
        .filter(move |&i| !degenerate(tris0[i]))
        .flat_map(move |i| {
            let bounds: BoundingBox<Point3> = tris0[i].iter().collect();
            bvh.overlapping(&bounds, &mut found);
            found.iter().map(move |&j| (i, j)).collect::<Vec<_>>()
        })
}

fn disjoint_bdbs(tri0: [Point3; 3], tri1: [Point3; 3]) -> bool {
    let bdb0: BoundingBox<Point3> = tri0.iter().collect();
    let bdb1: BoundingBox<Point3> = tri1.iter().collect();
    let gap = 1.0e-12 * f64::max(bdb0.diameter(), bdb1.diameter());
    (0..3).any(|k| bdb0.max()[k] + gap < bdb1.min()[k] || bdb1.max()[k] + gap < bdb0.min()[k])
}

fn collide_seg_triangle(seg: [Point3; 2], tri: [Point3; 3]) -> Option<Point3> {
    let ab = tri[1] - tri[0];
    let bc = tri[2] - tri[1];
    let ca = tri[0] - tri[2];
    let nor = ab.cross(ca);
    if nor.so_small() {
        return None;
    }
    let ap = seg[0] - tri[0];
    let aq = seg[1] - tri[0];
    let dotapnor = ap.dot(nor);
    let dotaqnor = aq.dot(nor);
    if dotapnor * dotaqnor > 0.0 {
        return None;
    }
    let h = seg[0] + dotapnor / (dotapnor - dotaqnor) * (seg[1] - seg[0]);
    if f64::signum(ab.cross(nor).dot(h - tri[0]) + TOLERANCE2)
        + f64::signum(bc.cross(nor).dot(h - tri[1]) + TOLERANCE2)
        + f64::signum(ca.cross(nor).dot(h - tri[2]) + TOLERANCE2)
        >= 2.0
    {
        Some(h)
    } else {
        None
    }
}

fn cross_seg_triangle(seg: [Point3; 2], tri: [Point3; 3]) -> Option<Point3> {
    let ab = tri[1] - tri[0];
    let bc = tri[2] - tri[1];
    let ca = tri[0] - tri[2];
    let nor = ab.cross(ca);
    if nor.so_small() {
        return None;
    }
    let ap = seg[0] - tri[0];
    let aq = seg[1] - tri[0];
    let scale = nor.magnitude() * ap.magnitude().max(aq.magnitude()).max(f64::MIN_POSITIVE);
    let side = |d: f64| d >= -1.0e-12 * scale;
    let dotapnor = ap.dot(nor);
    let dotaqnor = aq.dot(nor);
    if side(dotapnor) == side(dotaqnor) {
        return None;
    }
    let t = (dotapnor / (dotapnor - dotaqnor)).clamp(0.0, 1.0);
    let h = seg[0] + t * (seg[1] - seg[0]);
    if f64::signum(ab.cross(nor).dot(h - tri[0]) + TOLERANCE2)
        + f64::signum(bc.cross(nor).dot(h - tri[1]) + TOLERANCE2)
        + f64::signum(ca.cross(nor).dot(h - tri[2]) + TOLERANCE2)
        >= 2.0
    {
        Some(h)
    } else {
        None
    }
}

fn triangles_meet(
    tri0: [Point3; 3],
    tri1: [Point3; 3],
    seg_hit: fn([Point3; 2], [Point3; 3]) -> Option<Point3>,
) -> Option<(Point3, Point3)> {
    let mut tuple = (None, None);
    [
        seg_hit([tri0[0], tri0[1]], tri1),
        seg_hit([tri0[1], tri0[2]], tri1),
        seg_hit([tri0[2], tri0[0]], tri1),
        seg_hit([tri1[0], tri1[1]], tri0),
        seg_hit([tri1[1], tri1[2]], tri0),
        seg_hit([tri1[2], tri1[0]], tri0),
    ]
    .into_iter()
    .for_each(|pt| match tuple {
        (None, _) => tuple.0 = pt,
        (Some(_), None) => tuple.1 = pt,
        (Some(ref mut p), Some(ref mut q)) => {
            if let Some(pt) = pt {
                let dist0 = pt.distance2(*p);
                let dist1 = pt.distance2(*q);
                let dist2 = p.distance2(*q);
                if dist2 < dist0 {
                    *q = pt;
                } else if dist2 < dist1 {
                    *p = pt;
                }
            }
        }
    });
    match tuple {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

fn collide_triangles(tri0: [Point3; 3], tri1: [Point3; 3]) -> Option<(Point3, Point3)> {
    triangles_meet(tri0, tri1, collide_seg_triangle)
}

fn cross_triangles(tri0: [Point3; 3], tri1: [Point3; 3]) -> Option<(Point3, Point3)> {
    triangles_meet(tri0, tri1, cross_seg_triangle)
}

fn positioned(poly: &PolygonMesh) -> Vec<[Point3; 3]> {
    poly.faces()
        .triangle_iter()
        .map(|face| array![i => poly.positions()[face[i].pos]; 3])
        .collect()
}

fn collision(poly0: &PolygonMesh, poly1: &PolygonMesh) -> Vec<(Point3, Point3)> {
    let (tris0, tris1) = (positioned(poly0), positioned(poly1));
    candidate_pairs(&tris0, &tris1)
        .filter_map(|(i, j)| match disjoint_bdbs(tris0[i], tris1[j]) {
            true => None,
            false => cross_triangles(tris0[i], tris1[j]),
        })
        .collect()
}

fn are_colliding(poly0: &PolygonMesh, poly1: &PolygonMesh) -> Option<(Point3, Point3)> {
    let (tris0, tris1) = (positioned(poly0), positioned(poly1));
    candidate_pairs(&tris0, &tris1).find_map(|(i, j)| match disjoint_bdbs(tris0[i], tris1[j]) {
        true => None,
        false => collide_triangles(tris0[i], tris1[j]),
    })
}

#[test]
fn collide_triangles_test() {
    let tri0 = [
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let tri1 = [
        Point3::new(0.0, 0.0, -1.0),
        Point3::new(-1.0, -1.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
    ];
    assert!(collide_triangles(tri0, tri1).is_some());
    let tri0 = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let tri1 = [
        Point3::new(0.0, 0.0, 0.5),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
    ];
    assert!(collide_triangles(tri0, tri1).is_none());
}
