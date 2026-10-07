use super::super::integrate::{ShapeOpsCurve, ShapeOpsSurface};
use monstertruck_geometry::prelude::*;
use monstertruck_topology::compress::{CompressedEdge, CompressedEdgeIndex, CompressedShell};
use monstertruck_topology::*;
use rustc_hash::FxHashMap as HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PlaneOf {
    origin: Point3,
    normal: Vector3,
}

pub(super) fn plane_of<C, S: TryIntoAnalyticSurfaceKind + Clone>(
    face: &Face<Point3, C, S>,
) -> Option<PlaneOf> {
    match face.surface().try_into_analytic_surface_kind()? {
        AnalyticSurfaceKind::Plane(plane) => {
            let normal = plane.normal();
            Some(PlaneOf {
                origin: plane.origin(),
                normal: if face.orientation() { normal } else { -normal },
            })
        }
        _ => None,
    }
}

pub(super) fn coplanar(a: PlaneOf, b: PlaneOf, tol: f64) -> bool {
    a.normal.cross(b.normal).magnitude() < 1.0e-7 && (b.origin - a.origin).dot(a.normal).abs() < tol
}

struct Frame {
    origin: Point3,
    u: Vector3,
    v: Vector3,
}

impl Frame {
    fn new(plane: PlaneOf) -> Frame {
        let n = plane.normal;
        let helper = if n.x.abs() < 0.9 {
            Vector3::unit_x()
        } else {
            Vector3::unit_y()
        };
        let u = n.cross(helper).normalize();
        Frame {
            origin: plane.origin,
            u,
            v: n.cross(u),
        }
    }

    fn flat(&self, p: Point3) -> Point2 {
        let d = p - self.origin;
        Point2::new(d.dot(self.u), d.dot(self.v))
    }

    fn lift(&self, p: Point2) -> Point3 { self.origin + self.u * p.x + self.v * p.y }
}

fn samples<C, S>(curve: &C, tol: f64) -> Vec<(f64, Point3)>
where C: BoundedCurve + ParameterDivision1D<Point = Point3> {
    let (params, points) = curve.parameter_division(curve.range_tuple(), tol);
    params.into_iter().zip(points).collect()
}

fn cross2(a: Vector2, b: Vector2) -> f64 { a.x * b.y - a.y * b.x }

fn segment_distance(p: Point2, a: Point2, b: Point2) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.magnitude2().max(1.0e-300)).clamp(0.0, 1.0);
    (a + ab * t).distance(p)
}

fn crossing(a0: Point2, a1: Point2, b0: Point2, b1: Point2) -> Option<(f64, f64)> {
    let (da, db) = (a1 - a0, b1 - b0);
    let denom = cross2(da, db);
    if denom.abs() < 1.0e-14 * da.magnitude() * db.magnitude() {
        return None;
    }
    let w = b0 - a0;
    let s = cross2(w, db) / denom;
    let t = cross2(w, da) / denom;
    ((0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t)).then_some((s, t))
}

fn polygon_area(points: &[Point2]) -> f64 {
    let n = points.len();
    (0..n)
        .map(|i| cross2(points[i].to_vec(), points[(i + 1) % n].to_vec()))
        .sum::<f64>()
        / 2.0
}

fn inside(polygons: &[Vec<Point2>], p: Point2) -> bool {
    let mut odd = false;
    for polygon in polygons {
        let n = polygon.len();
        for i in 0..n {
            let (a, b) = (polygon[i], polygon[(i + 1) % n]);
            if (a.y > p.y) != (b.y > p.y) {
                let x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
                if x > p.x {
                    odd = !odd;
                }
            }
        }
    }
    odd
}

fn on_outline(polygons: &[Vec<Point2>], p: Point2, tol: f64) -> bool {
    polygons.iter().any(|polygon| {
        let n = polygon.len();
        (0..n).any(|i| segment_distance(p, polygon[i], polygon[(i + 1) % n]) < tol)
    })
}

fn oriented_points<C, S>(edge: &Edge<Point3, C>, tol: f64) -> Vec<Point3>
where C: Clone + BoundedCurve + ParameterDivision1D<Point = Point3> {
    let mut points: Vec<Point3> = samples::<C, S>(&edge.curve(), tol)
        .into_iter()
        .map(|(_, p)| p)
        .collect();
    if !edge.orientation() {
        points.reverse();
    }
    points
}

fn outline<C, S>(face: &Face<Point3, C, S>, frame: &Frame, tol: f64) -> Vec<Vec<Point2>>
where C: Clone + BoundedCurve + ParameterDivision1D<Point = Point3> {
    face.boundaries()
        .iter()
        .map(|wire| {
            wire.edge_iter()
                .flat_map(|edge| {
                    let points = oriented_points::<C, S>(&edge, tol);
                    let keep = points.len().saturating_sub(1);
                    points.into_iter().take(keep).collect::<Vec<_>>()
                })
                .map(|p| frame.flat(p))
                .collect()
        })
        .collect()
}

fn interior_point(polygons: &[Vec<Point2>], tol: f64) -> Option<Point2> {
    let mut candidates: Vec<(f64, Point2, Vector2)> = polygons
        .iter()
        .flat_map(|polygon| {
            let n = polygon.len();
            (0..n).map(move |i| {
                let (a, b) = (polygon[i], polygon[(i + 1) % n]);
                ((b - a).magnitude(), a + (b - a) * 0.5, b - a)
            })
        })
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates
        .into_iter()
        .take(16)
        .find_map(|(length, middle, along)| {
            if length < tol {
                return None;
            }
            let left = Vector2::new(-along.y, along.x).normalize();
            [1.0e-3, -1.0e-3, 1.0e-2, -1.0e-2, 1.0e-1, -1.0e-1]
                .into_iter()
                .map(|k: f64| middle + left * k.signum() * (length * k.abs()).max(tol * 4.0))
                .find(|p| inside(polygons, *p) && !on_outline(polygons, *p, tol))
        })
}

type Splits = HashMap<usize, Vec<(f64, Point3)>>;

fn unique_edges<C, S>(shell: &Shell<Point3, C, S>) -> Vec<Edge<Point3, C>> {
    let mut seen = std::collections::HashSet::new();
    shell
        .face_iter()
        .flat_map(|face| {
            face.absolute_boundaries()
                .iter()
                .flat_map(|w| w.iter().cloned())
                .collect::<Vec<_>>()
        })
        .filter(|edge| seen.insert(edge.id()))
        .map(|edge| edge.absolute_clone())
        .collect()
}

fn note_split<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    splits: &mut Splits,
    index: usize,
    edge: &Edge<Point3, C>,
    point: Point3,
    tol: f64,
) {
    let curve = edge.curve();
    let (t0, t1) = curve.range_tuple();
    let (front, back) = (edge.absolute_front().point(), edge.absolute_back().point());
    if point.distance(front) < tol || point.distance(back) < tol {
        return;
    }
    let Some(t) = curve.search_nearest_parameter(point, None, 100) else {
        return;
    };
    if t <= t0 || t >= t1 || curve.subs(t).distance(point) > tol {
        return;
    }
    let list = splits.entry(index).or_default();
    if !list.iter().any(|(_, p)| p.distance(point) < tol) {
        list.push((t, curve.subs(t)));
    }
}

fn split_points<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    edges0: &[Edge<Point3, C>],
    edges1: &[Edge<Point3, C>],
    pairs: &[(usize, usize)],
    faces0: &[Face<Point3, C, S>],
    faces1: &[Face<Point3, C, S>],
    tol: f64,
) -> (Splits, Splits) {
    let index_of = |edges: &[Edge<Point3, C>], edge: &Edge<Point3, C>| {
        edges.iter().position(|e| e.id() == edge.id())
    };
    let (mut splits0, mut splits1) = (Splits::default(), Splits::default());
    for &(i, j) in pairs {
        let plane = match plane_of(&faces0[i]) {
            Some(plane) => plane,
            None => continue,
        };
        let frame = Frame::new(plane);
        let own = |face: &Face<Point3, C, S>,
                   edges: &[Edge<Point3, C>]|
         -> Vec<(usize, Vec<(f64, Point3)>, Vec<Point2>)> {
            face.edge_iter()
                .filter_map(|edge| {
                    let index = index_of(edges, &edge)?;
                    let absolute = &edges[index];
                    let points = samples::<C, S>(&absolute.curve(), tol);
                    let flat = points.iter().map(|(_, p)| frame.flat(*p)).collect();
                    Some((index, points, flat))
                })
                .collect()
        };
        let a = own(&faces0[i], edges0);
        let b = own(&faces1[j], edges1);
        for (ia, _, fa) in &a {
            for (ib, _, fb) in &b {
                for sa in fa.windows(2) {
                    for sb in fb.windows(2) {
                        if let Some((s, _)) = crossing(sa[0], sa[1], sb[0], sb[1]) {
                            let at = frame.lift(sa[0] + (sa[1] - sa[0]) * s);
                            note_split(&mut splits0, *ia, &edges0[*ia], at, tol);
                            note_split(&mut splits1, *ib, &edges1[*ib], at, tol);
                        }
                    }
                }
            }
        }
        let corners = |edges: &[Edge<Point3, C>],
                       list: &[(usize, Vec<(f64, Point3)>, Vec<Point2>)]|
         -> Vec<Point3> {
            list.iter()
                .flat_map(|(index, _, _)| {
                    [
                        edges[*index].absolute_front().point(),
                        edges[*index].absolute_back().point(),
                    ]
                })
                .collect()
        };
        for corner in corners(edges1, &b) {
            let flat = frame.flat(corner);
            for (ia, _, fa) in &a {
                if fa
                    .windows(2)
                    .any(|s| segment_distance(flat, s[0], s[1]) < tol)
                {
                    note_split(&mut splits0, *ia, &edges0[*ia], corner, tol);
                }
            }
        }
        for corner in corners(edges0, &a) {
            let flat = frame.flat(corner);
            for (ib, _, fb) in &b {
                if fb
                    .windows(2)
                    .any(|s| segment_distance(flat, s[0], s[1]) < tol)
                {
                    note_split(&mut splits1, *ib, &edges1[*ib], corner, tol);
                }
            }
        }
    }
    (splits0, splits1)
}

fn apply_splits<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &Shell<Point3, C, S>,
    edges: &[Edge<Point3, C>],
    splits: &Splits,
) -> Shell<Point3, C, S> {
    let mut pieces: HashMap<EdgeId<C>, Vec<Edge<Point3, C>>> = HashMap::default();
    for (index, edge) in edges.iter().enumerate() {
        let Some(list) = splits.get(&index) else {
            continue;
        };
        let mut list = list.clone();
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut rest = edge.curve();
        let mut from = edge.absolute_front().clone();
        let mut made = Vec::new();
        for (_, point) in list {
            let Some(t) = rest.search_nearest_parameter(point, None, 100) else {
                continue;
            };
            let (t0, t1) = rest.range_tuple();
            if t <= t0 || t >= t1 {
                continue;
            }
            let tail = rest.cut(t);
            let vertex = Vertex::new(point);
            made.push(Edge::new(&from, &vertex, rest));
            rest = tail;
            from = vertex;
        }
        made.push(Edge::new(&from, edge.absolute_back(), rest));
        pieces.insert(edge.id(), made);
    }
    if pieces.is_empty() {
        return shell.clone();
    }
    shell
        .face_iter()
        .map(|face| {
            let wires: Vec<Wire<Point3, C>> = face
                .absolute_boundaries()
                .iter()
                .map(|wire| {
                    wire.iter()
                        .flat_map(|edge| match pieces.get(&edge.id()) {
                            None => vec![edge.clone()],
                            Some(made) if edge.orientation() => made.clone(),
                            Some(made) => made.iter().rev().map(Edge::inverse).collect(),
                        })
                        .collect()
                })
                .collect();
            let mut rebuilt = Face::new_unchecked(wires, face.surface());
            if !face.orientation() {
                rebuilt.invert();
            }
            rebuilt
        })
        .collect()
}

struct HalfEdge<C> {
    edge: Edge<Point3, C>,
    from: usize,
    to: usize,
    leave: f64,
    arrive: f64,
    flat: Vec<Point2>,
}

fn same_place(a: &[Point3], b: &[Point3], tol: f64) -> bool {
    let (a0, a1) = (a[0], a[a.len() - 1]);
    let (b0, b1) = (b[0], b[b.len() - 1]);
    let ends = (a0.distance(b0) < tol && a1.distance(b1) < tol)
        || (a0.distance(b1) < tol && a1.distance(b0) < tol);
    ends && a.iter().all(|p| {
        b.windows(2).any(|s| {
            let ab = s[1] - s[0];
            let t = ((*p - s[0]).dot(ab) / ab.magnitude2().max(1.0e-300)).clamp(0.0, 1.0);
            (s[0] + ab * t).distance(*p) < tol * 10.0
        })
    })
}

fn divide<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    face: &Face<Point3, C, S>,
    others: &[&Face<Point3, C, S>],
    tol: f64,
) -> Option<Vec<(Face<Point3, C, S>, Option<usize>)>> {
    let plane = plane_of(face)?;
    let frame = Frame::new(plane);
    let own_outline = outline(face, &frame, tol);
    let mut vertices: Vec<Vertex<Point3>> = Vec::new();
    fn vertex_of(
        vertices: &mut Vec<Vertex<Point3>>,
        point: Point3,
        existing: Option<&Vertex<Point3>>,
        tol: f64,
    ) -> usize {
        if let Some(found) = vertices
            .iter()
            .position(|v| v.point().distance(point) < tol)
        {
            return found;
        }
        vertices.push(existing.cloned().unwrap_or_else(|| Vertex::new(point)));
        vertices.len() - 1
    }
    let direction = |a: Point2, b: Point2| (b.y - a.y).atan2(b.x - a.x);
    let mut halves: Vec<HalfEdge<C>> = Vec::new();
    let mut boundary_points: Vec<Vec<Point3>> = Vec::new();
    for wire in face.boundaries() {
        for edge in wire.edge_iter() {
            let points = oriented_points::<C, S>(&edge, tol);
            let from = vertex_of(&mut vertices, edge.front().point(), Some(edge.front()), tol);
            let to = vertex_of(&mut vertices, edge.back().point(), Some(edge.back()), tol);
            let flat: Vec<Point2> = points.iter().map(|p| frame.flat(*p)).collect();
            let n = flat.len();
            halves.push(HalfEdge {
                edge: edge.clone(),
                from,
                to,
                leave: direction(flat[0], flat[1]),
                arrive: direction(flat[n - 1], flat[n - 2]),
                flat,
            });
            boundary_points.push(points);
        }
    }
    let mut seen = std::collections::HashSet::new();
    for other in others {
        for edge in other.edge_iter() {
            if !seen.insert(edge.id()) {
                continue;
            }
            let curve = edge.curve();
            let points: Vec<Point3> = samples::<C, S>(&curve, tol)
                .into_iter()
                .map(|(_, p)| p)
                .collect();
            if boundary_points.iter().any(|b| same_place(b, &points, tol)) {
                continue;
            }
            let flat: Vec<Point2> = points.iter().map(|p| frame.flat(*p)).collect();
            let n = flat.len();
            let middle = if n > 2 {
                flat[n / 2]
            } else {
                flat[0] + (flat[1] - flat[0]) * 0.5
            };
            if on_outline(&own_outline, middle, tol) || !inside(&own_outline, middle) {
                continue;
            }
            let (front, back) = (edge.absolute_front().point(), edge.absolute_back().point());
            let from = vertex_of(&mut vertices, front, None, tol);
            let to = vertex_of(&mut vertices, back, None, tol);
            let made = Edge::new(&vertices[from], &vertices[to], curve);
            halves.push(HalfEdge {
                edge: made.clone(),
                from,
                to,
                leave: direction(flat[0], flat[1]),
                arrive: direction(flat[n - 1], flat[n - 2]),
                flat: flat.clone(),
            });
            let reversed: Vec<Point2> = flat.iter().rev().copied().collect();
            halves.push(HalfEdge {
                edge: made.inverse(),
                from: to,
                to: from,
                leave: direction(reversed[0], reversed[1]),
                arrive: direction(reversed[n - 1], reversed[n - 2]),
                flat: reversed,
            });
        }
    }
    let count = halves.len();
    let mut used = vec![false; count];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for start in 0..count {
        if used[start] {
            continue;
        }
        let mut cycle = vec![start];
        used[start] = true;
        let mut here = start;
        loop {
            let back = halves[here].arrive;
            let next = (0..count)
                .filter(|&k| halves[k].from == halves[here].to)
                .min_by(|&a, &b| {
                    let turn = |k: usize| {
                        let twin = halves[k].edge.id() == halves[here].edge.id();
                        if twin {
                            return std::f64::consts::TAU;
                        }
                        let mut t = back - halves[k].leave;
                        while t <= 1.0e-12 {
                            t += std::f64::consts::TAU;
                        }
                        while t > std::f64::consts::TAU {
                            t -= std::f64::consts::TAU;
                        }
                        t
                    };
                    turn(a).total_cmp(&turn(b))
                })?;
            if next == start {
                break;
            }
            if used[next] || cycle.len() > count {
                return None;
            }
            used[next] = true;
            cycle.push(next);
            here = next;
        }
        cycles.push(cycle);
    }
    let polygon = |cycle: &[usize]| -> Vec<Point2> {
        cycle
            .iter()
            .flat_map(|&k| {
                let flat = &halves[k].flat;
                flat[..flat.len() - 1].to_vec()
            })
            .collect()
    };
    let shapes: Vec<(Vec<usize>, Vec<Point2>, f64)> = cycles
        .into_iter()
        .map(|cycle| {
            let points = polygon(&cycle);
            let area = polygon_area(&points);
            (cycle, points, area)
        })
        .collect();
    let outers: Vec<usize> = (0..shapes.len())
        .filter(|&k| shapes[k].2 > tol * tol)
        .collect();
    let mut holes_of: Vec<Vec<usize>> = vec![Vec::new(); shapes.len()];
    for hole in (0..shapes.len()).filter(|&k| shapes[k].2 < -tol * tol) {
        let hole_points = &shapes[hole].1;
        let n = hole_points.len();
        let Some((a, b)) = (0..n)
            .map(|i| (hole_points[i], hole_points[(i + 1) % n]))
            .max_by(|x, y| (x.1 - x.0).magnitude().total_cmp(&(y.1 - y.0).magnitude()))
        else {
            continue;
        };
        let along = b - a;
        let left = Vector2::new(-along.y, along.x).normalize();
        let point = a + along * 0.5 + left * (along.magnitude() * 1.0e-3).max(tol * 4.0);
        let owner = outers
            .iter()
            .copied()
            .filter(|&o| inside(&[shapes[o].1.clone()], point))
            .min_by(|&a, &b| shapes[a].2.total_cmp(&shapes[b].2))?;
        holes_of[owner].push(hole);
    }
    let surface = face.surface();
    let mut out = Vec::new();
    for &o in &outers {
        let wires: Vec<Wire<Point3, C>> = std::iter::once(o)
            .chain(holes_of[o].iter().copied())
            .map(|k| {
                shapes[k]
                    .0
                    .iter()
                    .map(|&h| halves[h].edge.clone())
                    .collect()
            })
            .collect();
        let polygons: Vec<Vec<Point2>> = std::iter::once(o)
            .chain(holes_of[o].iter().copied())
            .map(|k| shapes[k].1.clone())
            .collect();
        let point = interior_point(&polygons, tol)?;
        let overlap = others.iter().position(|other| {
            let theirs = outline(other, &frame, tol);
            inside(&theirs, point) && !on_outline(&theirs, point, tol)
        });
        let absolute: Vec<Wire<Point3, C>> = match face.orientation() {
            true => wires,
            false => wires.iter().map(Wire::inverse).collect(),
        };
        let mut piece = Face::new_unchecked(absolute, surface.clone());
        if !face.orientation() {
            piece.invert();
        }
        out.push((piece, overlap));
    }
    Some(out)
}

pub(super) fn inner_point<C, S: ShapeOpsSurface>(
    face: &Face<Point3, C, S>,
    tol: f64,
) -> Option<Point3>
where
    C: Clone + BoundedCurve + ParameterDivision1D<Point = Point3>,
{
    let frame = Frame::new(plane_of(face)?);
    interior_point(&outline(face, &frame, tol), tol).map(|p| frame.lift(p))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Overlap {
    Same,
    Opposite,
}

pub(super) struct Imprinted<C, S> {
    pub(super) shell0: Shell<Point3, C, S>,
    pub(super) shell1: Shell<Point3, C, S>,
    pub(super) overlap0: HashMap<FaceId<S>, Overlap>,
    pub(super) overlap1: HashMap<FaceId<S>, Overlap>,
}

fn overlapping<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    a: &Face<Point3, C, S>,
    b: &Face<Point3, C, S>,
    frame: &Frame,
    tol: f64,
) -> bool {
    let (pa, pb) = (outline(a, frame, tol), outline(b, frame, tol));
    let strictly =
        |polygons: &[Vec<Point2>], p: Point2| inside(polygons, p) && !on_outline(polygons, p, tol);
    let probes = |polygons: &[Vec<Point2>]| -> Vec<Point2> {
        let mut points: Vec<Point2> = polygons.iter().flatten().copied().collect();
        for polygon in polygons {
            let n = polygon.len();
            for i in 0..n {
                let (a, b) = (polygon[i], polygon[(i + 1) % n]);
                let along = b - a;
                let length = along.magnitude();
                if length < tol {
                    continue;
                }
                let left = Vector2::new(-along.y, along.x) / length;
                for k in [0.25, 0.5, 0.75] {
                    for side in [1.0, -1.0] {
                        points.push(a + along * k + left * side * (length * 1.0e-2).max(tol * 4.0));
                    }
                }
            }
        }
        let corners: Vec<Point2> = polygons.iter().flatten().copied().collect();
        if let Some(first) = corners.first() {
            let (lo, hi) = corners.iter().fold((*first, *first), |(lo, hi), p| {
                (
                    Point2::new(lo.x.min(p.x), lo.y.min(p.y)),
                    Point2::new(hi.x.max(p.x), hi.y.max(p.y)),
                )
            });
            for i in 1..8 {
                for j in 1..8 {
                    points.push(Point2::new(
                        lo.x + (hi.x - lo.x) * i as f64 / 8.0,
                        lo.y + (hi.y - lo.y) * j as f64 / 8.0,
                    ));
                }
            }
        }
        points
            .into_iter()
            .filter(|p| inside(polygons, *p) || on_outline(polygons, *p, tol))
            .collect()
    };
    if probes(&pa)
        .into_iter()
        .any(|p| strictly(&pa, p) && strictly(&pb, p))
        || probes(&pb)
            .into_iter()
            .any(|p| strictly(&pb, p) && strictly(&pa, p))
    {
        return true;
    }
    let segments = |polygons: &[Vec<Point2>]| -> Vec<(Point2, Point2)> {
        polygons
            .iter()
            .flat_map(|poly| (0..poly.len()).map(move |i| (poly[i], poly[(i + 1) % poly.len()])))
            .collect()
    };
    let (sa, sb) = (segments(&pa), segments(&pb));
    sa.iter().any(|(a0, a1)| {
        sb.iter().any(|(b0, b1)| {
            crossing(*a0, *a1, *b0, *b1).is_some_and(|(s, t)| {
                s > 1.0e-6 && s < 1.0 - 1.0e-6 && t > 1.0e-6 && t < 1.0 - 1.0e-6
            })
        })
    })
}

pub(super) fn imprint<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell0: &Shell<Point3, C, S>,
    shell1: &Shell<Point3, C, S>,
    tol: f64,
) -> Option<Imprinted<C, S>> {
    let planes0: Vec<Option<PlaneOf>> = shell0.face_iter().map(plane_of).collect();
    let planes1: Vec<Option<PlaneOf>> = shell1.face_iter().map(plane_of).collect();
    let faces0: Vec<Face<Point3, C, S>> = shell0.face_iter().cloned().collect();
    let faces1: Vec<Face<Point3, C, S>> = shell1.face_iter().cloned().collect();
    let pairs: Vec<(usize, usize)> = (0..faces0.len())
        .flat_map(|i| (0..faces1.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| match (planes0[i], planes1[j]) {
            (Some(a), Some(b)) if coplanar(a, b, tol) => {
                overlapping(&faces0[i], &faces1[j], &Frame::new(a), tol)
            }
            _ => false,
        })
        .collect();
    if pairs.is_empty() {
        return None;
    }
    let (edges0, edges1) = (unique_edges(shell0), unique_edges(shell1));
    let (splits0, splits1) = split_points(&edges0, &edges1, &pairs, &faces0, &faces1, tol);
    let split0 = apply_splits(shell0, &edges0, &splits0);
    let split1 = apply_splits(shell1, &edges1, &splits1);
    let faces0: Vec<Face<Point3, C, S>> = split0.face_iter().cloned().collect();
    let faces1: Vec<Face<Point3, C, S>> = split1.face_iter().cloned().collect();
    let mut overlap0 = HashMap::default();
    let mut overlap1 = HashMap::default();
    let rebuild = |faces: &[Face<Point3, C, S>],
                   others: &[Face<Point3, C, S>],
                   partners: &dyn Fn(usize) -> Vec<usize>,
                   planes: &[Option<PlaneOf>],
                   other_planes: &[Option<PlaneOf>],
                   overlaps: &mut HashMap<FaceId<S>, Overlap>|
     -> Option<Shell<Point3, C, S>> {
        let mut shell = Shell::new();
        for (i, face) in faces.iter().enumerate() {
            let mine = partners(i);
            if mine.is_empty() {
                shell.push(face.clone());
                continue;
            }
            let theirs: Vec<&Face<Point3, C, S>> = mine.iter().map(|&j| &others[j]).collect();
            for (piece, over) in divide(face, &theirs, tol)? {
                if let Some(k) = over {
                    let same = match (planes[i], other_planes[mine[k]]) {
                        (Some(a), Some(b)) => a.normal.dot(b.normal) > 0.0,
                        _ => return None,
                    };
                    overlaps.insert(
                        piece.id(),
                        if same {
                            Overlap::Same
                        } else {
                            Overlap::Opposite
                        },
                    );
                }
                shell.push(piece);
            }
        }
        Some(shell)
    };
    let partners0 = |i: usize| pairs.iter().filter(|p| p.0 == i).map(|p| p.1).collect();
    let partners1 = |j: usize| pairs.iter().filter(|p| p.1 == j).map(|p| p.0).collect();
    let shell0 = rebuild(
        &faces0,
        &faces1,
        &partners0,
        &planes0,
        &planes1,
        &mut overlap0,
    )?;
    let shell1 = rebuild(
        &faces1,
        &faces0,
        &partners1,
        &planes1,
        &planes0,
        &mut overlap1,
    )?;
    Some(Imprinted {
        shell0,
        shell1,
        overlap0,
        overlap1,
    })
}

pub(super) fn sew<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &Shell<Point3, C, S>,
    tol: f64,
) -> Option<Shell<Point3, C, S>> {
    let compressed = shell.compress();
    let mut vertices: Vec<Point3> = Vec::new();
    let vertex_map: Vec<usize> = compressed
        .vertices
        .iter()
        .map(|p| {
            vertices
                .iter()
                .position(|q| q.distance(*p) < tol)
                .unwrap_or_else(|| {
                    vertices.push(*p);
                    vertices.len() - 1
                })
        })
        .collect();
    let middle = |curve: &C| {
        let (t0, t1) = curve.range_tuple();
        curve.subs((t0 + t1) / 2.0)
    };
    let mut edges: Vec<CompressedEdge<C>> = Vec::new();
    let edge_map: Vec<(usize, bool)> = compressed
        .edges
        .iter()
        .map(|edge| {
            let (a, b) = (vertex_map[edge.vertices.0], vertex_map[edge.vertices.1]);
            let m = middle(&edge.curve);
            edges
                .iter()
                .position(|known| {
                    (known.vertices == (a, b) || known.vertices == (b, a))
                        && middle(&known.curve).distance(m) < tol * 10.0
                })
                .map(|k| (k, edges[k].vertices == (a, b)))
                .unwrap_or_else(|| {
                    edges.push(CompressedEdge {
                        vertices: (a, b),
                        curve: edge.curve.clone(),
                    });
                    (edges.len() - 1, true)
                })
        })
        .collect();
    let faces = compressed
        .faces
        .into_iter()
        .map(|mut face| {
            face.boundaries.iter_mut().flatten().for_each(|use_| {
                let (index, same) = edge_map[use_.index];
                *use_ = CompressedEdgeIndex {
                    index,
                    orientation: use_.orientation == same,
                };
            });
            face
        })
        .collect();
    Shell::extract(CompressedShell {
        vertices,
        edges,
        faces,
        vertex_stable_ids: None,
        edge_stable_ids: None,
        face_stable_ids: None,
    })
    .ok()
}
