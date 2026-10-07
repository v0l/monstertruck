use algo::curve::{presearch_closest_point, search_closest_parameter};
use monstertruck_geometry::prelude::*;

use super::error::FilletError;
use super::params::FilletProfile;
use super::topology::*;
use super::types::*;

type Result<T> = std::result::Result<T, FilletError>;

const MITER_SPANS: usize = 8;
const SMOOTH_COSINE: f64 = 0.999_999;
const MEET_TOLERANCE: f64 = 1.0e-4;

fn failed(context: &'static str) -> FilletError { FilletError::GeometryFailed { context } }

fn is_smooth(incoming: &Edge, outgoing: &Edge) -> bool {
    let incoming = incoming.oriented_curve();
    let outgoing = outgoing.oriented_curve();
    let end = incoming.der(incoming.range_tuple().1).normalize();
    let start = outgoing.der(outgoing.range_tuple().0).normalize();
    end.dot(start) > SMOOTH_COSINE
}

struct Junction {
    shared: Vertex,
    side: Vertex,
    edge: Edge,
    incoming: (f64, f64),
    outgoing: (f64, f64),
}

fn closest<C0, C1>(curve0: &C0, curve1: &C1) -> Option<(f64, f64)>
where
    C0: ParametricCurve3D + BoundedCurve,
    C1: ParametricCurve3D + BoundedCurve, {
    let hint = presearch_closest_point(
        curve0,
        curve1,
        (curve0.range_tuple(), curve1.range_tuple()),
        50,
    );
    let (t0, t1) = search_closest_parameter(curve0, curve1, hint, 100)?;
    (curve0.subs(t0).distance(curve1.subs(t1)) < MEET_TOLERANCE).then_some((t0, t1))
}

fn signed_distance(
    point: Point3,
    surface: &NurbsSurface<Vector4>,
    hint: &mut Option<(f64, f64)>,
) -> Option<f64> {
    let (u, v) = surface.search_nearest_parameter(point, *hint, 100)?;
    *hint = Some((u, v));
    Some((point - surface.subs(u, v)).dot(surface.normal(u, v)))
}

fn meet_along_row(
    surface: &NurbsSurface<Vector4>,
    u: f64,
    guess: f64,
    other: &NurbsSurface<Vector4>,
    hint: &mut Option<(f64, f64)>,
) -> Option<f64> {
    let (_, (v_low, v_high)) = surface.range_tuple();
    let mut f = |v: f64| signed_distance(surface.subs(u, v), other, hint);
    let (mut v0, mut v1) = (guess, guess + (v_high - v_low) * 1.0e-3);
    let (mut f0, mut f1) = (f(v0)?, f(v1)?);
    for _ in 0..64 {
        if f1.abs() < TOLERANCE * 1.0e-3 {
            return Some(v1);
        }
        let slope = f1 - f0;
        if slope.abs() < f64::MIN_POSITIVE {
            return None;
        }
        let v2 = v1 - f1 * (v1 - v0) / slope;
        (v0, f0, v1) = (v1, f1, v2);
        f1 = f(v1)?;
    }
    (f1.abs() < TOLERANCE).then_some(v1)
}

fn interpolate(points: Vec<Point3>) -> Option<NurbsCurve<Vector4>> {
    let count = points.len();
    let degree = 3.min(count - 1);
    let knot_vector = KnotVector::uniform_knot(degree, count - degree);
    let knots: Vec<f64> = knot_vector.iter().copied().collect();
    let parameter_points: Vec<(f64, Point3)> = points
        .into_iter()
        .enumerate()
        .map(|(i, point)| {
            let greville = knots[i + 1..=i + degree].iter().sum::<f64>() / degree as f64;
            (greville, point)
        })
        .collect();
    BsplineCurve::try_interpolate(knot_vector, parameter_points)
        .ok()
        .map(NurbsCurve::from)
}

fn trim(mut curve: NurbsCurve<Vector4>, t0: f64, t1: f64) -> NurbsCurve<Vector4> {
    let (low, high) = curve.range_tuple();
    if t1 < high - TOLERANCE {
        curve.cut(t1);
    }
    if t0 > low + TOLERANCE {
        curve = curve.cut(t0);
    }
    curve
}

fn seam_at(side_face: &Face, wire_edge: &Edge) -> Result<Edge> {
    let corner = wire_edge.front();
    side_face
        .edge_iter()
        .find(|edge| !edge.is_same(wire_edge) && (edge.front() == corner || edge.back() == corner))
        .ok_or(failed("seam edge at chain junction"))
}

fn cut_seam(
    seam: &Edge,
    corner: &Vertex,
    point: Point3,
    replacements: &mut EdgeReplacements,
) -> Result<Vertex> {
    let curve = seam.oriented_curve();
    let t = curve
        .search_nearest_parameter(point, None, 100)
        .ok_or(failed("project contact point onto seam"))?;
    let vertex = Vertex::new(curve.subs(t));
    let (head, tail) = seam
        .not_strictly_cut(&vertex)
        .ok_or(failed("cut seam edge at contact point"))?;
    replacements.insert(seam, if seam.front() == corner { tail } else { head });
    Ok(vertex)
}

fn smooth_junction(
    incoming: &NurbsSurface<Vector4>,
    outgoing: &NurbsSurface<Vector4>,
    seam: &Edge,
    corner: &Vertex,
    replacements: &mut EdgeReplacements,
) -> Result<Junction> {
    let last_row = outgoing.control_points().len() - 1;
    let (_, (_, incoming_end)) = incoming.range_tuple();
    let (_, (outgoing_start, _)) = outgoing.range_tuple();
    let shared = Vertex::new(outgoing.curve_v(0).front());
    let side = cut_seam(
        seam,
        corner,
        outgoing.curve_v(last_row).front(),
        replacements,
    )?;
    let edge = Edge::new(&shared, &side, outgoing.curve_u(0).into());
    Ok(Junction {
        shared,
        side,
        edge,
        incoming: (incoming_end, incoming_end),
        outgoing: (outgoing_start, outgoing_start),
    })
}

fn miter_junction(
    incoming: &NurbsSurface<Vector4>,
    outgoing: &NurbsSurface<Vector4>,
    seam: &Edge,
    corner: &Vertex,
    replacements: &mut EdgeReplacements,
) -> Result<Junction> {
    let last_row = incoming.control_points().len() - 1;
    let not_meeting = || failed("fillets do not meet at corner");
    let (a0, b0) = closest(&incoming.curve_v(0), &outgoing.curve_v(0)).ok_or_else(not_meeting)?;
    let seam_curve = seam.oriented_curve();
    let (a1, seam_a) = closest(&incoming.curve_v(last_row), &seam_curve).ok_or_else(not_meeting)?;
    let (b1, seam_b) = closest(&outgoing.curve_v(last_row), &seam_curve).ok_or_else(not_meeting)?;

    let shared_point = incoming
        .curve_v(0)
        .subs(a0)
        .midpoint(outgoing.curve_v(0).subs(b0));
    let side_point = seam_curve.subs(seam_a).midpoint(seam_curve.subs(seam_b));
    let shared = Vertex::new(shared_point);
    let side = cut_seam(seam, corner, side_point, replacements)?;

    let ((u_low, u_high), _) = incoming.range_tuple();
    let mut hint = None;
    let interior = (1..MITER_SPANS)
        .map(|j| {
            let s = j as f64 / MITER_SPANS as f64;
            let u = u_low + (u_high - u_low) * s;
            let guess = a0 + (a1 - a0) * s;
            meet_along_row(incoming, u, guess, outgoing, &mut hint).map(|v| incoming.subs(u, v))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(not_meeting)?;
    let points = std::iter::once(shared_point)
        .chain(interior)
        .chain(std::iter::once(side.point()))
        .collect();
    let curve = interpolate(points).ok_or(failed("interpolate miter curve"))?;
    let edge = Edge::new(&shared, &side, curve.into());
    Ok(Junction {
        shared,
        side,
        edge,
        incoming: (a0, a1),
        outgoing: (b0, b1),
    })
}

fn average_seam(previous: &mut NurbsSurface<Vector4>, next: &mut NurbsSurface<Vector4>) {
    (0..next.control_points().len()).for_each(|j| {
        let len = previous.control_points()[j].len();
        let p = *previous.control_point(j, len - 1);
        let q = *next.control_point(j, 0);
        let c = (p + q) / 2.0;
        *previous.control_point_mut(j, len - 1) = c;
        *next.control_point_mut(j, 0) = c;
    });
}

pub(super) fn fillet_closed_chain(
    shell: &mut Shell,
    wire: &Wire,
    shared_face_index: FaceBoundaryEdgeIndex,
    adjacent_faces: &[FaceBoundaryEdgeIndex],
    radius: impl Fn(f64) -> f64,
    division: usize,
    profile: &FilletProfile,
) -> Result<()> {
    let n = wire.len();
    let previous = |k: usize| (k + n - 1) % n;
    let next = |k: usize| (k + 1) % n;
    let smooth: Vec<bool> = (0..n)
        .map(|k| is_smooth(&wire[previous(k)], &wire[k]))
        .collect();
    let extensions: Vec<(bool, bool)> = (0..n).map(|k| (!smooth[k], !smooth[next(k)])).collect();

    let mut surfaces = fillet_surfaces_with_extensions(
        shell,
        wire,
        shared_face_index,
        adjacent_faces,
        radius,
        division,
        profile,
        &extensions,
    )
    .ok_or(FilletError::FilletSurfaceComputationFailed)?;
    (0..n).filter(|&k| smooth[k]).for_each(|k| {
        let p = previous(k);
        if p == k {
            return;
        }
        let (low, high) = if p < k {
            surfaces.split_at_mut(k)
        } else {
            surfaces.split_at_mut(p)
        };
        let (prev_surface, next_surface) = if p < k {
            (&mut low[p], &mut high[0])
        } else {
            (&mut high[0], &mut low[k])
        };
        average_seam(prev_surface, next_surface);
    });

    let mut side_replacements = EdgeReplacements::default();
    let junctions: Vec<Junction> = (0..n)
        .map(|k| {
            let seam = seam_at(&shell[adjacent_faces[k].face_index], &wire[k])?;
            let corner = wire[k].front();
            let (incoming, outgoing) = (&surfaces[previous(k)], &surfaces[k]);
            match smooth[k] {
                true => smooth_junction(incoming, outgoing, &seam, corner, &mut side_replacements),
                false => miter_junction(incoming, outgoing, &seam, corner, &mut side_replacements),
            }
        })
        .collect::<Result<_>>()?;

    let last_row = surfaces[0].control_points().len() - 1;
    let segment = |k: usize, row: usize, pick: fn(&(f64, f64)) -> f64| {
        let (start, end) = (&junctions[k], &junctions[next(k)]);
        trim(
            surfaces[k].curve_v(row),
            pick(&start.outgoing),
            pick(&end.incoming),
        )
    };
    let shared_edges: Vec<Edge> = (0..n)
        .map(|k| {
            let curve = segment(k, 0, |params| params.0);
            Edge::new(
                &junctions[k].shared,
                &junctions[next(k)].shared,
                curve.into(),
            )
        })
        .collect();
    let side_edges: Vec<Edge> = (0..n)
        .map(|k| {
            let curve = segment(k, last_row, |params| params.1);
            Edge::new(&junctions[k].side, &junctions[next(k)].side, curve.into())
        })
        .collect();

    let mut shared_replacements = EdgeReplacements::default();
    (0..n).for_each(|k| {
        shared_replacements.insert(&wire[k], shared_edges[k].clone());
        side_replacements.insert(&wire[k], side_edges[k].clone());
    });

    let fillet_faces: Vec<Face> = (0..n)
        .map(|k| {
            let boundary: Wire = vec![
                shared_edges[k].inverse(),
                junctions[k].edge.clone(),
                side_edges[k].clone(),
                junctions[next(k)].edge.inverse(),
            ]
            .into();
            Face::new_unchecked(vec![boundary], surfaces[k].clone())
        })
        .collect();

    let shared_face = shared_face_index.face_index;
    shell[shared_face] = shared_replacements.apply(&shell[shared_face]);
    let side_faces: crate::HashSet<usize> = adjacent_faces
        .iter()
        .map(|index| index.face_index)
        .collect();
    side_faces.into_iter().for_each(|face| {
        shell[face] = side_replacements.apply(&shell[face]);
    });
    shell.extend(fillet_faces);
    Ok(())
}
