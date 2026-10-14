//! Classic (0.3.2) boolean pipeline -- the self-contained default backend.
//!
//! This subtree grafts the proven upstream boolean assembly from the published
//! 0.3.2 crate. It is compiled only when the marching SSI is the active backend
//! (default features); a build with an external SSI backend never touches it
//! and runs the current `integrate` pipeline unchanged. The public entry points in
//! `transversal::integrate` cfg-dispatch `and`/`or`/`difference`/
//! `symmetric_difference` here.
//!
//! Graft-boundary adaptations vs. the 0.3.2 source (all documented inline):
//! - The Alternative intersection-curve arm is a raw
//!   `IntersectionCurve<PolylineCurve<Point3>, S, S>` (0.3.2 wrapped it with
//!   parameter polylines the pipeline never consumed).
//! - `altshell_to_shell` converts that arm back to `C` via
//!   `SurfaceCurve::with_boundaries(..).into()` (the conversion the current
//!   `ShapeOpsCurve` bound provides), not `IntersectionCurve::new(..).into()`.
//! - Failures map onto the current `ShapeOpsError` (`EmptyOutputShell` for a
//!   `None` pipeline result, `InvalidOutputShell` for a rejected solid), since
//!   the current error enum has no single `Internal` variant.

mod coplanar;
mod divide_face;
mod faces_classification;
mod intersection_curve;
mod loops_store;

use super::integrate::{ShapeOpsCurve, ShapeOpsError, ShapeOpsSurface};
use crate::alternative::Alternative;
use monstertruck_geometry::prelude::*;
use monstertruck_meshing::prelude::*;
use monstertruck_topology::*;
use std::iter;

type ClassicResult<T> = std::result::Result<T, ShapeOpsError>;

type AltCurve<C, S> = Alternative<C, IntersectionCurve<PolylineCurve<Point3>, S, S>>;
type AltCurveShell<C, S> = Shell<Point3, AltCurve<C, S>, S>;

/// Convert an Alternative-curve shell back to a target-curve shell, approximating
/// each intersection-curve arm with a polyline B-spline wrapped in a
/// `SurfaceCurve` (uses the `From<SurfaceCurve<BsplineCurve<Point3>, ..>>`
/// conversion the current `ShapeOpsCurve` bound guarantees).
fn altshell_to_shell<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    altshell: &AltCurveShell<C, S>,
) -> Option<Shell<Point3, C, S>> {
    altshell.try_mapped(
        |p| Some(*p),
        |c| match c {
            Alternative::FirstType(c) => Some(c.clone()),
            Alternative::SecondType(ic) => {
                let surface0 = ic.surface0().clone();
                let surface1 = ic.surface1().clone();
                let mut points = ic.leader().as_slice().to_vec();
                points.dedup_by(|lhs, rhs| (*lhs).near(&*rhs));
                if points.len() < 2 {
                    return None;
                }
                let denominator = (points.len() - 1) as f64;
                let knot_vec = KnotVector::from(
                    iter::once(0.0)
                        .chain((0..points.len()).map(|index| index as f64 / denominator))
                        .chain(iter::once(1.0))
                        .collect::<Vec<_>>(),
                );
                let bspline = BsplineCurve::new(knot_vec, points);
                let boundary0: Option<ParameterCurve<BoundaryCurve2D, S>> = None;
                let boundary1: Option<ParameterCurve<BoundaryCurve2D, S>> = None;
                Some(
                    SurfaceCurve::with_boundaries(
                        surface0, surface1, bspline, boundary0, boundary1,
                    )
                    .into(),
                )
            }
        },
        |s| Some(s.clone()),
    )
}

/// Split one pair of shells into `[and_shell, or_shell]` (0.3.2 verbatim): build
/// intersection loops, divide faces, classify each divided face as `and`/`or`,
/// then ray-cast the remaining `unknown` faces against the other shell.
fn process_one_pair_of_shells<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell0: &Shell<Point3, C, S>,
    shell1: &Shell<Point3, C, S>,
    tol: f64,
) -> Option<[Shell<Point3, C, S>; 2]> {
    if tol <= 0.0 {
        return None;
    }
    let imprinted = coplanar::imprint(shell0, shell1, tol);
    let (shell0, shell1) = match &imprinted {
        Some(found) => (&found.shell0, &found.shell1),
        None => (shell0, shell1),
    };
    let flat = imprinted.is_some();
    let poly_shell0 = shell0.triangulation(tol);
    let poly_shell1 = shell1.triangulation(tol);
    let altshell0: AltCurveShell<C, S> =
        shell0.mapped(|x| *x, |c| Alternative::FirstType(c.clone()), Clone::clone);
    let altshell1: AltCurveShell<C, S> =
        shell1.mapped(|x| *x, |c| Alternative::FirstType(c.clone()), Clone::clone);
    let loops_store::LoopsStoreQuadruple {
        geom_loops_store0: loops_store0,
        geom_loops_store1: loops_store1,
        ..
    } = loops_store::create_loops_stores(
        &altshell0,
        &poly_shell0,
        &altshell1,
        &poly_shell1,
        flat.then_some(tol),
    )?;
    let mut cls0 = divide_face::divide_faces(&altshell0, &loops_store0, tol)?;
    let mut cls1 = divide_face::divide_faces(&altshell1, &loops_store1, tol)?;
    let alt_ids = |shell: &Shell<Point3, C, S>,
                   alt: &AltCurveShell<C, S>,
                   marks: &rustc_hash::FxHashMap<FaceId<S>, coplanar::Overlap>| {
        shell
            .face_iter()
            .zip(alt.face_iter())
            .filter_map(|(face, mapped)| marks.get(&face.id()).map(|m| (mapped.id(), *m)))
            .collect::<rustc_hash::FxHashMap<_, _>>()
    };
    let (taken0, taken1) = match &imprinted {
        Some(found) => (
            cls0.take(&alt_ids(shell0, &altshell0, &found.overlap0)),
            cls1.take(&alt_ids(shell1, &altshell1, &found.overlap1)),
        ),
        None => (Vec::new(), Vec::new()),
    };
    cls0.integrate_by_component();
    cls1.integrate_by_component();
    let meshes: rustc_hash::FxHashMap<FaceId<S>, PolygonMesh> = match flat {
        true => altshell0
            .face_iter()
            .zip(poly_shell0.face_iter())
            .chain(altshell1.face_iter().zip(poly_shell1.face_iter()))
            .filter_map(|(face, poly)| Some((face.id(), poly.surface()?)))
            .collect(),
        false => Default::default(),
    };
    let probe = |face: &Face<Point3, AltCurve<C, S>, S>| -> Point3 {
        let from_mesh = || {
            let mesh = meshes.get(&face.id())?;
            let positions = mesh.positions();
            mesh.faces()
                .triangle_iter()
                .map(|t| t.map(|v| positions[v.pos]))
                .max_by(|a, b| {
                    let area = |t: &[Point3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).magnitude();
                    area(a).total_cmp(&area(b))
                })
                .map(|[a, b, c]| Point3::from_vec((a.to_vec() + b.to_vec() + c.to_vec()) / 3.0))
        };
        flat.then(|| coplanar::inner_point(face, tol).or_else(from_mesh))
            .flatten()
            .unwrap_or_else(|| face.boundaries()[0].vertex_iter().next().unwrap().point())
    };
    let inverted = |poly: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>| -> isize {
        isize::from(flat && poly.to_polygon().volume() < 0.0)
    };
    let (shift0, shift1) = (inverted(&poly_shell0), inverted(&poly_shell1));
    let boxes0 = face_boxes(&poly_shell0);
    let boxes1 = face_boxes(&poly_shell1);
    let rays = if flat { 5 } else { 1 };
    let [mut and0, mut or0, unknown0] = cls0.and_or_unknown();
    let (inside0, outside0) = split_inside(
        unknown0.into_iter().collect(),
        &probe,
        rays,
        &poly_shell1,
        &boxes1,
        shift1,
    )?;
    and0.extend(inside0);
    or0.extend(outside0);
    let [mut and1, mut or1, unknown1] = cls1.and_or_unknown();
    let (inside1, outside1) = split_inside(
        unknown1.into_iter().collect(),
        &probe,
        rays,
        &poly_shell0,
        &boxes0,
        shift0,
    )?;
    and1.extend(inside1);
    or1.extend(outside1);
    for (face, mark) in taken0 {
        if mark == coplanar::Overlap::Same {
            or0.push(face);
        }
    }
    for (face, mark) in taken1 {
        if mark == coplanar::Overlap::Same {
            and1.push(face);
        }
    }
    and0.append(&mut and1);
    or0.append(&mut or1);
    let (and, or) = (altshell_to_shell(&and0)?, altshell_to_shell(&or0)?);
    match flat {
        true => Some([coplanar::sew(&and, tol)?, coplanar::sew(&or, tol)?]),
        false => Some([and, or]),
    }
}

type PolyShell = Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>;

fn face_boxes(poly: &PolyShell) -> Vec<Option<BoundingBox<Point3>>> {
    poly.face_iter()
        .map(|face| {
            let mesh = face.surface()?;
            let mut bounds: BoundingBox<Point3> = mesh.positions().iter().collect();
            let pad = bounds.diameter() * 1.0e-9 + 1.0e-12;
            bounds.push(bounds.max() + Vector3::new(pad, pad, pad));
            bounds.push(bounds.min() - Vector3::new(pad, pad, pad));
            Some(bounds)
        })
        .collect()
}

fn ray_meets_box(bounds: &BoundingBox<Point3>, origin: Point3, direction: Vector3) -> bool {
    let (min, max) = (bounds.min(), bounds.max());
    let (mut near, mut far) = (0.0_f64, f64::INFINITY);
    for k in 0..3 {
        if direction[k] == 0.0 {
            if origin[k] < min[k] || origin[k] > max[k] {
                return false;
            }
            continue;
        }
        let (a, b) = (
            (min[k] - origin[k]) / direction[k],
            (max[k] - origin[k]) / direction[k],
        );
        near = near.max(a.min(b));
        far = far.min(a.max(b));
        if near > far {
            return false;
        }
    }
    true
}

fn crossings(
    poly: &PolyShell,
    boxes: &[Option<BoundingBox<Point3>>],
    point: Point3,
    direction: Vector3,
) -> Option<isize> {
    poly.face_iter()
        .zip(boxes)
        .try_fold(0, |count, (face, bounds)| {
            let mesh = face.surface()?;
            if bounds
                .as_ref()
                .is_some_and(|b| !ray_meets_box(b, point, direction))
            {
                return Some(count);
            }
            let crossed = mesh.signed_crossing_faces(point, direction);
            Some(
                count
                    + if face.orientation() {
                        crossed
                    } else {
                        -crossed
                    },
            )
        })
}

#[allow(clippy::type_complexity)]
fn split_inside<F: Send + Sync, P: Fn(&F) -> Point3 + Sync>(
    unknown: Vec<F>,
    probe: &P,
    rays: usize,
    other: &PolyShell,
    boxes: &[Option<BoundingBox<Point3>>],
    shift: isize,
) -> Option<(Vec<F>, Vec<F>)> {
    use rayon::prelude::*;
    let verdicts = unknown
        .par_iter()
        .map(|face| {
            let pt = probe(face);
            let mut inside = 0;
            for k in 0..rays {
                let dir = hash::take_one_unit(
                    pt + Vector3::new(k as f64 * 0.37, k as f64 * 0.11, k as f64 * 0.23),
                );
                if crossings(other, boxes, pt, dir)? + shift >= 1 {
                    inside += 1;
                }
            }
            Some(inside * 2 > rays)
        })
        .collect::<Option<Vec<bool>>>()?;
    let (mut ins, mut outs) = (Vec::new(), Vec::new());
    for (face, verdict) in unknown.into_iter().zip(verdicts) {
        match verdict {
            true => ins.push(face),
            false => outs.push(face),
        }
    }
    Some((ins, outs))
}

fn finalize<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    operation: &'static str,
    shell: Shell<Point3, C, S>,
) -> ClassicResult<Solid<Point3, C, S>> {
    let boundaries = shell.connected_components();
    Solid::try_new(boundaries)
        .map_err(|source| ShapeOpsError::InvalidOutputShell { operation, source })
}

/// AND operation between two solids (classic backend).
pub(crate) fn and<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> ClassicResult<Solid<Point3, C, S>> {
    let operation = "and";
    let pair = |a: &Shell<Point3, C, S>, b: &Shell<Point3, C, S>| {
        process_one_pair_of_shells(a, b, tol).ok_or(ShapeOpsError::EmptyOutputShell { operation })
    };
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let [mut and_shell, _] = pair(shell0, shell1)?;
    for shell in iter0 {
        let [res, _] = pair(&and_shell, shell)?;
        and_shell = res;
    }
    for shell in iter1 {
        let [res, _] = pair(&and_shell, shell)?;
        and_shell = res;
    }
    finalize(operation, and_shell)
}

/// OR operation between two solids (classic backend).
pub(crate) fn or<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> ClassicResult<Solid<Point3, C, S>> {
    let operation = "or";
    let pair = |a: &Shell<Point3, C, S>, b: &Shell<Point3, C, S>| {
        process_one_pair_of_shells(a, b, tol).ok_or(ShapeOpsError::EmptyOutputShell { operation })
    };
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let [_, mut or_shell] = pair(shell0, shell1)?;
    for shell in iter0 {
        let [_, res] = pair(&or_shell, shell)?;
        or_shell = res;
    }
    for shell in iter1 {
        let [_, res] = pair(&or_shell, shell)?;
        or_shell = res;
    }
    finalize(operation, or_shell)
}

/// Difference: the region inside `solid0` but outside `solid1` (classic backend).
pub(crate) fn difference<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> ClassicResult<Solid<Point3, C, S>> {
    let mut neg = solid1.clone();
    neg.not();
    and(solid0, &neg, tol)
}

/// Symmetric difference (XOR): the region inside exactly one solid (classic backend).
pub(crate) fn symmetric_difference<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> ClassicResult<Solid<Point3, C, S>> {
    let d0 = difference(solid0, solid1, tol)?;
    let d1 = difference(solid1, solid0, tol)?;
    or(&d0, &d1, tol)
}
