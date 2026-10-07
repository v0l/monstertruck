use monstertruck_geometry::prelude::*;
use proptest::prelude::*;
use std::f64::consts::PI;

proptest! {
    #[test]
    fn sphere_case(t in 0f64..=1.0) {
        let sphere0 = Sphere::new(Point3::new(0.0, 0.0, 1.0), f64::sqrt(2.0));
        let sphere1 = Sphere::new(Point3::new(0.0, 0.0, -1.0), f64::sqrt(2.0));
        let bsp = BsplineCurve::new(
            KnotVector::bezier_knot(2),
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 2.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
            ],
        );
        let curve = IntersectionCurve::new(sphere0, sphere1, bsp);
        let p = curve.subs(t);
        let v = curve.der(t);

        prop_assert_near!(p.to_vec().magnitude(), 1.0);
        prop_assert!(p.to_vec().dot(v).so_small());

        let t0 = match curve.search_parameter(p, None, 100) {
            Some(t0) => t0,
            None => {
                let reason = "search_parameter failed".into();
                return Err(TestCaseError::Fail(reason))
            }
        };
        prop_assert_near!(t, t0);
    }

    #[test]
    fn cylinder_case(t in 0.0..=2.0 * PI, n in 0usize..=4) {
        let line0 = Line(Point3::new(1.0, 0.0, 2.0), Point3::new(-1.0, 0.0, 2.0));
        let cylinder0 = RevolutionSurface::by_revolution(line0, Point3::origin(), Vector3::unit_x());
        let line1 = Line(Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 0.0, -1.0));
        let cylinder1 = RevolutionSurface::by_revolution(line1, Point3::origin(), Vector3::unit_z());
        let z = (1.0 + f64::sqrt(3.0)) / 2.0;
        let lead_circle = Processor::with_transform(
            UnitCircle::<Point3>::new(),
            Matrix4::from_translation(z * Vector3::unit_z()),
        );
        let curve = IntersectionCurve::new(cylinder0, cylinder1, lead_circle);

        let p = curve.subs(t);
        prop_assert_near!(p.x * p.x + p.y * p.y, 1.0);
        prop_assert_near!(p.z * p.z + p.y * p.y, 4.0);

        let t0 = match curve.search_parameter(p, None, 100) {
            Some(t0) => t0,
            None => {
                let reason = "search_parameter failed".into();
                return Err(TestCaseError::Fail(reason));
            }
        };
        let diff = (t - t0).abs();
        prop_assert!(diff.near(&0.0) || diff.near(&(2.0 * PI)));

        const EPS: f64 = 1.0e-4;
        let v0 = curve.der_n(n + 1, t);
        let v1 = (curve.der_n(n, t + EPS) - curve.der_n(n, t - EPS)) / (2.0 * EPS);
        prop_assert!((v0 - v1).magnitude() < EPS * 10.0, "{v0:?} {v1:?}");

        let ders0 = (0..=n).map(|i| curve.der_n(i, t)).collect::<Vec<_>>();
        let ders1 = curve.ders(n, t);

        prop_assert_eq!(ders0.len(), ders1.len());
        let mut iter = ders0.into_iter().zip(&*ders1);
        iter.try_for_each(|(v0, v1)| {
            prop_assert_near!(v0, v1);
            Ok(())
        })?;
    }
}

fn spheres() -> (Sphere, Sphere) {
    (
        Sphere::new(Point3::new(0.0, 0.0, 1.0), f64::sqrt(2.0)),
        Sphere::new(Point3::new(0.0, 0.0, -1.0), f64::sqrt(2.0)),
    )
}

#[test]
fn division_keeps_an_exact_polyline_leader() {
    let (sphere0, sphere1) = spheres();
    let corners: Vec<Point3> = (0..=32)
        .map(|i| {
            let t = PI * i as f64 / 32.0;
            Point3::new(t.cos(), t.sin(), 0.0)
        })
        .collect();
    let leader = BsplineCurve::new(KnotVector::uniform_knot(1, 32), corners.clone());
    let curve = IntersectionCurve::new(sphere0, sphere1, leader);
    let (params, points) = curve.parameter_division((0.0, 1.0), 1.0e-3);
    assert_eq!(params.len(), points.len());
    assert!(points.len() <= corners.len());
    for point in &points {
        assert!((point.to_vec().magnitude() - 1.0).abs() < 1.0e-9 && point.z.abs() < 1.0e-9);
    }
    assert!(
        points
            .iter()
            .all(|p| corners.iter().any(|c| c.distance(*p) < 1.0e-12))
    );
}

#[test]
fn division_leaves_a_rough_leader() {
    let (sphere0, sphere1) = spheres();
    let leader = BsplineCurve::new(
        KnotVector::bezier_knot(2),
        vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
        ],
    );
    let curve = IntersectionCurve::new(sphere0, sphere1, leader);
    let (_, points) = curve.parameter_division((0.0, 1.0), 1.0e-3);
    assert!(points.len() > 3);
    for point in &points {
        assert!(
            (point.to_vec().magnitude() - 1.0).abs() < 1.0e-6,
            "{point:?}"
        );
    }
}
