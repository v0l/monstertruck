use monstertruck_meshing::prelude::*;
use monstertruck_modeling::*;
use std::f64::consts::PI;

fn cylinder(center: Point3, radius: f64, height: f64) -> Solid {
    let seed = builder::vertex(center + Vector3::unit_x() * radius);
    let rim = builder::revolve(
        &seed,
        center,
        Vector3::unit_z(),
        builder::SweepAngle::Closed,
        4,
    );
    let base = builder::try_attach_plane(&[rim]).unwrap();
    builder::extrude(&base, Vector3::unit_z() * height)
}

#[test]
fn a_long_cylinder_is_not_meshed_as_a_square_grid() {
    let rod = cylinder(Point3::origin(), 10.0, 900.0);
    let mesh = rod.triangulation(0.01).to_polygon();
    let triangles = mesh.faces().triangle_iter().count();
    assert!(triangles < 20_000, "{triangles} triangles");
    let expected = PI * 100.0 * 900.0;
    let volume = mesh.volume();
    assert!(
        (volume - expected).abs() < expected * 1.0e-3,
        "volume {volume}, expected {expected}"
    );
}

#[test]
fn a_long_tube_is_bored_through() {
    let rod = cylinder(Point3::origin(), 10.0, 900.0);
    let tool = cylinder(Point3::new(0.0, 0.0, -10.0), 8.5, 920.0);
    let tube = monstertruck_solid::difference_normalized(&rod, &tool).expect("bore");
    let expected = PI * (100.0 - 72.25) * 900.0;
    let volume = tube.triangulation(0.01).to_polygon().volume();
    assert!(
        (volume - expected).abs() < expected * 1.0e-3,
        "volume {volume}, expected {expected}"
    );
}
