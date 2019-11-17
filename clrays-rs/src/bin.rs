extern crate clrays_rs;
use clrays_rs as clr;
//use clr::test_platform::PlatformTest;
use clr::window;
use clr::state;
use clr::scene::{Scene,Material,Plane,Sphere};
use clr::vec3::Vec3;

pub fn main(){
    //clr::test(PlatformTest::OpenCl1);
    let mut scene = Scene::new();
    scene.sky_col = Vec3::new(0.2, 0.2, 0.9).normalized();
    scene.sky_intensity = 0.0;
    scene.cam_pos = Vec3::zero();
    scene.cam_dir = Vec3::backward();
    scene.add_plane(Plane{
        pos: Vec3::down(),
        nor: Vec3::up(),
        mat: Material::basic(),
    });
    scene.add_sphere(Sphere{
        pos: Vec3::new(2.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic(),
    });

    let mut window = window::Window::<state::StdState>::new("ClRays", 960, 540);
    window.run(window::std_input_handler);
}
