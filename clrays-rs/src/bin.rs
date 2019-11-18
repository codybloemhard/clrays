extern crate clrays_rs;
use clrays_rs as clr;
//use clr::test_platform::PlatformTest;
use clr::window;
use clr::state;
use clr::scene::{Scene,Material,Plane,Sphere};
use clr::vec3::Vec3;
use clr::kernels::{VoidKernel,ResultKernel,TraceKernel};
use clr::cl_helpers;

pub fn main() -> Result<(),String>{
    //clr::test(PlatformTest::OpenCl1);
    let mut scene = Scene::new();
    scene.sky_col = Vec3::new(0.2, 0.2, 0.9).normalized();
    scene.sky_intensity = 0.0;
    scene.cam_pos = Vec3::zero();
    scene.cam_dir = Vec3::backward();
    scene.add_texture("sky".to_string(), "../Assets/Textures/sky1.jpg".to_string(), clr::trace_tex::TexType::Vector3c8bpc);
    scene.add_texture("wood".to_string(), "../Assets/Textures/wood.png".to_string(), clr::trace_tex::TexType::Vector3c8bpc);
    scene.set_skybox("wood");
    scene.add_plane(Plane{
        pos: Vec3::zero(),
        nor: Vec3::down(),
        mat: Material::basic(),
    });
    scene.add_sphere(Sphere{
        pos: Vec3::new(0.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic(),
    });
    scene.update();

    use std::io::prelude::*;
    let mut file = std::fs::File::open("../Assets/Kernels/raytrace.cl").unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).expect("file to string aaah broken");

    let platform = ocl::Platform::default();
    let device = match ocl::Device::first(platform){
        Ok(x) => x,
        Err(_) => return Err("Error: device".to_string()),
    };
    let context = match ocl::Context::builder()
    .platform(platform)
    .devices(device.clone())
    .build(){
        Ok(x) => x,
        Err(_) => return Err("Error: context".to_string()),
    };
    let program = match ocl::Program::builder()
    .devices(device)
    .src(src)
    .build(&context){
        Ok(x) => x,
        Err(_) => return Err("Error: program".to_string()),
    };
    let queue = match ocl::Queue::new(&context, device, None){
        Ok(x) => x,
        Err(_) => return Err("Error: queue".to_string()),
    };
    let mut trace_kernel = match TraceKernel::new("raytracing", (960,540), &program, &queue, &mut scene){
        Ok(x) => x,
        Err(e) => return Err("Error: kernel new".to_string()),
    };
    trace_kernel.update(&queue).expect("expect: kernel update");
    trace_kernel.execute(&queue).expect("expect: kernel execute");
    let tex = match trace_kernel.get_result(&queue){
        Ok(x) => x,
        Err(e) => return Err("Error: kernel result".to_string()),
    };

    let mut window = window::Window::<state::StdState>::new("ClRays", 960, 540);
    window.run(window::std_input_handler, Some(tex));
    Ok(())
}
